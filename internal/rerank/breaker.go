package rerank

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// BreakerState is a snapshot of a Breaker's current state suitable for
// status reporting. OpenUntil is zero when the breaker is closed.
type BreakerState struct {
	// Open reports whether the breaker is currently rejecting calls.
	Open bool
	// OpenUntil is the wall-clock time when an open breaker will trip
	// back to closed (half-open in spirit — the next call probes the
	// inner endpoint). Zero when Open is false.
	OpenUntil time.Time
	// ConsecutiveFails is the count of back-to-back reachability
	// failures observed on the inner reranker. Resets on the first
	// success. Visible even while closed so `dex index status` can
	// show a trending failure rate.
	ConsecutiveFails int
}

// Breaker decorates a HealthChecker (Reranker + Health + Endpoint +
// ModelName) with a consecutive-failure circuit breaker. After
// `Threshold` back-to-back ErrUnreachable failures, the breaker opens
// for `OpenFor` and short-circuits Rerank/Health with ErrUnreachable so
// the caller's existing fallback path triggers immediately, without
// extending the request budget waiting on a sick endpoint.
//
// Failures that aren't reachability-shaped (4xx, decode errors) do not
// trip the breaker — those are configuration bugs we'd rather surface
// every time than mask under a degraded mode.
type Breaker struct {
	Inner     HealthChecker
	Threshold int           // default 3
	OpenFor   time.Duration // default 30s

	mu               sync.Mutex
	openUntil        time.Time
	consecutiveFails int
}

// NewBreaker wraps inner with a breaker using the given threshold and
// open duration. threshold <= 0 → 3; openFor <= 0 → 30s.
func NewBreaker(inner HealthChecker, threshold int, openFor time.Duration) *Breaker {
	if threshold <= 0 {
		threshold = 3
	}
	if openFor <= 0 {
		openFor = 30 * time.Second
	}
	return &Breaker{Inner: inner, Threshold: threshold, OpenFor: openFor}
}

// Endpoint passes through to the inner reranker.
func (b *Breaker) Endpoint() string { return b.Inner.Endpoint() }

// ModelName passes through to the inner reranker.
func (b *Breaker) ModelName() string { return b.Inner.ModelName() }

// Rerank short-circuits with ErrUnreachable when the breaker is open;
// otherwise delegates and records the outcome.
func (b *Breaker) Rerank(ctx context.Context, query string, docs []string) ([]Score, error) {
	if b.isOpen() {
		return nil, fmt.Errorf("%w: breaker open", ErrUnreachable)
	}
	scores, err := b.Inner.Rerank(ctx, query, docs)
	b.record(err)
	return scores, err
}

// Health short-circuits with ErrUnreachable when the breaker is open;
// otherwise delegates and records the outcome. This means `dex index
// status` reports the rerank backend as unreachable for the duration
// of the open window — the operator sees the breaker state in the
// status output rather than a stale "reachable=true".
func (b *Breaker) Health(ctx context.Context) error {
	if b.isOpen() {
		return fmt.Errorf("%w: breaker open", ErrUnreachable)
	}
	err := b.Inner.Health(ctx)
	b.record(err)
	return err
}

// State snapshots the breaker for status reporting.
func (b *Breaker) State() BreakerState {
	b.mu.Lock()
	defer b.mu.Unlock()
	st := BreakerState{ConsecutiveFails: b.consecutiveFails}
	if now := time.Now(); !b.openUntil.IsZero() && now.Before(b.openUntil) {
		st.Open = true
		st.OpenUntil = b.openUntil
	}
	return st
}

func (b *Breaker) isOpen() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.openUntil.IsZero() {
		return false
	}
	if time.Now().Before(b.openUntil) {
		return true
	}
	// Window elapsed — reset to half-open (the next call probes).
	b.openUntil = time.Time{}
	b.consecutiveFails = 0
	return false
}

func (b *Breaker) record(err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err == nil {
		b.consecutiveFails = 0
		b.openUntil = time.Time{}
		return
	}
	// Only reachability-shaped failures advance the counter.
	if !errors.Is(err, ErrUnreachable) {
		return
	}
	b.consecutiveFails++
	if b.consecutiveFails >= b.Threshold {
		b.openUntil = time.Now().Add(b.OpenFor)
	}
}
