package rerank

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"
)

// stubInner is a HealthChecker that returns canned outcomes for Rerank
// and Health, recording call counts so tests can assert pass-through
// vs short-circuit behavior.
type stubInner struct {
	rerankErr   error
	healthErr   error
	rerankCalls int
	healthCalls int
}

func (s *stubInner) Rerank(_ context.Context, _ string, _ []string) ([]Score, error) {
	s.rerankCalls++
	return nil, s.rerankErr
}
func (s *stubInner) Health(_ context.Context) error {
	s.healthCalls++
	return s.healthErr
}
func (*stubInner) Endpoint() string  { return "stub" }
func (*stubInner) ModelName() string { return "stub-model" }

func TestBreakerTripsAfterThreshold(t *testing.T) {
	inner := &stubInner{rerankErr: fmt.Errorf("%w: down", ErrUnreachable)}
	b := NewBreaker(inner, 3, 30*time.Second)

	for i := 0; i < 3; i++ {
		_, err := b.Rerank(context.Background(), "q", []string{"d"})
		if !errors.Is(err, ErrUnreachable) {
			t.Fatalf("call %d err = %v, want ErrUnreachable", i, err)
		}
	}
	if inner.rerankCalls != 3 {
		t.Errorf("inner.rerankCalls = %d, want 3", inner.rerankCalls)
	}

	// Fourth call should short-circuit without reaching the inner client.
	_, err := b.Rerank(context.Background(), "q", []string{"d"})
	if !errors.Is(err, ErrUnreachable) {
		t.Fatalf("after-trip err = %v, want ErrUnreachable", err)
	}
	if inner.rerankCalls != 3 {
		t.Errorf("after-trip inner.rerankCalls = %d, want 3 (short-circuited)", inner.rerankCalls)
	}

	st := b.State()
	if !st.Open {
		t.Errorf("State.Open = false, want true")
	}
	if st.ConsecutiveFails != 3 {
		t.Errorf("State.ConsecutiveFails = %d, want 3", st.ConsecutiveFails)
	}
}

func TestBreakerSuccessResetsCounter(t *testing.T) {
	inner := &stubInner{rerankErr: fmt.Errorf("%w: down", ErrUnreachable)}
	b := NewBreaker(inner, 3, 30*time.Second)

	// Two failures.
	_, _ = b.Rerank(context.Background(), "q", []string{"d"})
	_, _ = b.Rerank(context.Background(), "q", []string{"d"})

	// One success.
	inner.rerankErr = nil
	if _, err := b.Rerank(context.Background(), "q", []string{"d"}); err != nil {
		t.Fatalf("success call err = %v", err)
	}
	if got := b.State().ConsecutiveFails; got != 0 {
		t.Errorf("ConsecutiveFails = %d, want 0", got)
	}

	// Three more reachability failures still needed to trip.
	inner.rerankErr = fmt.Errorf("%w: down", ErrUnreachable)
	for i := 0; i < 3; i++ {
		_, _ = b.Rerank(context.Background(), "q", []string{"d"})
	}
	if !b.State().Open {
		t.Errorf("expected breaker open after counter reset + 3 fails")
	}
}

func TestBreakerSkipsNonReachabilityErrors(t *testing.T) {
	// A 4xx-shaped error should not advance the counter — those are
	// configuration bugs we want surfaced every call, not masked.
	inner := &stubInner{rerankErr: errors.New("rerank: http 400: bad request")}
	b := NewBreaker(inner, 3, 30*time.Second)

	for i := 0; i < 10; i++ {
		_, _ = b.Rerank(context.Background(), "q", []string{"d"})
	}
	if got := b.State().ConsecutiveFails; got != 0 {
		t.Errorf("ConsecutiveFails = %d, want 0 (4xx should not count)", got)
	}
	if b.State().Open {
		t.Errorf("breaker opened on non-reachability errors")
	}
}

func TestBreakerReopensAfterWindow(t *testing.T) {
	inner := &stubInner{rerankErr: fmt.Errorf("%w: down", ErrUnreachable)}
	b := NewBreaker(inner, 2, 10*time.Millisecond)

	// Trip the breaker.
	_, _ = b.Rerank(context.Background(), "q", []string{"d"})
	_, _ = b.Rerank(context.Background(), "q", []string{"d"})
	if !b.State().Open {
		t.Fatal("breaker not open after threshold")
	}

	// Wait past the open window.
	time.Sleep(15 * time.Millisecond)

	// Next call should reach the inner client (probe), not short-circuit.
	callsBefore := inner.rerankCalls
	_, _ = b.Rerank(context.Background(), "q", []string{"d"})
	if inner.rerankCalls == callsBefore {
		t.Errorf("probe call did not reach inner client after window elapsed")
	}
}
