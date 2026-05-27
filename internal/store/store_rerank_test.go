package store

import (
	"context"
	"errors"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alehatsman/dex/internal/rerank"
)

func TestRerankCacheKeyStableUnderReorder(t *testing.T) {
	a := rerankCacheKey("q", []int64{3, 1, 2})
	b := rerankCacheKey("q", []int64{1, 2, 3})
	if a != b {
		t.Errorf("key differs by input order: %s vs %s", a, b)
	}
	c := rerankCacheKey("other", []int64{1, 2, 3})
	if a == c {
		t.Errorf("different queries hashed to same key")
	}
}

func TestRerankLRUEviction(t *testing.T) {
	c := newRerankLRU(2)
	c.Put("a", rerankCached{scored: []scored{{id: 1, score: 0.5}}})
	c.Put("b", rerankCached{scored: []scored{{id: 2, score: 0.5}}})
	c.Put("c", rerankCached{scored: []scored{{id: 3, score: 0.5}}})

	if _, ok := c.Get("a"); ok {
		t.Errorf("a should have been evicted (LRU cap=2, inserted a→b→c)")
	}
	if _, ok := c.Get("b"); !ok {
		t.Errorf("b should still be present")
	}
	if _, ok := c.Get("c"); !ok {
		t.Errorf("c should still be present")
	}

	// Touch b so a fourth insert evicts c (now LRU).
	_, _ = c.Get("b")
	c.Put("d", rerankCached{scored: []scored{{id: 4, score: 0.5}}})
	if _, ok := c.Get("c"); ok {
		t.Errorf("c should have been evicted after touching b then inserting d")
	}
	if _, ok := c.Get("b"); !ok {
		t.Errorf("b should still be present after eviction of c")
	}
}

// countingReranker wraps a reranker and counts Rerank calls. Used to
// prove the LRU short-circuits the second identical call.
type countingReranker struct {
	calls atomic.Int64
}

func (c *countingReranker) Rerank(_ context.Context, _ string, docs []string) ([]rerank.Score, error) {
	c.calls.Add(1)
	out := make([]rerank.Score, len(docs))
	for i := range docs {
		out[i] = rerank.Score{Index: i, Score: 1.0 - float32(i)/float32(len(docs))}
	}
	return out, nil
}

func TestSearchRerankCacheHits(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now()
	var rows []PendingChunk
	for i := range 10 {
		rows = append(rows, PendingChunk{
			Path: "f" + strconv.Itoa(i) + ".go", Kind: "fn",
			ContentSHA: "sha" + strconv.Itoa(i),
			Content:    "rerank cache candidate " + strconv.Itoa(i),
			Vec:        []float32{1.0 - float32(i)*0.01, float32(i) * 0.01, 0, 0},
		})
	}
	if err := st.UpsertMany(ctx, rows, now); err != nil {
		t.Fatal(err)
	}

	rr := &countingReranker{}
	st.opts.Reranker = rr

	queryVec := []float32{1, 0, 0, 0}
	if _, err := st.Search(ctx, queryVec, "rerank cache candidate", 5); err != nil {
		t.Fatal(err)
	}
	if got := rr.calls.Load(); got != 1 {
		t.Fatalf("first call: rerank invocations = %d, want 1", got)
	}

	// Identical query + ids → cache hit, no second network call.
	if _, err := st.Search(ctx, queryVec, "rerank cache candidate", 5); err != nil {
		t.Fatal(err)
	}
	if got := rr.calls.Load(); got != 1 {
		t.Errorf("second call: rerank invocations = %d, want 1 (cache hit)", got)
	}

	// Different query text → cache miss, second call to the reranker.
	if _, err := st.Search(ctx, queryVec, "rerank cache different", 5); err != nil {
		t.Fatal(err)
	}
	if got := rr.calls.Load(); got != 2 {
		t.Errorf("different query: rerank invocations = %d, want 2", got)
	}
}

// hangingReranker blocks until ctx is cancelled, then returns the ctx
// error. Used to prove RerankTimeout caps the per-call latency.
type hangingReranker struct{}

func (hangingReranker) Rerank(ctx context.Context, _ string, _ []string) ([]rerank.Score, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestSearchRerankTimeoutDegradesToFused(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now()
	var rows []PendingChunk
	for i := range 10 {
		rows = append(rows, PendingChunk{
			Path: "f" + strconv.Itoa(i) + ".go", Kind: "fn",
			ContentSHA: "sha" + strconv.Itoa(i),
			Content:    "timeout candidate " + strconv.Itoa(i),
			Vec:        []float32{1.0 - float32(i)*0.01, float32(i) * 0.01, 0, 0},
		})
	}
	if err := st.UpsertMany(ctx, rows, now); err != nil {
		t.Fatal(err)
	}

	st.opts.Reranker = hangingReranker{}
	st.opts.RerankTimeout = 50 * time.Millisecond

	start := time.Now()
	hits, err := st.Search(ctx, []float32{1, 0, 0, 0}, "timeout candidate", 5)
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("Search returned err = %v; expected graceful fallback", err)
	}
	if len(hits) == 0 {
		t.Errorf("Search returned 0 hits; should have degraded to fused order")
	}
	// 1 second is generous; the actual rerank ctx is bounded at 50ms.
	// We just want to catch the case where the timeout was silently
	// ignored and we waited indefinitely.
	if elapsed > time.Second {
		t.Errorf("Search took %s with RerankTimeout=50ms; timeout did not bound the call", elapsed)
	}
	// Fallback hits must have no RerankScore set.
	for _, h := range hits {
		if h.RerankScore != 0 {
			t.Errorf("hit %q has RerankScore=%v after timeout; expected fallback path", h.Path, h.RerankScore)
		}
	}
}

func TestRerankTimeoutDoesNotMaskCallerCancel(t *testing.T) {
	// If the *outer* ctx is cancelled, the error should not be
	// reinterpreted as ErrUnreachable — it's the caller's intent.
	st, ctx := newStore(t)
	cancellableCtx, cancel := context.WithCancel(ctx)

	now := time.Now()
	var rows []PendingChunk
	for i := range 10 {
		rows = append(rows, PendingChunk{
			Path: "f" + strconv.Itoa(i) + ".go", Kind: "fn",
			ContentSHA: "sha" + strconv.Itoa(i),
			Content:    "x " + strconv.Itoa(i),
			Vec:        []float32{1.0 - float32(i)*0.01, float32(i) * 0.01, 0, 0},
		})
	}
	if err := st.UpsertMany(ctx, rows, now); err != nil {
		t.Fatal(err)
	}
	st.opts.Reranker = hangingReranker{}
	st.opts.RerankTimeout = 5 * time.Second

	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()

	_, err := st.Search(cancellableCtx, []float32{1, 0, 0, 0}, "x", 5)
	if err == nil {
		t.Fatal("expected an error when outer ctx is cancelled")
	}
	if errors.Is(err, rerank.ErrUnreachable) {
		t.Errorf("outer cancel was rewritten to ErrUnreachable: %v", err)
	}
}
