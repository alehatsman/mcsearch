package store

import (
	"context"
	"fmt"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/alehatsman/dex/internal/rerank"
)

func newStore(t *testing.T) (*Store, context.Context) {
	t.Helper()
	db := filepath.Join(t.TempDir(), "test.db")
	ctx := context.Background()
	st, err := Open(ctx, db)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = st.Close() })
	return st, ctx
}

func TestUpsertAndSearch(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now()
	rows := []PendingChunk{
		{Path: "a.go", Kind: "fn", StartLine: 1, EndLine: 2, ContentSHA: "h1", Content: "func A(){}", Vec: []float32{1, 0, 0, 0}},
		{Path: "b.go", Kind: "fn", StartLine: 1, EndLine: 2, ContentSHA: "h2", Content: "func B(){}", Vec: []float32{0, 1, 0, 0}},
		{Path: "c.go", Kind: "fn", StartLine: 1, EndLine: 2, ContentSHA: "h3", Content: "func C(){}", Vec: []float32{1, 1, 0, 0}},
	}
	if err := st.UpsertMany(ctx, rows, now); err != nil {
		t.Fatal(err)
	}
	stats, _ := st.Stats(ctx)
	if stats.Chunks != 3 {
		t.Errorf("chunks=%d, want 3", stats.Chunks)
	}
	if stats.Dim != 4 {
		t.Errorf("dim=%d, want 4", stats.Dim)
	}

	// Query along (1,0,0,0) — `a.go` should rank first (cosine 1.0),
	// then `c.go` (cosine 1/√2 ≈ 0.707), then `b.go`.
	hits, err := st.Search(ctx, []float32{1, 0, 0, 0}, "", 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 3 {
		t.Fatalf("got %d hits, want 3", len(hits))
	}
	if hits[0].Path != "a.go" {
		t.Errorf("top hit = %q, want a.go", hits[0].Path)
	}
	if hits[1].Path != "c.go" {
		t.Errorf("second hit = %q, want c.go", hits[1].Path)
	}
}

func TestDimMismatchRejected(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now()
	_ = st.UpsertMany(ctx, []PendingChunk{
		{Path: "a", ContentSHA: "x", Content: "x", Vec: []float32{1, 0}},
	}, now)
	err := st.UpsertMany(ctx, []PendingChunk{
		{Path: "b", ContentSHA: "y", Content: "y", Vec: []float32{1, 0, 0}},
	}, now)
	if err == nil {
		t.Fatal("expected dim mismatch error")
	}
}

func TestPruneUnseen(t *testing.T) {
	st, ctx := newStore(t)
	t0 := time.Now()
	_ = st.UpsertMany(ctx, []PendingChunk{
		{Path: "old.go", ContentSHA: "h1", Content: "x", Vec: []float32{1, 0}},
		{Path: "live.go", ContentSHA: "h2", Content: "y", Vec: []float32{0, 1}},
	}, t0)
	// Advance: touch only `live.go`.
	t1 := t0.Add(time.Millisecond)
	if err := st.TouchSeen(ctx, "live.go", "h2", "", 0, 0, t1); err != nil {
		t.Fatal(err)
	}
	n, err := st.PruneUnseen(ctx, t1)
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("PruneUnseen pruned %d, want 1", n)
	}
	stats, _ := st.Stats(ctx)
	if stats.Files != 1 {
		t.Errorf("files=%d, want 1", stats.Files)
	}
}

func TestTouchPath(t *testing.T) {
	st, ctx := newStore(t)
	t0 := time.Now()
	_ = st.UpsertMany(ctx, []PendingChunk{
		{Path: "a.go", ContentSHA: "h1", Content: "x", Vec: []float32{1, 0}},
		{Path: "a.go", ContentSHA: "h2", Content: "y", Vec: []float32{0, 1}},
		{Path: "b.go", ContentSHA: "h3", Content: "z", Vec: []float32{1, 1}},
	}, t0)
	t1 := t0.Add(time.Millisecond)
	n, err := st.TouchPath(ctx, "a.go", t1)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Errorf("TouchPath rows = %d, want 2 (both chunks of a.go)", n)
	}
}

func TestDeletePathPrefix(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now()
	_ = st.UpsertMany(ctx, []PendingChunk{
		{Path: "vendor/a.go", ContentSHA: "h1", Content: "x", Vec: []float32{1, 0}},
		{Path: "vendor/b.go", ContentSHA: "h2", Content: "y", Vec: []float32{0, 1}},
		{Path: "src/main.go", ContentSHA: "h3", Content: "z", Vec: []float32{1, 1}},
	}, now)
	if err := st.DeletePathPrefix(ctx, "vendor/"); err != nil {
		t.Fatal(err)
	}
	stats, _ := st.Stats(ctx)
	if stats.Chunks != 1 {
		t.Errorf("chunks after prefix delete = %d, want 1", stats.Chunks)
	}
}

func TestSearchEmptyIndex(t *testing.T) {
	st, ctx := newStore(t)
	hits, err := st.Search(ctx, []float32{1, 0}, "", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 0 {
		t.Errorf("expected 0 hits on empty index; got %d", len(hits))
	}
}

func TestSearchZeroQueryRejected(t *testing.T) {
	st, ctx := newStore(t)
	_, err := st.Search(ctx, []float32{0, 0, 0}, "", 5)
	if err == nil {
		t.Error("expected error for zero-norm query")
	}
}

// TestSearchCacheInvalidation guards against the failure mode where
// the in-RAM vector cache outlives a mutating operation and surfaces
// chunks that were deleted/replaced. Each mutator must invalidate.
func TestSearchCacheInvalidation(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now()
	if err := st.UpsertMany(ctx, []PendingChunk{
		{Path: "a.go", ContentSHA: "h1", Content: "func A(){}", Vec: []float32{1, 0, 0, 0}},
		{Path: "b.go", ContentSHA: "h2", Content: "func B(){}", Vec: []float32{0, 1, 0, 0}},
	}, now); err != nil {
		t.Fatal(err)
	}
	// Warm the cache.
	hits, _ := st.Search(ctx, []float32{1, 0, 0, 0}, "", 5)
	if len(hits) != 2 {
		t.Fatalf("baseline: got %d hits, want 2", len(hits))
	}
	// Delete a.go and re-query — cache must reflect the removal.
	if err := st.DeletePath(ctx, "a.go"); err != nil {
		t.Fatal(err)
	}
	hits, _ = st.Search(ctx, []float32{1, 0, 0, 0}, "", 5)
	if len(hits) != 1 || hits[0].Path != "b.go" {
		t.Errorf("after DeletePath, got %d hits, top=%q; want 1 hit b.go", len(hits), pathOrNone(hits))
	}
	// Re-upsert and verify the new content lands in subsequent searches.
	if err := st.UpsertMany(ctx, []PendingChunk{
		{Path: "c.go", ContentSHA: "h3", Content: "func C(){}", Vec: []float32{1, 1, 0, 0}},
	}, now); err != nil {
		t.Fatal(err)
	}
	hits, _ = st.Search(ctx, []float32{1, 0, 0, 0}, "", 5)
	if len(hits) != 2 {
		t.Errorf("after UpsertMany, got %d hits, want 2", len(hits))
	}
	// PruneUnseen also invalidates (deletes everything because cutoff
	// is in the future).
	if _, err := st.PruneUnseen(ctx, time.Now().Add(time.Hour)); err != nil {
		t.Fatal(err)
	}
	hits, _ = st.Search(ctx, []float32{1, 0, 0, 0}, "", 5)
	if len(hits) != 0 {
		t.Errorf("after PruneUnseen all, got %d hits, want 0", len(hits))
	}
}

// TestSearchLargeIndex confirms search keeps working past the size at
// which the legacy in-RAM slab cache used to fail. The slab is gone —
// vec0 handles arbitrary row counts — but the assertion is still cheap
// to keep as a smoke test against future regressions.
func TestSearchLargeIndex(t *testing.T) {
	st, ctx := newStore(t)
	const n = 1100
	pending := make([]PendingChunk, n)
	for i := range n {
		pending[i] = PendingChunk{
			Path:       fmt.Sprintf("pkg/f%d.go", i),
			ContentSHA: fmt.Sprintf("h%d", i),
			Content:    fmt.Sprintf("func F%d(){}", i),
			Vec:        []float32{float32(i + 1), 0, 0, 0},
		}
	}
	if err := st.UpsertMany(ctx, pending, time.Now()); err != nil {
		t.Fatal(err)
	}
	hits, err := st.Search(ctx, []float32{1, 0, 0, 0}, "", 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) == 0 {
		t.Fatal("expected hits from large-index search")
	}
}

// TestCrossProcessVisibility simulates what happens when a long-lived
// Store (e.g. the MCP server) is running and a separate process runs
// `dex index` and adds new chunks. With the slab cache gone the
// reader hits chunk_vecs directly each Search, so SQLite WAL alone has
// to make the new rows visible — no client-side invalidation required.
func TestCrossProcessVisibility(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	ctx := context.Background()
	now := time.Now()

	// Writer: represents the `dex index` process.
	writer, err := Open(ctx, dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer writer.Close()
	if err := writer.UpsertMany(ctx, []PendingChunk{
		{Path: "a.go", ContentSHA: "h1", Content: "func A(){}", Vec: []float32{1, 0, 0, 0}},
	}, now); err != nil {
		t.Fatal(err)
	}
	if err := writer.SetLastIndexedAt(ctx, now); err != nil {
		t.Fatal(err)
	}

	// Reader: represents the long-running MCP server.
	reader, err := Open(ctx, dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	// Warm the reader's cache.
	hits, _ := reader.Search(ctx, []float32{1, 0, 0, 0}, "", 5)
	if len(hits) != 1 {
		t.Fatalf("baseline: got %d hits, want 1", len(hits))
	}

	// Writer adds a new chunk and advances last_indexed_at — simulates a
	// second `dex index` run completing while the MCP server is live.
	now2 := now.Add(time.Second)
	if err := writer.UpsertMany(ctx, []PendingChunk{
		{Path: "b.go", ContentSHA: "h2", Content: "func B(){}", Vec: []float32{0, 1, 0, 0}},
	}, now2); err != nil {
		t.Fatal(err)
	}
	if err := writer.SetLastIndexedAt(ctx, now2); err != nil {
		t.Fatal(err)
	}

	// Reader searches again — must detect staleness, rebuild cache, and
	// return both chunks.
	hits, _ = reader.Search(ctx, []float32{1, 0, 0, 0}, "", 5)
	if len(hits) != 2 {
		t.Errorf("after cross-process upsert: got %d hits, want 2; cache not invalidated", len(hits))
	}
}

func pathOrNone(hits []Hit) string {
	if len(hits) == 0 {
		return "<none>"
	}
	return hits[0].Path
}

// TestSearchAfterDeletePath verifies the vec0 trigger drops a chunk's
// embedding when its row leaves the chunks table, so deleted paths
// disappear from search results without a manual cache flush.
func TestSearchAfterDeletePath(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	ctx := context.Background()
	st, err := Open(ctx, dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	now := time.Now()
	if err := st.UpsertMany(ctx, []PendingChunk{
		{Path: "a.go", Kind: "fn", ContentSHA: "h1", Content: "func A(){}", Vec: []float32{1, 0, 0, 0}},
		{Path: "b.go", Kind: "fn", ContentSHA: "h2", Content: "func B(){}", Vec: []float32{0, 1, 0, 0}},
		{Path: "c.go", Kind: "fn", ContentSHA: "h3", Content: "func C(){}", Vec: []float32{1, 1, 0, 0}},
	}, now); err != nil {
		t.Fatal(err)
	}
	hits, err := st.Search(ctx, []float32{1, 0, 0, 0}, "", 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 3 {
		t.Fatalf("got %d hits, want 3", len(hits))
	}
	if hits[0].Path != "a.go" {
		t.Errorf("top hit = %q, want a.go", hits[0].Path)
	}
	if hits[1].Path != "c.go" {
		t.Errorf("second hit = %q, want c.go", hits[1].Path)
	}
	if err := st.DeletePath(ctx, "a.go"); err != nil {
		t.Fatal(err)
	}
	hits, _ = st.Search(ctx, []float32{1, 0, 0, 0}, "", 3)
	if len(hits) != 2 || hits[0].Path == "a.go" {
		t.Errorf("after DeletePath, got %d hits top=%q; want 2 hits without a.go", len(hits), pathOrNone(hits))
	}
}

// TestHybridSearchBM25Surfaces verifies hybrid search recovers an
// exact-identifier match even when the semantic vector intentionally
// points elsewhere. The "needle" chunk has a near-zero cosine to the
// query vector, but its content contains the unique token
// "validateToken" — BM25 should rank it #1, RRF lifts it into top-k.
func TestHybridSearchBM25Surfaces(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now()
	// 8 noise chunks with arbitrary content (all close to [1,0,0,0]).
	rows := []PendingChunk{
		{Path: "noise1.go", Kind: "fn", ContentSHA: "n1", Content: "func a() { return 1 }", Vec: []float32{1, 0, 0, 0}},
		{Path: "noise2.go", Kind: "fn", ContentSHA: "n2", Content: "func b() { return 2 }", Vec: []float32{0.99, 0.1, 0, 0}},
		{Path: "noise3.go", Kind: "fn", ContentSHA: "n3", Content: "func c() { return 3 }", Vec: []float32{0.98, 0.2, 0, 0}},
		{Path: "noise4.go", Kind: "fn", ContentSHA: "n4", Content: "func d() { return 4 }", Vec: []float32{0.97, 0.3, 0, 0}},
		// The needle: semantically orthogonal but contains the literal
		// identifier "validateToken" that the query is asking for.
		{Path: "auth.go", Kind: "fn", ContentSHA: "needle",
			Content: "func validateToken(tok string) bool { return tok != \"\" }",
			Vec:     []float32{0, 0, 1, 0}},
	}
	if err := st.UpsertMany(ctx, rows, now); err != nil {
		t.Fatal(err)
	}

	queryVec := []float32{1, 0, 0, 0}

	// Semantic-only — the needle should rank LAST because its vector
	// is orthogonal to the query.
	semHits, err := st.Search(ctx, queryVec, "", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(semHits) == 0 || semHits[0].Path == "auth.go" {
		t.Fatalf("semantic-only should NOT put auth.go first (got top=%q)", semHits[0].Path)
	}

	// Hybrid — same query vector, but with the natural-language text
	// that contains "validateToken". RRF should lift auth.go to #1.
	hybridHits, err := st.Search(ctx, queryVec, "validateToken function", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hybridHits) == 0 || hybridHits[0].Path != "auth.go" {
		t.Errorf("hybrid: top hit = %q, want auth.go (BM25 should surface it)", pathOrNone(hybridHits))
	}
	// The needle should have a populated BM25Score and RRFScore.
	for _, h := range hybridHits {
		if h.Path == "auth.go" {
			if h.BM25Score <= 0 {
				t.Errorf("auth.go BM25Score = %v, want > 0", h.BM25Score)
			}
			if h.RRFScore <= 0 {
				t.Errorf("auth.go RRFScore = %v, want > 0", h.RRFScore)
			}
		}
	}
}

// TestHybridDegradesGracefully covers the failure modes around BM25:
// FTS5 query with only stop-symbols → empty MATCH expression → search
// silently falls back to semantic-only, no error. Same for an explicit
// DisableBM25.
func TestHybridDegradesGracefully(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now()
	if err := st.UpsertMany(ctx, []PendingChunk{
		{Path: "a.go", ContentSHA: "h1", Content: "func A(){}", Vec: []float32{1, 0, 0, 0}},
		{Path: "b.go", ContentSHA: "h2", Content: "func B(){}", Vec: []float32{0, 1, 0, 0}},
	}, now); err != nil {
		t.Fatal(err)
	}
	// Query text with no usable tokens (single-char + punctuation).
	hits, err := st.Search(ctx, []float32{1, 0, 0, 0}, "@ ; ,", 5)
	if err != nil {
		t.Fatalf("hybrid search with unusable query text should not error: %v", err)
	}
	if len(hits) != 2 {
		t.Errorf("got %d hits, want 2 (semantic fallback)", len(hits))
	}
}

// TestDisableBM25 confirms the env-driven kill switch turns off the
// lexical leg even when query text is present.
func TestDisableBM25(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	ctx := context.Background()
	st, err := OpenWith(ctx, dbPath, Options{DisableBM25: true})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	now := time.Now()
	if err := st.UpsertMany(ctx, []PendingChunk{
		{Path: "noise.go", ContentSHA: "n1", Content: "func irrelevant()", Vec: []float32{1, 0, 0, 0}},
		{Path: "needle.go", ContentSHA: "n2", Content: "func validateToken()", Vec: []float32{0, 0, 1, 0}},
	}, now); err != nil {
		t.Fatal(err)
	}
	// With BM25 disabled, even a perfect lexical match for "validateToken"
	// should NOT lift needle.go above the semantically-aligned noise.go.
	hits, err := st.Search(ctx, []float32{1, 0, 0, 0}, "validateToken", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].Path != "noise.go" {
		t.Errorf("DisableBM25: top = %q, want noise.go (semantic-only)", pathOrNone(hits))
	}
	if hits[0].BM25Score != 0 || hits[0].RRFScore != 0 {
		t.Errorf("DisableBM25 should leave BM25Score/RRFScore zero; got %v / %v",
			hits[0].BM25Score, hits[0].RRFScore)
	}
}

// reverseReranker is a deterministic stub: it returns the input docs
// in reversed order so we can verify the rerank stage actually
// reorders the fused candidates.
type reverseReranker struct{}

func (reverseReranker) Rerank(_ context.Context, _ string, docs []string) ([]rerank.Score, error) {
	out := make([]rerank.Score, len(docs))
	for i := range docs {
		out[i] = rerank.Score{
			Index: len(docs) - 1 - i,
			// Descending scores so fetchHits's order = ranking order.
			Score: 1.0 - float32(i)/float32(len(docs)),
		}
	}
	return out, nil
}

// unreachableReranker mirrors what a network-down rerank.Client returns.
// store.Search must catch this and degrade to pre-rerank truncation.
type unreachableReranker struct{}

func (unreachableReranker) Rerank(_ context.Context, _ string, _ []string) ([]rerank.Score, error) {
	return nil, rerank.ErrUnreachable
}

func TestSearchReranks(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now()
	// 10 chunks, distinguishable by path and content. Vec varies just
	// enough to give a deterministic cosine ordering; every content has
	// the literal "rerank" so all chunks enter the BM25 pool too. Both
	// legs of hybrid retrieval are populated → fused has 10 candidates.
	var rows []PendingChunk
	for i := range 10 {
		rows = append(rows, PendingChunk{
			Path: "f" + strconv.Itoa(i) + ".go", Kind: "fn",
			ContentSHA: "sha" + strconv.Itoa(i),
			Content:    "rerank candidate " + strconv.Itoa(i),
			Vec:        []float32{1.0 - float32(i)*0.01, float32(i) * 0.01, 0, 0},
		})
	}
	if err := st.UpsertMany(ctx, rows, now); err != nil {
		t.Fatal(err)
	}

	queryVec := []float32{1, 0, 0, 0}
	const k = 5

	// Baseline: no reranker, capture top-k order.
	baseline, err := st.Search(ctx, queryVec, "rerank candidate", k)
	if err != nil {
		t.Fatal(err)
	}
	if len(baseline) != k {
		t.Fatalf("baseline returned %d hits, want %d", len(baseline), k)
	}
	for _, h := range baseline {
		if h.RerankScore != 0 {
			t.Errorf("baseline %q: RerankScore = %v, want 0 (no reranker wired)", h.Path, h.RerankScore)
		}
	}

	// Reranked: stub reverses the fused order.
	st.opts.Reranker = reverseReranker{}
	reranked, err := st.Search(ctx, queryVec, "rerank candidate", k)
	if err != nil {
		t.Fatal(err)
	}
	if len(reranked) != k {
		t.Fatalf("reranked returned %d hits, want %d", len(reranked), k)
	}

	// Order must differ — the reverse stub guarantees a permutation.
	sameOrder := true
	for i := range k {
		if baseline[i].Path != reranked[i].Path {
			sameOrder = false
			break
		}
	}
	if sameOrder {
		t.Errorf("reranked order equals baseline; rerank stage did not reorder.\nbaseline: %v\nreranked: %v",
			paths(baseline), paths(reranked))
	}

	// Every reranked hit has a non-zero RerankScore, and the slice is
	// sorted descending (stub returns sorted; fetchHits preserves order).
	for i, h := range reranked {
		if h.RerankScore <= 0 {
			t.Errorf("reranked[%d] %q: RerankScore = %v, want > 0", i, h.Path, h.RerankScore)
		}
		if i > 0 && reranked[i-1].RerankScore < h.RerankScore {
			t.Errorf("reranked not sorted desc by RerankScore: %v < %v at i=%d",
				reranked[i-1].RerankScore, h.RerankScore, i)
		}
		// RRFScore must survive the reorder — Search keeps the original
		// RRF score map alive precisely for this.
		if h.RRFScore <= 0 {
			t.Errorf("reranked[%d] %q: RRFScore = %v, want > 0 (should survive rerank)", i, h.Path, h.RRFScore)
		}
	}
}

func TestSearchRerankerUnreachableFallsBack(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now()
	var rows []PendingChunk
	for i := range 10 {
		rows = append(rows, PendingChunk{
			Path: "f" + strconv.Itoa(i) + ".go", Kind: "fn",
			ContentSHA: "sha" + strconv.Itoa(i),
			Content:    "rerank candidate " + strconv.Itoa(i),
			Vec:        []float32{1.0 - float32(i)*0.01, float32(i) * 0.01, 0, 0},
		})
	}
	if err := st.UpsertMany(ctx, rows, now); err != nil {
		t.Fatal(err)
	}

	queryVec := []float32{1, 0, 0, 0}
	const k = 5

	baseline, err := st.Search(ctx, queryVec, "rerank candidate", k)
	if err != nil {
		t.Fatal(err)
	}

	st.opts.Reranker = unreachableReranker{}
	fallback, err := st.Search(ctx, queryVec, "rerank candidate", k)
	if err != nil {
		t.Fatalf("unreachable reranker should not surface as error: %v", err)
	}
	if len(fallback) != len(baseline) {
		t.Fatalf("fallback returned %d hits, baseline %d", len(fallback), len(baseline))
	}
	for i := range fallback {
		if fallback[i].Path != baseline[i].Path {
			t.Errorf("fallback[%d] = %q, want %q (should equal pre-rerank order)",
				i, fallback[i].Path, baseline[i].Path)
		}
		if fallback[i].RerankScore != 0 {
			t.Errorf("fallback[%d] %q: RerankScore = %v, want 0 (rerank didn't run)",
				i, fallback[i].Path, fallback[i].RerankScore)
		}
	}
}

func paths(hits []Hit) []string {
	out := make([]string, len(hits))
	for i, h := range hits {
		out[i] = h.Path
	}
	return out
}

func TestProjectRootRoundTrip(t *testing.T) {
	st, ctx := newStore(t)

	got, err := st.ProjectRoot(ctx)
	if err != nil {
		t.Fatalf("ProjectRoot on fresh db: %v", err)
	}
	if got != "" {
		t.Errorf("fresh db should have empty project_root; got %q", got)
	}

	want := "/abs/path/to/proj"
	if err := st.SetProjectRoot(ctx, want); err != nil {
		t.Fatal(err)
	}
	got, err = st.ProjectRoot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Errorf("ProjectRoot = %q, want %q", got, want)
	}

	// Overwrite — upsert semantics, not insert-or-error.
	want = "/abs/path/to/other"
	if err := st.SetProjectRoot(ctx, want); err != nil {
		t.Fatal(err)
	}
	got, _ = st.ProjectRoot(ctx)
	if got != want {
		t.Errorf("after overwrite ProjectRoot = %q, want %q", got, want)
	}
}

func TestPersistsDimAcrossOpen(t *testing.T) {
	dir := t.TempDir()
	db := filepath.Join(dir, "test.db")
	ctx := context.Background()

	st, err := Open(ctx, db)
	if err != nil {
		t.Fatal(err)
	}
	if err := st.UpsertMany(ctx, []PendingChunk{
		{Path: "a", ContentSHA: "h", Content: "x", Vec: []float32{1, 2, 3, 4}},
	}, time.Now()); err != nil {
		t.Fatal(err)
	}
	st.Close()

	st2, err := Open(ctx, db)
	if err != nil {
		t.Fatal(err)
	}
	defer st2.Close()
	stats, _ := st2.Stats(ctx)
	if stats.Dim != 4 {
		t.Errorf("dim not persisted: got %d", stats.Dim)
	}
}

func TestTouchSeenRefreshesPositions(t *testing.T) {
	// TouchSeen is called when a chunk's SHA already exists. If the
	// caller passes new start_line/end_line, those must be persisted —
	// the chunk's content can stay the same while its position in the
	// file shifts (something above grew/shrank). Without this, find_
	// symbol returns the ORIGINAL line range across edits.
	st, ctx := newStore(t)
	now := time.Now()
	if err := st.UpsertMany(ctx, []PendingChunk{
		{Path: "a.go", Kind: "fn", Name: "F", ContentSHA: "h1", Content: "x",
			StartLine: 10, EndLine: 20, Vec: []float32{1, 0, 0, 0}},
	}, now); err != nil {
		t.Fatal(err)
	}

	// File edited above: F's content didn't change but it moved down 24 lines.
	later := now.Add(time.Second)
	if err := st.TouchSeen(ctx, "a.go", "h1", "F", 34, 44, later); err != nil {
		t.Fatal(err)
	}

	hits, err := st.FindSymbol(ctx, "F", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 {
		t.Fatalf("got %d hits, want 1", len(hits))
	}
	if hits[0].StartLine != 34 || hits[0].EndLine != 44 {
		t.Errorf("position not refreshed; got %d-%d, want 34-44", hits[0].StartLine, hits[0].EndLine)
	}

	// Sentinel startLine=0 leaves positions alone (summary callers).
	later2 := later.Add(time.Second)
	if err := st.TouchSeen(ctx, "a.go", "h1", "F", 0, 0, later2); err != nil {
		t.Fatal(err)
	}
	hits, _ = st.FindSymbol(ctx, "F", 5)
	if hits[0].StartLine != 34 || hits[0].EndLine != 44 {
		t.Errorf("sentinel 0,0 should preserve positions; got %d-%d", hits[0].StartLine, hits[0].EndLine)
	}
}

func TestFindSymbolFallsBackToGraphNodes(t *testing.T) {
	// Struct fields and type-only entities aren't represented as chunks
	// (the chunker emits per-function/method chunks), but they ARE in
	// graph_nodes. FindSymbol should fall back to that table when
	// chunks misses, so a query like `MaxFileSize` resolves to the
	// field even though there's no matching chunk row.
	st, ctx := newStore(t)
	now := time.Now()

	// One unrelated chunk so the chunks table isn't empty.
	if err := st.UpsertMany(ctx, []PendingChunk{
		{Path: "a.go", Kind: "fn", Name: "Other", ContentSHA: "h", Content: "x", Vec: []float32{1, 0, 0, 0}},
	}, now); err != nil {
		t.Fatal(err)
	}

	// Insert a graph node mimicking a struct field — same shape the
	// Go static-graph layer produces.
	gnodes := []GraphNodeRow{{
		ID:            "field:Options.MaxFileSize",
		Kind:          "field",
		Name:          "MaxFileSize",
		QualifiedName: "Options.MaxFileSize",
		PackagePath:   "github.com/example/index",
		FilePath:      "internal/index/index.go",
		StartLine:     39,
		EndLine:       39,
		ContentHash:   "h1",
	}}
	if err := st.GraphUpsertNodes(ctx, gnodes, now); err != nil {
		t.Fatal(err)
	}

	hits, err := st.FindSymbol(ctx, "MaxFileSize", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 {
		t.Fatalf("expected 1 hit via graph fallback, got %d: %+v", len(hits), hits)
	}
	if hits[0].Kind != "field" {
		t.Errorf("hit.Kind=%q, want field", hits[0].Kind)
	}
	if hits[0].Path != "internal/index/index.go" || hits[0].StartLine != 39 {
		t.Errorf("hit path/line wrong: %+v", hits[0])
	}

	// When chunks DOES have the name, fallback shouldn't kick in.
	if err := st.UpsertMany(ctx, []PendingChunk{
		{Path: "b.go", Kind: "fn", Name: "ChunkName", ContentSHA: "h2", Content: "x", Vec: []float32{0, 1, 0, 0}},
	}, now); err != nil {
		t.Fatal(err)
	}
	chunkHits, err := st.FindSymbol(ctx, "ChunkName", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunkHits) != 1 || chunkHits[0].Path != "b.go" {
		t.Errorf("chunks lookup should win when present; got %+v", chunkHits)
	}
}

func TestFindSymbolSortsByCentrality(t *testing.T) {
	// Two chunks with the same name. Wire one to a graph_node with a
	// high in_degree and PageRank; the other to a zero-centrality node.
	// FindSymbol should now return the central one first regardless of
	// path order. Prevents the regression where a same-named glue
	// function shipped ahead of the real domain symbol.
	st, ctx := newStore(t)
	now := time.Now()

	if err := st.UpsertMany(ctx, []PendingChunk{
		{Path: "a/glue.go", Kind: "fn", Name: "Run", StartLine: 1, EndLine: 3, ContentSHA: "g", Content: "x", Vec: []float32{1, 0, 0, 0}},
		{Path: "b/core.go", Kind: "fn", Name: "Run", StartLine: 1, EndLine: 3, ContentSHA: "c", Content: "x", Vec: []float32{0, 1, 0, 0}},
	}, now); err != nil {
		t.Fatal(err)
	}
	// Recover chunk IDs so the graph node's chunk_id link is correct.
	var glueID, coreID int64
	if err := st.db.QueryRowContext(ctx, `SELECT id FROM chunks WHERE path='a/glue.go'`).Scan(&glueID); err != nil {
		t.Fatal(err)
	}
	if err := st.db.QueryRowContext(ctx, `SELECT id FROM chunks WHERE path='b/core.go'`).Scan(&coreID); err != nil {
		t.Fatal(err)
	}
	nodes := []GraphNodeRow{
		{ID: "n-glue", Kind: "function", Name: "Run", QualifiedName: "Run", PackagePath: "pkg/glue", FilePath: "a/glue.go", StartLine: 1, EndLine: 3, ChunkID: glueID, ContentHash: "h1"},
		{ID: "n-core", Kind: "function", Name: "Run", QualifiedName: "Run", PackagePath: "pkg/core", FilePath: "b/core.go", StartLine: 1, EndLine: 3, ChunkID: coreID, ContentHash: "h2"},
	}
	if err := st.GraphUpsertNodes(ctx, nodes, now); err != nil {
		t.Fatal(err)
	}
	if err := st.GraphSetCentrality(ctx, []GraphCentralityRow{
		{ID: "n-glue", InDegree: 0, PageRank: 0.001},
		{ID: "n-core", InDegree: 8, CrossPkgCallers: 3, PageRank: 0.12},
	}); err != nil {
		t.Fatal(err)
	}

	hits, err := st.FindSymbol(ctx, "Run", 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 {
		t.Fatalf("want 2 hits, got %d", len(hits))
	}
	// Central node should sort first even though "a/" precedes "b/" by
	// path.
	if hits[0].Path != "b/core.go" {
		t.Errorf("hits[0].Path = %q, want b/core.go (the central one); raw=%+v", hits[0].Path, hits)
	}
	if hits[0].InDegree != 8 || hits[0].CrossPkgCallers != 3 {
		t.Errorf("centrality columns not propagated: %+v", hits[0])
	}
	if hits[1].Path != "a/glue.go" || hits[1].InDegree != 0 {
		t.Errorf("hits[1] should be the glue (zero centrality); got %+v", hits[1])
	}
}

func TestFindSymbolCandidates(t *testing.T) {
	st, ctx := newStore(t)
	now := time.Now()
	rows := []PendingChunk{
		{Path: "a.go", Kind: "fn", Name: "Indexer", ContentSHA: "h1", Content: "x", Vec: []float32{1, 0, 0, 0}},
		{Path: "b.go", Kind: "fn", Name: "IndexableExt", ContentSHA: "h2", Content: "x", Vec: []float32{0, 1, 0, 0}},
		{Path: "c.go", Kind: "fn", Name: "indexBase", ContentSHA: "h3", Content: "x", Vec: []float32{0, 0, 1, 0}},
		{Path: "d.go", Kind: "fn", Name: "cmdIndex", ContentSHA: "h4", Content: "x", Vec: []float32{0, 0, 0, 1}},
		{Path: "e.go", Kind: "fn", Name: "Unrelated", ContentSHA: "h5", Content: "x", Vec: []float32{1, 1, 0, 0}},
	}
	if err := st.UpsertMany(ctx, rows, now); err != nil {
		t.Fatal(err)
	}

	got, err := st.FindSymbolCandidates(ctx, "Index", 5)
	if err != nil {
		t.Fatal(err)
	}
	// All four names contain "Index" as substring; "Unrelated" does not.
	want := map[string]bool{"Indexer": true, "IndexableExt": true, "indexBase": false, "cmdIndex": true}
	// Note: SQLite LIKE is case-insensitive by default for ASCII, so
	// `indexBase` also matches; we'll see all four.
	for _, name := range got {
		want[name] = true
	}
	if len(got) != 4 {
		t.Errorf("want 4 candidates (substring of 'Index'); got %d: %v", len(got), got)
	}

	// Exact-match name should NOT come back in candidates (the caller
	// already knows that one failed).
	got2, _ := st.FindSymbolCandidates(ctx, "Indexer", 5)
	for _, n := range got2 {
		if n == "Indexer" {
			t.Errorf("exact-match name %q should be excluded from candidates", n)
		}
	}

	// Empty query → empty result.
	got3, _ := st.FindSymbolCandidates(ctx, "", 5)
	if len(got3) != 0 {
		t.Errorf("empty query should yield 0 candidates; got %v", got3)
	}

	// k caps the output.
	got4, _ := st.FindSymbolCandidates(ctx, "Index", 2)
	if len(got4) != 2 {
		t.Errorf("k=2 cap not honored; got %d: %v", len(got4), got4)
	}
}
