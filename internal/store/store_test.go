package store

import (
	"context"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/alehatsman/mcsearch/internal/rerank"
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
	if err := st.TouchSeen(ctx, "live.go", "h2", t1); err != nil {
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

// TestCrossProcessCacheInvalidation simulates what happens when a long-lived
// Store (e.g. the MCP server) has its cache built, then a separate process
// runs `mcsearch index` and adds new chunks. The Store must detect the
// changed last_indexed_at and rebuild its cache before the next Search so
// the new chunks appear in results.
func TestCrossProcessCacheInvalidation(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	ctx := context.Background()
	now := time.Now()

	// Writer: represents the `mcsearch index` process.
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
	// second `mcsearch index` run completing while the MCP server is live.
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

// TestSearchDisabledCache exercises the fallback hot path used when
// the caller (or the user via MCSEARCH_DISABLE_VEC_CACHE=1) explicitly
// asks Store not to hold decoded vectors in RAM. Top-k results must
// match the cached path's ordering exactly.
func TestSearchDisabledCache(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "test.db")
	ctx := context.Background()
	st, err := OpenWith(ctx, dbPath, Options{DisableVecCache: true})
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
	// Mutate; cached path also re-invalidates here but we don't care —
	// just verify the no-cache path stays consistent after an update.
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
