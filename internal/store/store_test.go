package store

import (
	"context"
	"path/filepath"
	"testing"
	"time"
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
	hits, err := st.Search(ctx, []float32{1, 0, 0, 0}, 3)
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
	hits, err := st.Search(ctx, []float32{1, 0}, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 0 {
		t.Errorf("expected 0 hits on empty index; got %d", len(hits))
	}
}

func TestSearchZeroQueryRejected(t *testing.T) {
	st, ctx := newStore(t)
	_, err := st.Search(ctx, []float32{0, 0, 0}, 5)
	if err == nil {
		t.Error("expected error for zero-norm query")
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
