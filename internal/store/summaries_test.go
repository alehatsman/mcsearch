package store

import (
	"testing"
	"time"
)

// AllSummaryChunks enumerates only the three summary kinds (code chunks
// excluded), and dedupes to the freshest row per (path, kind) — mirroring how
// successive prompt iterations leave multiple rows for the same path.
func TestAllSummaryChunks(t *testing.T) {
	st, ctx := newStore(t)
	older := time.Unix(1000, 0)
	newer := time.Unix(2000, 0)

	if err := st.UpsertMany(ctx, []PendingChunk{
		{Path: ".", Kind: "repo_summary", ContentSHA: "r1", Content: "repo", Vec: []float32{1, 0, 0, 0}},
		{Path: "internal/graph", Kind: "package_summary", ContentSHA: "p1", Content: "pkg v1", Vec: []float32{0, 1, 0, 0}},
		{Path: "internal/graph/export.go", Kind: "file_summary", ContentSHA: "f1", Content: "file", Vec: []float32{0, 0, 1, 0}},
		// A non-summary code chunk that must be excluded.
		{Path: "internal/graph/export.go", Kind: "fn", ContentSHA: "c1", Content: "code", Vec: []float32{0, 0, 0, 1}},
	}, older); err != nil {
		t.Fatal(err)
	}
	// A fresher row for an existing (path, kind) — must win the dedupe.
	if err := st.UpsertMany(ctx, []PendingChunk{
		{Path: "internal/graph", Kind: "package_summary", ContentSHA: "p2", Content: "pkg v2", Vec: []float32{0, 1, 0, 0}},
	}, newer); err != nil {
		t.Fatal(err)
	}

	got, err := st.AllSummaryChunks(ctx)
	if err != nil {
		t.Fatal(err)
	}

	want := map[string]string{ // path|kind -> content
		".|repo_summary":                        "repo",
		"internal/graph|package_summary":        "pkg v2",
		"internal/graph/export.go|file_summary": "file",
	}
	if len(got) != len(want) {
		t.Fatalf("got %d rows %+v, want %d", len(got), got, len(want))
	}
	for _, c := range got {
		key := c.Path + "|" + c.Kind
		w, ok := want[key]
		if !ok {
			t.Errorf("unexpected row %+v", c)
			continue
		}
		if c.Content != w {
			t.Errorf("%s content = %q, want %q", key, c.Content, w)
		}
	}
}
