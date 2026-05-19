package index

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alehatsman/mcsearch/internal/chat"
	"github.com/alehatsman/mcsearch/internal/embed"
	"github.com/alehatsman/mcsearch/internal/ignore"
	"github.com/alehatsman/mcsearch/internal/proj"
	"github.com/alehatsman/mcsearch/internal/store"
)

// fakeEmbedServer hashes each input into a deterministic 16-dim float
// vector. Same input → same vector; lets us assert reasonable retrieval
// behavior without a real model.
func fakeEmbedServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			http.Error(w, "no", 404)
			return
		}
		var body struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		type item struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}
		out := struct {
			Data  []item `json:"data"`
			Model string `json:"model"`
		}{Model: body.Model}
		for i, in := range body.Input {
			out.Data = append(out.Data, item{Index: i, Embedding: hashVec(in, 16)})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(out)
	}))
}

func hashVec(s string, dim int) []float32 {
	out := make([]float32, dim)
	h := sha256.Sum256([]byte(s))
	for i := range dim {
		u := binary.LittleEndian.Uint32(h[(i*4)%len(h):])
		out[i] = float32(int32(u)) / float32(math.MaxInt32)
	}
	return out
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestIndexAndQuery(t *testing.T) {
	srv := fakeEmbedServer(t)
	defer srv.Close()

	projDir := t.TempDir()
	cacheDir := t.TempDir()

	// A Go file with two top-level declarations.
	writeFile(t, filepath.Join(projDir, "alpha.go"), `package main

// Alpha is the first function.
func Alpha() string { return "alpha" }

// Beta is the second function.
func Beta() string { return "beta" }
`)
	// A Markdown file → line-window chunking.
	writeFile(t, filepath.Join(projDir, "README.md"),
		"# Project\n\nThis is a README that should be indexed via line-window chunks.\n"+
			"It has more than one paragraph.\n\nMore text.\n")
	// A secret-like file → should be skipped.
	writeFile(t, filepath.Join(projDir, "creds.txt"),
		"-----BEGIN RSA PRIVATE KEY-----\nMIIB...\n")
	// A node_modules directory → should be skipped by default ignore.
	writeFile(t, filepath.Join(projDir, "node_modules/foo/index.js"),
		"function ignored() {}\n")

	ctx := context.Background()
	p, err := proj.Resolve(projDir, cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.EnsureCacheDir(); err != nil {
		t.Fatal(err)
	}
	st, err := store.Open(ctx, p.DBPath)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ig, err := ignore.New(p.Root)
	if err != nil {
		t.Fatal(err)
	}
	em := embed.New(srv.URL, "fake", 8, 10*time.Second)
	ix := New(p, st, em, ig, Options{Verbose: false})

	if err := ix.Run(ctx); err != nil {
		t.Fatalf("Run: %v", err)
	}

	stats, err := st.Stats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Files < 2 {
		t.Errorf("expected at least 2 files indexed (alpha.go, README.md); got %d", stats.Files)
	}
	if stats.Dim != 16 {
		t.Errorf("dim: got %d, want 16", stats.Dim)
	}
	if stats.Chunks < 2 {
		t.Errorf("expected at least 2 chunks; got %d", stats.Chunks)
	}

	// Query — the embedded text for "Alpha" should be closer to alpha.go
	// than to README.md (since the same hash function is used).
	qvecs, err := em.Embed(ctx, []string{
		"// path: alpha.go\n// kind: function_declaration\n// Alpha is the first function.\nfunc Alpha() string { return \"alpha\" }",
	})
	if err != nil {
		t.Fatal(err)
	}
	hits, err := st.Search(ctx, qvecs[0], "", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 {
		t.Fatal("Search returned no hits")
	}
	if hits[0].Path != "alpha.go" {
		t.Errorf("top hit path: got %q, want alpha.go", hits[0].Path)
	}

	// Re-run: nothing should change in chunk count (idempotent).
	before := stats.Chunks
	if err := ix.Run(ctx); err != nil {
		t.Fatalf("Run #2: %v", err)
	}
	stats2, _ := st.Stats(ctx)
	if stats2.Chunks != before {
		t.Errorf("re-index changed chunk count: %d → %d", before, stats2.Chunks)
	}

	// Remove a file; re-index; chunk count should drop.
	if err := os.Remove(filepath.Join(projDir, "alpha.go")); err != nil {
		t.Fatal(err)
	}
	if err := ix.Run(ctx); err != nil {
		t.Fatalf("Run #3: %v", err)
	}
	stats3, _ := st.Stats(ctx)
	if stats3.Chunks >= before {
		t.Errorf("expected chunk count to drop after removing alpha.go; got %d (was %d)", stats3.Chunks, before)
	}
}

// TestNewlyIgnoredEviction makes sure that adding a path to
// .mcsearch-ignore (or .gitignore) between runs evicts the chunks that
// were previously indexed under that path. Without explicit eviction
// the walker would simply skip the subtree on the next run and the
// stale chunks would live forever in the index.
func TestNewlyIgnoredEviction(t *testing.T) {
	srv := fakeEmbedServer(t)
	defer srv.Close()

	projDir := t.TempDir()
	cacheDir := t.TempDir()
	writeFile(t, filepath.Join(projDir, "main.go"),
		"package main\nfunc Main() {}\n")
	writeFile(t, filepath.Join(projDir, "drafts/wip.go"),
		"package drafts\nfunc WIP() {}\n")

	ctx := context.Background()
	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, _ := store.Open(ctx, p.DBPath)
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(srv.URL, "fake", 8, 5*time.Second)
	ix := New(p, st, em, ig, Options{})

	if err := ix.Run(ctx); err != nil {
		t.Fatal(err)
	}
	stats0, _ := st.Stats(ctx)
	if stats0.Files < 2 {
		t.Fatalf("expected both files indexed, got %d", stats0.Files)
	}

	// Add an ignore rule and reload the matcher.
	writeFile(t, filepath.Join(projDir, ".mcsearch-ignore"), "drafts/\n")
	ig2, _ := ignore.New(p.Root)
	ix2 := New(p, st, em, ig2, Options{})
	if err := ix2.Run(ctx); err != nil {
		t.Fatal(err)
	}
	stats1, _ := st.Stats(ctx)
	if stats1.Files != 1 {
		t.Errorf("expected drafts/ to be evicted, got %d files in index", stats1.Files)
	}
}

// TestPruneAtSameMillisecond exercises the regression where two successive
// Run() calls completing inside the same millisecond used to share a
// last_seen_at value, defeating the strict-less-than PruneUnseen filter.
// With nanosecond timestamps each call must produce a distinct cutoff.
func TestPruneAtSameMillisecond(t *testing.T) {
	srv := fakeEmbedServer(t)
	defer srv.Close()

	projDir := t.TempDir()
	cacheDir := t.TempDir()
	writeFile(t, filepath.Join(projDir, "a.go"),
		"package main\nfunc A() {}\n")
	writeFile(t, filepath.Join(projDir, "b.go"),
		"package main\nfunc B() {}\n")

	ctx := context.Background()
	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, _ := store.Open(ctx, p.DBPath)
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(srv.URL, "fake", 8, 5*time.Second)
	ix := New(p, st, em, ig, Options{})

	if err := ix.Run(ctx); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(projDir, "a.go")); err != nil {
		t.Fatal(err)
	}
	// Re-run in the same millisecond as the first. With nanosecond
	// precision the second cutoff strictly succeeds the first, so the
	// stale chunks for a.go must be pruned.
	if err := ix.Run(ctx); err != nil {
		t.Fatal(err)
	}
	stats, _ := st.Stats(ctx)
	if stats.Files != 1 {
		t.Errorf("expected 1 file after pruning a.go; got %d", stats.Files)
	}
}

// TestChunkSummaryIndexing verifies that per-chunk summaries are generated for
// structural chunks with ≥ chunkSummaryMinLines lines, skipped for tiny chunks,
// and cache-hit on a second run (no extra chat calls).
func TestChunkSummaryIndexing(t *testing.T) {
	var chatCalls int32
	chatSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&chatCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": "LongFunc processes the input."}},
			},
		})
	}))
	defer chatSrv.Close()

	embedSrv := fakeEmbedServer(t)
	defer embedSrv.Close()

	projDir := t.TempDir()
	cacheDir := t.TempDir()

	// Build a function body with ≥ chunkSummaryMinLines (30) lines.
	body := "package main\n\n// LongFunc is deliberately long.\nfunc LongFunc() {\n"
	for i := range chunkSummaryMinLines {
		body += fmt.Sprintf("\t// line %d\n", i+1)
	}
	body += "}\n"
	writeFile(t, filepath.Join(projDir, "long.go"), body)

	// A tiny function that must NOT trigger a chunk summary.
	writeFile(t, filepath.Join(projDir, "tiny.go"), "package main\nfunc Tiny() {}\n")

	ctx := context.Background()
	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, _ := store.Open(ctx, p.DBPath)
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(embedSrv.URL, "fake", 8, 5*time.Second)
	cc := chat.New(chatSrv.URL, "fake", 10*time.Second)
	ix := New(p, st, em, ig, Options{Summarize: true, Chat: cc})

	if err := ix.Run(ctx); err != nil {
		t.Fatalf("Run: %v", err)
	}

	callsAfterFirst := atomic.LoadInt32(&chatCalls)
	if callsAfterFirst == 0 {
		t.Fatal("expected at least one chat call for LongFunc chunk summary; got 0")
	}

	// Search for the summary text. The chunk_summary for long.go is in the
	// index (the chat call above proves that), but dedupChunkSummaries removes
	// it when the source function_declaration also appears in the top-k —
	// that would waste two slots on the same function. Verify the source
	// appears and the summary is not a duplicate alongside it.
	summaryText := "LongFunc processes the input."
	qvec, err := em.Embed(ctx, []string{summaryText})
	if err != nil {
		t.Fatal(err)
	}
	hits, err := st.Search(ctx, qvec[0], summaryText, 10)
	if err != nil {
		t.Fatal(err)
	}
	var foundDecl, foundSummaryDuplicate bool
	for _, h := range hits {
		if h.Kind == "function_declaration" && h.Path == "long.go" {
			foundDecl = true
		}
		if h.Kind == "chunk_summary" && h.Path == "long.go" {
			foundSummaryDuplicate = true
		}
	}
	if !foundDecl {
		t.Errorf("expected a function_declaration hit for long.go; got hits: %+v", hits)
	}
	if foundSummaryDuplicate {
		t.Errorf("chunk_summary should be deduped when its source function_declaration is present; got hits: %+v", hits)
	}

	// Second run: cache hit — chat must not be called again.
	if err := ix.Run(ctx); err != nil {
		t.Fatalf("Run #2: %v", err)
	}
	callsAfterSecond := atomic.LoadInt32(&chatCalls)
	if callsAfterSecond != callsAfterFirst {
		t.Errorf("second run made %d extra chat calls; expected 0 (cache hit)",
			callsAfterSecond-callsAfterFirst)
	}
}

// TestParallelWalkIndexesAllFiles asserts that the concurrent walk-and-chunk
// pipeline produces a complete index when there are many more files than
// workers. Catches regressions in producer/consumer plumbing — dropped
// tasks, channel-close races, or missing waitgroup synchronization would
// show up as a chunk count below the expected floor.
func TestParallelWalkIndexesAllFiles(t *testing.T) {
	srv := fakeEmbedServer(t)
	defer srv.Close()

	projDir := t.TempDir()
	cacheDir := t.TempDir()

	const fileCount = 64
	for i := 0; i < fileCount; i++ {
		writeFile(t, filepath.Join(projDir, fmt.Sprintf("f%02d.go", i)),
			fmt.Sprintf(`package main

func F%d() string { return "f%d" }
`, i, i))
	}

	ctx := context.Background()
	p, err := proj.Resolve(projDir, cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.EnsureCacheDir(); err != nil {
		t.Fatal(err)
	}
	st, err := store.Open(ctx, p.DBPath)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ig, err := ignore.New(p.Root)
	if err != nil {
		t.Fatal(err)
	}
	em := embed.New(srv.URL, "fake", 16, 10*time.Second)
	// Concurrency well above GOMAXPROCS to stress the channels.
	ix := New(p, st, em, ig, Options{Concurrency: 8})

	if err := ix.Run(ctx); err != nil {
		t.Fatalf("Run: %v", err)
	}
	stats, err := st.Stats(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Files != fileCount {
		t.Errorf("Files indexed = %d, want %d", stats.Files, fileCount)
	}
	if stats.Chunks < fileCount {
		t.Errorf("Chunks indexed = %d, want >= %d (one func per file)", stats.Chunks, fileCount)
	}
}
