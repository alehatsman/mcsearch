package index

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

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
