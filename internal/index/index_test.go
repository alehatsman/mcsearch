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
	for i := 0; i < dim; i++ {
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
	hits, err := st.Search(ctx, qvecs[0], 5)
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
