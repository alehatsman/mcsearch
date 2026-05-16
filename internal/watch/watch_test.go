package watch

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
	"testing"
	"time"

	"github.com/alehatsman/mcsearch/internal/embed"
	"github.com/alehatsman/mcsearch/internal/ignore"
	"github.com/alehatsman/mcsearch/internal/index"
	"github.com/alehatsman/mcsearch/internal/proj"
	"github.com/alehatsman/mcsearch/internal/store"
)

func fakeEmbedServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		type item struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}
		out := struct {
			Data  []item `json:"data"`
			Model string `json:"model"`
		}{Model: body.Model}
		for i, s := range body.Input {
			out.Data = append(out.Data, item{Index: i, Embedding: hashVec(s, 8)})
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

func TestWatchReindexesOnSave(t *testing.T) {
	srv := fakeEmbedServer(t)
	defer srv.Close()

	projDir := t.TempDir()
	cacheDir := t.TempDir()
	_ = os.WriteFile(filepath.Join(projDir, "alpha.go"),
		[]byte("package main\nfunc Alpha() {}\n"), 0o644)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	p, err := proj.Resolve(projDir, cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	_ = p.EnsureCacheDir()
	st, err := store.Open(ctx, p.DBPath)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(srv.URL, "fake", 8, 5*time.Second)
	ix := index.New(p, st, em, ig, index.Options{})
	w := New(ix, ig, p.Root, Options{Debounce: 50 * time.Millisecond})

	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()

	// Wait for the initial index pass to complete.
	for range 50 {
		stats, _ := st.Stats(ctx)
		if stats.Chunks > 0 {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	before, _ := st.Stats(ctx)
	if before.Chunks == 0 {
		t.Fatal("initial pass produced no chunks")
	}

	// Add a new file. The watcher should pick it up within debounce.
	_ = os.WriteFile(filepath.Join(projDir, "beta.go"),
		[]byte("package main\nfunc Beta() {}\nfunc Gamma() {}\n"), 0o644)

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		stats, _ := st.Stats(ctx)
		if stats.Files >= 2 && stats.Chunks > before.Chunks {
			cancel()
			<-done
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	cancel()
	<-done
	stats, _ := st.Stats(ctx)
	t.Fatalf("watch did not reindex after save: before=%d/%d after=%d/%d",
		before.Files, before.Chunks, stats.Files, stats.Chunks)
}

// TestWatchSingleFlight verifies that bursts of events while a re-index
// is in flight do not spawn a second concurrent indexer (which would
// race on the SQLite writer lock and surface "database is locked"
// errors to the operator). All events end up reflected in the index
// regardless of how rapidly they arrived.
func TestWatchSingleFlight(t *testing.T) {
	srv := fakeEmbedServer(t)
	defer srv.Close()

	projDir := t.TempDir()
	cacheDir := t.TempDir()
	_ = os.WriteFile(filepath.Join(projDir, "seed.go"),
		[]byte("package main\nfunc Seed() {}\n"), 0o644)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, _ := store.Open(ctx, p.DBPath)
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(srv.URL, "fake", 8, 5*time.Second)
	ix := index.New(p, st, em, ig, index.Options{})
	// Very short debounce so the timer fires while bursts are still arriving.
	w := New(ix, ig, p.Root, Options{Debounce: 5 * time.Millisecond})

	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()

	// Fire 50 file-create bursts as fast as we can.
	const N = 50
	for i := range N {
		_ = os.WriteFile(filepath.Join(projDir, fmt.Sprintf("f%d.go", i)),
			fmt.Appendf(nil, "package main\nfunc F%d() {}\n", i), 0o644)
	}

	// All N files should end up indexed (plus the seed). If a second
	// concurrent flush had clobbered last_seen_at or wedged on the
	// writer lock, the count would be off.
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		stats, _ := st.Stats(ctx)
		if stats.Files >= N+1 {
			cancel()
			<-done
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	cancel()
	<-done
	stats, _ := st.Stats(ctx)
	t.Fatalf("only %d/%d files reached the index", stats.Files, N+1)
}
