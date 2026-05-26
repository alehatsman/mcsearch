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
	"sync/atomic"
	"testing"
	"time"

	"github.com/alehatsman/dex/internal/embed"
	"github.com/alehatsman/dex/internal/ignore"
	"github.com/alehatsman/dex/internal/index"
	"github.com/alehatsman/dex/internal/proj"
	"github.com/alehatsman/dex/internal/store"
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

// newIdleTestRig builds the boilerplate (project, store, indexer,
// matcher, embed client) used by the OnIdle tests. Callers supply the
// OnIdle callback shape they want to assert against.
func newIdleTestRig(t *testing.T) (context.Context, context.CancelFunc, *index.Indexer, *ignore.Matcher, *proj.Project, string) {
	t.Helper()
	srv := fakeEmbedServer(t)
	t.Cleanup(srv.Close)

	projDir := t.TempDir()
	cacheDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(projDir, "seed.go"),
		[]byte("package main\nfunc Seed() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, err := store.Open(ctx, p.DBPath)
	if err != nil {
		cancel()
		t.Fatalf("store.Open: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	ig, _ := ignore.New(p.Root)
	em := embed.New(srv.URL, "fake", 8, 5*time.Second)
	ix := index.New(p, st, em, ig, index.Options{})
	return ctx, cancel, ix, ig, p, projDir
}

// TestWatchIdleFires verifies the OnIdle hook is invoked once the
// watcher has been quiet for OnIdleAfter, and only once when the hook
// returns done=true.
func TestWatchIdleFires(t *testing.T) {
	ctx, cancel, ix, ig, p, _ := newIdleTestRig(t)
	defer cancel()

	idleFired := make(chan struct{}, 4)
	w := New(ix, ig, p.Root, Options{
		Debounce:    5 * time.Millisecond,
		OnIdleAfter: 20 * time.Millisecond,
		OnIdle: func(c context.Context) (bool, error) {
			idleFired <- struct{}{}
			return true, nil
		},
	})

	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()

	select {
	case <-idleFired:
	case <-time.After(2 * time.Second):
		t.Fatal("OnIdle never fired")
	}

	// done=true should stop the cycle; no further fires.
	select {
	case <-idleFired:
		t.Error("OnIdle re-fired despite done=true")
	case <-time.After(100 * time.Millisecond):
	}

	cancel()
	<-done
}

// TestWatchIdleCanceledByEvent verifies that a fresh fs event during
// an in-flight OnIdle cancels the hook's ctx, so a long drain yields
// immediately to the user's edit.
func TestWatchIdleCanceledByEvent(t *testing.T) {
	ctx, cancel, ix, ig, p, projDir := newIdleTestRig(t)
	defer cancel()

	idleStarted := make(chan struct{}, 1)
	idleCancelled := make(chan struct{}, 1)
	w := New(ix, ig, p.Root, Options{
		Debounce:    5 * time.Millisecond,
		OnIdleAfter: 10 * time.Millisecond,
		OnIdle: func(c context.Context) (bool, error) {
			select {
			case idleStarted <- struct{}{}:
			default:
			}
			select {
			case <-c.Done():
				select {
				case idleCancelled <- struct{}{}:
				default:
				}
				return true, c.Err()
			case <-time.After(3 * time.Second):
				return true, nil
			}
		},
	})

	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()

	select {
	case <-idleStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("idle never started")
	}

	// Trigger a fs event mid-idle; OnIdle's ctx should cancel promptly.
	if err := os.WriteFile(filepath.Join(projDir, "next.go"),
		[]byte("package main\nfunc Next() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	select {
	case <-idleCancelled:
	case <-time.After(2 * time.Second):
		t.Fatal("idle ctx never cancelled by fs event")
	}

	cancel()
	<-done
}

// TestWatchIdleReArm verifies OnIdle is re-invoked when it returns
// done=false. The hook returns false twice before returning true on
// the third call, simulating a batched drainer that needs more passes.
func TestWatchIdleReArm(t *testing.T) {
	ctx, cancel, ix, ig, p, _ := newIdleTestRig(t)
	defer cancel()

	var count int32
	idleDone := make(chan struct{}, 1)
	w := New(ix, ig, p.Root, Options{
		Debounce:    5 * time.Millisecond,
		OnIdleAfter: 10 * time.Millisecond,
		OnIdle: func(c context.Context) (bool, error) {
			n := atomic.AddInt32(&count, 1)
			if n >= 3 {
				select {
				case idleDone <- struct{}{}:
				default:
				}
				return true, nil
			}
			return false, nil
		},
	})

	done := make(chan error, 1)
	go func() { done <- w.Run(ctx) }()

	select {
	case <-idleDone:
	case <-time.After(3 * time.Second):
		t.Fatalf("OnIdle did not re-arm to 3 calls; got %d", atomic.LoadInt32(&count))
	}
	if got := atomic.LoadInt32(&count); got != 3 {
		t.Errorf("expected 3 idle calls, got %d", got)
	}

	cancel()
	<-done
}
