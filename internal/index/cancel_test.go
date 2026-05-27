package index

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/alehatsman/dex/internal/embed"
	"github.com/alehatsman/dex/internal/ignore"
	"github.com/alehatsman/dex/internal/proj"
	"github.com/alehatsman/dex/internal/store"
)

// slowEmbedServer blocks every /v1/embeddings request until either
// r.Context() (the client side) or `stop` (the test cleanup) fires.
// Lets a test prove ix.Run honours ctx.Done mid-batch without dragging
// the test's defer srv.Close() through the worst-case http timeout.
func slowEmbedServer(t *testing.T) (*httptest.Server, func()) {
	t.Helper()
	stop := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
			return
		case <-stop:
			return
		}
	}))
	return srv, func() {
		close(stop)
		srv.Close()
	}
}

// TestRunReturnsPromptlyOnCancel proves that ix.Run honours ctx.Done
// between embed batches: a cancel mid-batch should return within ~1s
// rather than waiting for the http client's full per-call timeout (60s
// in production).
func TestRunReturnsPromptlyOnCancel(t *testing.T) {
	srv, teardown := slowEmbedServer(t)
	defer teardown()

	projDir := t.TempDir()
	cacheDir := t.TempDir()
	// Enough files to ensure embedding actually starts (Pass 4).
	for i := 0; i < 5; i++ {
		writeFile(t, filepath.Join(projDir, "f"+itoa(i)+".go"),
			"package main\n\nfunc F"+itoa(i)+"() string { return \"x\" }\n")
	}

	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, _ := store.Open(context.Background(), p.DBPath)
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(srv.URL, "fake", 8, 60*time.Second)

	ix := New(p, st, em, ig, Options{Verbose: false})

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	err := ix.Run(ctx)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatalf("Run returned nil after cancel; expected an error")
	}
	// 3s gives generous slack for the http client to notice ctx via
	// transport cancellation. Production target is <1s.
	if elapsed > 3*time.Second {
		t.Errorf("Run after cancel took %v; expected <3s (target <1s)", elapsed)
	}
}

// itoa avoids pulling strconv just for one helper.
func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var buf [20]byte
	pos := len(buf)
	neg := i < 0
	if neg {
		i = -i
	}
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}
