package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alehatsman/dex/internal/chat"
	"github.com/alehatsman/dex/internal/chunk"
	"github.com/alehatsman/dex/internal/embed"
	"github.com/alehatsman/dex/internal/ignore"
	"github.com/alehatsman/dex/internal/index"
	"github.com/alehatsman/dex/internal/proj"
	"github.com/alehatsman/dex/internal/rerank"
	"github.com/alehatsman/dex/internal/store"
)

func TestNewRerankClientNilWhenURLEmpty(t *testing.T) {
	t.Setenv("DEX_RERANK_URL", "")
	if c := newRerankClient(); c != nil {
		t.Errorf("newRerankClient() = %+v, want nil when URL unset", c)
	}
}

func TestNewRerankClientReturnsClientWhenURLSet(t *testing.T) {
	t.Setenv("DEX_RERANK_URL", "http://127.0.0.1:9999")
	t.Setenv("DEX_RERANK_MODEL", "custom-reranker")
	t.Setenv("DEX_DISABLE_RERANK", "")
	t.Setenv("DEX_RERANK_STYLE", "") // ensure Cohere-style client

	c := newRerankClient()
	if c == nil {
		t.Fatal("newRerankClient() = nil, want non-nil when URL is set")
	}
	if c.Endpoint() != "http://127.0.0.1:9999" {
		t.Errorf("Endpoint() = %q, want http://127.0.0.1:9999", c.Endpoint())
	}
	if c.ModelName() != "custom-reranker" {
		t.Errorf("ModelName() = %q, want custom-reranker", c.ModelName())
	}
}

func TestNewRerankClientChatStyleWhenRequested(t *testing.T) {
	t.Setenv("DEX_RERANK_URL", "http://127.0.0.1:9999")
	t.Setenv("DEX_RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")
	t.Setenv("DEX_RERANK_STYLE", "chat")
	t.Setenv("DEX_DISABLE_RERANK", "")

	c := newRerankClient()
	if c == nil {
		t.Fatal("newRerankClient() = nil, want non-nil when URL is set")
	}
	if _, ok := c.(*rerank.ChatReranker); !ok {
		t.Errorf("expected *rerank.ChatReranker, got %T", c)
	}
	if c.Endpoint() != "http://127.0.0.1:9999" {
		t.Errorf("Endpoint() = %q, want http://127.0.0.1:9999", c.Endpoint())
	}
}

func TestNewRerankClientNilWhenDisableSet(t *testing.T) {
	// URL is set, but the kill switch is on. nil should still be returned.
	t.Setenv("DEX_RERANK_URL", "http://127.0.0.1:9999")
	t.Setenv("DEX_DISABLE_RERANK", "1")

	if c := newRerankClient(); c != nil {
		t.Errorf("newRerankClient() = %+v, want nil when DISABLE_RERANK=1", c)
	}
}

func TestNewRerankClientDefaultTimeout(t *testing.T) {
	t.Setenv("DEX_RERANK_URL", "http://127.0.0.1:9999")
	t.Setenv("DEX_RERANK_TIMEOUT", "")
	t.Setenv("DEX_DISABLE_RERANK", "")
	t.Setenv("DEX_RERANK_STYLE", "") // Cohere-style; access concrete type

	c := newRerankClient()
	if c == nil {
		t.Fatal("expected non-nil client")
	}
	cc, ok := c.(*rerank.Client)
	if !ok {
		t.Fatalf("expected *rerank.Client, got %T", c)
	}
	if cc.HTTP.Timeout != 5*time.Second {
		t.Errorf("default timeout = %s, want 5s", cc.HTTP.Timeout)
	}
}

func TestRerankPoolDefault(t *testing.T) {
	t.Setenv("DEX_RERANK_POOL", "")
	if got := rerankPool(); got != 40 {
		t.Errorf("rerankPool() = %d, want 40 (default)", got)
	}
}

func TestRerankPoolHonoredInRange(t *testing.T) {
	t.Setenv("DEX_RERANK_POOL", "60")
	if got := rerankPool(); got != 60 {
		t.Errorf("rerankPool() = %d, want 60", got)
	}
}

func TestRerankPoolClampsHigh(t *testing.T) {
	t.Setenv("DEX_RERANK_POOL", "9999")
	if got := rerankPool(); got != 100 {
		t.Errorf("rerankPool() = %d, want 100 (clamped)", got)
	}
}

func TestRerankPoolFallbackOnInvalid(t *testing.T) {
	t.Setenv("DEX_RERANK_POOL", "not-an-int")
	if got := rerankPool(); got != 40 {
		t.Errorf("rerankPool() = %d, want 40 (fallback after warning)", got)
	}
}

func TestRerankPoolFallbackOnNonPositive(t *testing.T) {
	t.Setenv("DEX_RERANK_POOL", "0")
	if got := rerankPool(); got != 40 {
		t.Errorf("rerankPool() = %d, want 40 (zero falls back)", got)
	}
	t.Setenv("DEX_RERANK_POOL", "-5")
	if got := rerankPool(); got != 40 {
		t.Errorf("rerankPool() = %d, want 40 (negative falls back)", got)
	}
}

// ─── auto-summarize wiring (idle drainer) ─────────────────────────────────

func TestAutoSummarizeEnabledDefaults(t *testing.T) {
	t.Setenv("DEX_AUTO_SUMMARIZE", "")
	t.Setenv("DEX_POWER_SAVE", "")
	t.Setenv("DEX_SUMMARY_URL", "")
	t.Setenv("DEX_CHAT_URL", "")
	if autoSummarizeEnabled() {
		t.Error("auto-summarize should be off with no chat endpoint configured")
	}

	t.Setenv("DEX_SUMMARY_URL", "http://x")
	if !autoSummarizeEnabled() {
		t.Error("auto-summarize should be on once DEX_SUMMARY_URL is set")
	}

	t.Setenv("DEX_AUTO_SUMMARIZE", "off")
	if autoSummarizeEnabled() {
		t.Error("DEX_AUTO_SUMMARIZE=off must win over the auto-on default")
	}

	t.Setenv("DEX_AUTO_SUMMARIZE", "on")
	t.Setenv("DEX_POWER_SAVE", "1")
	if autoSummarizeEnabled() {
		t.Error("DEX_POWER_SAVE=1 must override explicit on")
	}
}

func TestEnvBool(t *testing.T) {
	cases := []struct {
		raw  string
		def  bool
		want bool
	}{
		{"", false, false},
		{"", true, true},
		{"1", false, true}, {"on", false, true}, {"true", false, true}, {"yes", false, true},
		{"0", true, false}, {"off", true, false}, {"false", true, false}, {"no", true, false},
		{"weird", false, false}, // unknown values fall back to def
		{"weird", true, true},
		{"  ON  ", false, true}, // trimmed + case-insensitive
	}
	for _, c := range cases {
		t.Setenv("DEX_TEST_BOOL", c.raw)
		got := envBool("DEX_TEST_BOOL", c.def)
		if got != c.want {
			t.Errorf("envBool(%q, def=%v) = %v, want %v", c.raw, c.def, got, c.want)
		}
	}
}

// fakeEmbedServer mirrors the helper in internal/index/*_test.go so the
// cmd-level test can build an Indexer end-to-end without depending on
// a real embedding endpoint.
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

// TestIdleSummaryDrainerEndToEnd queues a few rows via a defer-mode
// index, then drives newIdleSummaryDrainer's callback until it
// signals done. Verifies the queue empties and the cascade produces
// package + repo summaries — i.e. `dex watch`'s idle hook is
// behaviourally equivalent to `dex index summarize`.
func TestIdleSummaryDrainerEndToEnd(t *testing.T) {
	var chatCalls int32
	chatSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&chatCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": "Stub."}},
			},
		})
	}))
	defer chatSrv.Close()
	embedSrv := fakeEmbedServer(t)
	defer embedSrv.Close()

	projDir := t.TempDir()
	cacheDir := t.TempDir()
	// Two source files = at least two file_summary rows.
	for i := range 2 {
		_ = os.WriteFile(filepath.Join(projDir, fmt.Sprintf("f%d.go", i)),
			fmt.Appendf(nil, "package main\nfunc F%d() {}\n", i), 0o644)
	}

	ctx := context.Background()
	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, _ := store.Open(ctx, p.DBPath)
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(embedSrv.URL, "fake", 8, 5*time.Second)
	cc := chat.New(chatSrv.URL, "fake", 10*time.Second)

	ix := index.New(p, st, em, ig, index.Options{
		Summarize:      true,
		DeferSummaries: true,
		Chat:           cc,
	})
	if err := ix.Run(ctx); err != nil {
		t.Fatalf("Run: %v", err)
	}
	startQueue, _ := st.CountPendingSummaries(ctx)
	if startQueue == 0 {
		t.Fatalf("expected queue non-empty after defer index")
	}

	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	drainer := newIdleSummaryDrainer(ix, logger, 1 /* batch */, false)
	if drainer == nil {
		t.Fatal("newIdleSummaryDrainer returned nil despite chat being configured")
	}

	// Drive the drainer like the watcher would, batch by batch.
	for i := 0; i < startQueue+5; i++ {
		done, err := drainer(ctx)
		if err != nil {
			t.Fatalf("drainer(%d): %v", i, err)
		}
		if done {
			break
		}
	}

	if remaining, _ := st.CountPendingSummaries(ctx); remaining != 0 {
		t.Errorf("queue not drained: %d remaining", remaining)
	}
	pkgs, _ := st.AllSummariesByKind(ctx, chunk.KindPackageSummary)
	if len(pkgs) == 0 {
		t.Error("expected package_summary after cascade")
	}
	repos, _ := st.AllSummariesByKind(ctx, chunk.KindRepoSummary)
	if len(repos) == 0 {
		t.Error("expected repo_summary after cascade")
	}
}

func TestIdleSummaryDrainerNilWithoutChat(t *testing.T) {
	embedSrv := fakeEmbedServer(t)
	defer embedSrv.Close()
	projDir := t.TempDir()
	cacheDir := t.TempDir()

	ctx := context.Background()
	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, _ := store.Open(ctx, p.DBPath)
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(embedSrv.URL, "fake", 8, 5*time.Second)

	ix := index.New(p, st, em, ig, index.Options{}) // no Chat
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	if d := newIdleSummaryDrainer(ix, logger, 10, false); d != nil {
		t.Error("expected nil drainer when chat is not configured")
	}
}

// TestIdleSummaryDrainerStopsOnNoProgress simulates a misconfigured
// chat endpoint: every drain attempt fails so the queue depth doesn't
// move. The drainer must return done=true to break the idle cycle
// rather than spinning until the user closes the watcher.
func TestIdleSummaryDrainerStopsOnNoProgress(t *testing.T) {
	chatSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer chatSrv.Close()
	embedSrv := fakeEmbedServer(t)
	defer embedSrv.Close()

	projDir := t.TempDir()
	cacheDir := t.TempDir()
	_ = os.WriteFile(filepath.Join(projDir, "f.go"),
		[]byte("package main\nfunc F() {}\n"), 0o644)

	ctx := context.Background()
	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, _ := store.Open(ctx, p.DBPath)
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(embedSrv.URL, "fake", 8, 5*time.Second)
	cc := chat.New(chatSrv.URL, "fake", 2*time.Second)

	ix := index.New(p, st, em, ig, index.Options{
		Summarize: true, DeferSummaries: true, Chat: cc,
	})
	if err := ix.Run(ctx); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if n, _ := st.CountPendingSummaries(ctx); n == 0 {
		t.Fatal("expected pending rows")
	}

	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	drainer := newIdleSummaryDrainer(ix, logger, 10, false)
	done, err := drainer(ctx)
	if err != nil {
		t.Fatalf("drainer: %v", err)
	}
	if !done {
		t.Error("drainer should signal done=true when no progress was made (avoids busy loop)")
	}
}
