package index

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alehatsman/dex/internal/chat"
	"github.com/alehatsman/dex/internal/chunk"
	"github.com/alehatsman/dex/internal/embed"
	"github.com/alehatsman/dex/internal/ignore"
	"github.com/alehatsman/dex/internal/proj"
	"github.com/alehatsman/dex/internal/store"
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
// .dex-ignore (or .gitignore) between runs evicts the chunks that
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
	writeFile(t, filepath.Join(projDir, ".dex-ignore"), "drafts/\n")
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

// TestSummarizeMtimeFastPath asserts that a re-index with --summarize on an
// unchanged file takes the mtime fast-path (no file read, no chunk parse),
// preserves file_summary / chunk_summary / package_summary rows, and does
// not regenerate any summaries via chat.
func TestSummarizeMtimeFastPath(t *testing.T) {
	var chatCalls int32
	chatSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&chatCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": "summary text"}},
			},
		})
	}))
	defer chatSrv.Close()

	embedSrv := fakeEmbedServer(t)
	defer embedSrv.Close()

	projDir := t.TempDir()
	cacheDir := t.TempDir()

	// Two files in a single dir → exercises the package_summary keepalive
	// path. One file is large enough to trigger a chunk_summary.
	body := "package main\n\n// LongFunc is deliberately long.\nfunc LongFunc() {\n"
	for i := 0; i < chunkSummaryMinLines; i++ {
		body += fmt.Sprintf("\t// line %d\n", i+1)
	}
	body += "}\n"
	writeFile(t, filepath.Join(projDir, "pkg", "long.go"), body)
	writeFile(t, filepath.Join(projDir, "pkg", "tiny.go"), "package main\nfunc Tiny() {}\n")

	ctx := context.Background()
	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, _ := store.Open(ctx, p.DBPath)
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(embedSrv.URL, "fake", 8, 5*time.Second)
	cc := chat.New(chatSrv.URL, "fake", 10*time.Second)

	// First run: populates file_summary, chunk_summary, package_summary, repo_summary.
	ix1 := New(p, st, em, ig, Options{Summarize: true, Chat: cc})
	if err := ix1.Run(ctx); err != nil {
		t.Fatalf("Run #1: %v", err)
	}
	callsAfterFirst := atomic.LoadInt32(&chatCalls)
	if callsAfterFirst == 0 {
		t.Fatal("expected chat calls during first summarize run; got 0")
	}

	// Confirm the various summary kinds landed in the store.
	chunksBefore := countByKind(t, ctx, st)
	for _, k := range []string{chunk.KindFileSummary, chunk.KindChunkSummary, chunk.KindPackageSummary, chunk.KindRepoSummary} {
		if chunksBefore[k] == 0 {
			t.Fatalf("expected at least one %s row after run #1; counts=%v", k, chunksBefore)
		}
	}

	// Second run: every file's mtime is < lastIndexed (set during run #1),
	// so every file is eligible for the new summarize fast-path. Capture
	// logs to verify the fast-path actually fired.
	var logBuf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&logBuf, &slog.HandlerOptions{Level: slog.LevelInfo}))
	ix2 := New(p, st, em, ig, Options{Summarize: true, Chat: cc, Verbose: true, Logger: logger})
	if err := ix2.Run(ctx); err != nil {
		t.Fatalf("Run #2: %v", err)
	}

	if got := atomic.LoadInt32(&chatCalls); got != callsAfterFirst {
		t.Errorf("second run made %d extra chat calls; expected 0", got-callsAfterFirst)
	}
	logs := logBuf.String()
	if !strings.Contains(logs, "files_fast_path=") || strings.Contains(logs, "files_fast_path=0") {
		t.Errorf("expected files_fast_path>0 in indexed log line; logs:\n%s", logs)
	}

	// Every summary kind must still be present — PruneUnseen would have
	// dropped them if last_seen_at wasn't refreshed.
	chunksAfter := countByKind(t, ctx, st)
	for _, k := range []string{chunk.KindFileSummary, chunk.KindChunkSummary, chunk.KindPackageSummary, chunk.KindRepoSummary} {
		if chunksAfter[k] != chunksBefore[k] {
			t.Errorf("%s count changed: before=%d after=%d (fast-path should preserve all summary rows)",
				k, chunksBefore[k], chunksAfter[k])
		}
	}
}

func countByKind(t *testing.T, ctx context.Context, st *store.Store) map[string]int {
	t.Helper()
	out := make(map[string]int)
	for _, k := range []string{chunk.KindFileSummary, chunk.KindChunkSummary, chunk.KindPackageSummary, chunk.KindRepoSummary} {
		s, err := st.AllSummariesByKind(ctx, k)
		if err != nil {
			t.Fatalf("AllSummariesByKind(%s): %v", k, err)
		}
		out[k] = len(s)
	}
	return out
}

// TestDrainPendingSummariesEndToEnd is the round-trip test for Layer 3:
// run an index with DeferSummaries=true (no chat calls), then invoke
// DrainPendingSummaries with a stub chat. Verify that:
//   - the drainer calls chat exactly once per pending row;
//   - chunks table receives file_summary, chunk_summary,
//     package_summary, and repo_summary entries with the expected SHAs;
//   - pending_summaries empties;
//   - a second drain on an empty queue is a no-op (no chat calls).
func TestDrainPendingSummariesEndToEnd(t *testing.T) {
	var chatCalls int32
	chatSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&chatCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": "Stub summary."}},
			},
		})
	}))
	defer chatSrv.Close()

	embedSrv := fakeEmbedServer(t)
	defer embedSrv.Close()

	projDir := t.TempDir()
	cacheDir := t.TempDir()

	writeFile(t, filepath.Join(projDir, "short.go"),
		"package main\n\nfunc S() string { return \"x\" }\n")
	long := "package main\n\nfunc LongFunc() {\n"
	for i := range chunkSummaryMinLines {
		long += fmt.Sprintf("\t// line %d\n", i+1)
	}
	long += "}\n"
	writeFile(t, filepath.Join(projDir, "long.go"), long)

	ctx := context.Background()
	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, _ := store.Open(ctx, p.DBPath)
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(embedSrv.URL, "fake", 8, 5*time.Second)
	cc := chat.New(chatSrv.URL, "fake", 10*time.Second)

	// Phase: index in defer mode. No chat calls yet.
	ix := New(p, st, em, ig, Options{
		Summarize:          true,
		DeferSummaries:     true,
		Chat:               cc,
		SummaryConcurrency: 4,
	})
	if err := ix.Run(ctx); err != nil {
		t.Fatalf("Run (index defer): %v", err)
	}
	if got := atomic.LoadInt32(&chatCalls); got != 0 {
		t.Fatalf("defer-mode index must not call chat; got %d", got)
	}

	startQueue, _ := st.CountPendingSummaries(ctx)
	if startQueue == 0 {
		t.Fatalf("expected queue to be non-empty after defer-mode index")
	}

	// Phase: drain. Expect chat to be called for each pending row, plus
	// package_summary + repo_summary cascade calls.
	generated, err := ix.DrainPendingSummaries(ctx)
	if err != nil {
		t.Fatalf("DrainPendingSummaries: %v", err)
	}
	if generated < startQueue {
		t.Errorf("expected at least %d generated (queue depth); got %d", startQueue, generated)
	}

	endQueue, _ := st.CountPendingSummaries(ctx)
	if endQueue != 0 {
		t.Errorf("queue should be empty after drain; got %d", endQueue)
	}

	chatAfterDrain := atomic.LoadInt32(&chatCalls)
	if int(chatAfterDrain) < startQueue {
		t.Errorf("expected ≥ %d chat calls during drain; got %d", startQueue, chatAfterDrain)
	}

	// Verify the chunks table actually got the summary chunks.
	stats, _ := st.Stats(ctx)
	if stats.Chunks < startQueue {
		t.Errorf("expected at least %d chunks after drain; got %d", startQueue, stats.Chunks)
	}
	// Spot-check that the cascaded summaries landed.
	pkgSummaries, _ := st.AllSummariesByKind(ctx, chunk.KindPackageSummary)
	if len(pkgSummaries) == 0 {
		t.Errorf("expected at least one package_summary chunk after cascade")
	}
	repoSummaries, _ := st.AllSummariesByKind(ctx, chunk.KindRepoSummary)
	if len(repoSummaries) == 0 {
		t.Errorf("expected one repo_summary chunk after cascade")
	}

	// Phase: drain again. Queue is empty, so no new chat calls
	// (the cascade also no-ops since all package/repo summaries are
	// already present with matching SHAs).
	atomic.StoreInt32(&chatCalls, 0)
	gen2, err := ix.DrainPendingSummaries(ctx)
	if err != nil {
		t.Fatalf("DrainPendingSummaries (2nd): %v", err)
	}
	if gen2 != 0 {
		t.Errorf("expected 0 generated on 2nd drain; got %d", gen2)
	}
	if got := atomic.LoadInt32(&chatCalls); got != 0 {
		t.Errorf("2nd drain should not call chat; got %d", got)
	}
}

// TestDrainDropsStaleFileSummary verifies that if a file's content
// changes between enqueue and drain, the drainer drops the stale
// pending row instead of generating an incorrectly-SHA-keyed chunk.
func TestDrainDropsStaleFileSummary(t *testing.T) {
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
	target := filepath.Join(projDir, "f.go")
	writeFile(t, target, "package main\n\nfunc F1() {}\n")

	ctx := context.Background()
	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, _ := store.Open(ctx, p.DBPath)
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(embedSrv.URL, "fake", 8, 5*time.Second)
	cc := chat.New(chatSrv.URL, "fake", 10*time.Second)

	ix := New(p, st, em, ig, Options{
		Summarize:      true,
		DeferSummaries: true,
		Chat:           cc,
	})
	if err := ix.Run(ctx); err != nil {
		t.Fatalf("Run: %v", err)
	}

	// Mutate the file out from under the pending row.
	writeFile(t, target, "package main\n\nfunc Different() {}\n")

	if _, err := ix.DrainPendingSummaries(ctx); err != nil {
		t.Fatalf("Drain: %v", err)
	}

	// The file_summary pending row should have been dropped without
	// producing a chunk — its enqueued ContentSHA no longer matches
	// the file's current content. (Cascade still tries to run, but
	// FileSummariesForPaths returns nothing for f.go so nothing is
	// produced for the package layer either.)
	remaining, _ := st.CountPendingSummaries(ctx)
	if remaining != 0 {
		t.Errorf("stale rows should be dropped; got %d remaining", remaining)
	}
	// No file_summary chunk for f.go (the source content changed and
	// we dropped the pending row).
	shas, _ := st.FileSummarySHAs(ctx)
	if _, ok := shas["f.go"]; ok {
		t.Errorf("expected no file_summary for f.go (was stale); got SHA in store")
	}
}

// TestDrainPendingSummariesBatchRespectsMax verifies the bounded
// drainer used by the watcher's idle hook: each call processes at
// most `max` queued rows, leaves the queue at the expected depth,
// and does NOT cascade — package/repo summaries only appear once the
// caller invokes CascadePackageRepoSummaries.
func TestDrainPendingSummariesBatchRespectsMax(t *testing.T) {
	var chatCalls int32
	chatSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&chatCalls, 1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": "Stub summary."}},
			},
		})
	}))
	defer chatSrv.Close()

	embedSrv := fakeEmbedServer(t)
	defer embedSrv.Close()

	projDir := t.TempDir()
	cacheDir := t.TempDir()

	// Four files, each with a long structural chunk — that gives the
	// queue 8 rows (4 file_summary + 4 chunk_summary) so a max=3 batch
	// leaves a meaningful remainder.
	long := "package main\n\nfunc Long%d() {\n"
	for i := range chunkSummaryMinLines {
		long += fmt.Sprintf("\t// line %d\n", i+1)
	}
	long += "}\n"
	for i := range 4 {
		writeFile(t, filepath.Join(projDir, fmt.Sprintf("f%d.go", i)),
			fmt.Sprintf(long, i))
	}

	ctx := context.Background()
	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, _ := store.Open(ctx, p.DBPath)
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(embedSrv.URL, "fake", 8, 5*time.Second)
	cc := chat.New(chatSrv.URL, "fake", 10*time.Second)

	ix := New(p, st, em, ig, Options{
		Summarize:          true,
		DeferSummaries:     true,
		Chat:               cc,
		SummaryConcurrency: 4,
	})
	if err := ix.Run(ctx); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := atomic.LoadInt32(&chatCalls); got != 0 {
		t.Fatalf("defer-mode index must not call chat; got %d", got)
	}

	start, _ := st.CountPendingSummaries(ctx)
	if start < 4 {
		t.Fatalf("expected ≥4 pending rows; got %d", start)
	}

	// First batch: max=3 → processes 3 rows, leaves the rest.
	gen1, remaining1, err := ix.DrainPendingSummariesBatch(ctx, 3)
	if err != nil {
		t.Fatalf("Batch(3): %v", err)
	}
	if gen1 != 3 {
		t.Errorf("Batch(3) generated: want 3, got %d", gen1)
	}
	if remaining1 != start-3 {
		t.Errorf("Batch(3) remaining: want %d, got %d", start-3, remaining1)
	}

	// Cascade has NOT run yet — no package_summary / repo_summary chunks.
	pkgs, _ := st.AllSummariesByKind(ctx, chunk.KindPackageSummary)
	if len(pkgs) != 0 {
		t.Errorf("Batch must not cascade; saw %d package_summary chunks", len(pkgs))
	}
	repos, _ := st.AllSummariesByKind(ctx, chunk.KindRepoSummary)
	if len(repos) != 0 {
		t.Errorf("Batch must not cascade; saw %d repo_summary chunks", len(repos))
	}

	// Second batch: max=0 (no limit) → drains the rest.
	gen2, remaining2, err := ix.DrainPendingSummariesBatch(ctx, 0)
	if err != nil {
		t.Fatalf("Batch(0): %v", err)
	}
	if gen2 != start-3 {
		t.Errorf("Batch(0) generated: want %d, got %d", start-3, gen2)
	}
	if remaining2 != 0 {
		t.Errorf("Batch(0) remaining: want 0, got %d", remaining2)
	}

	// Third batch on an empty queue: no-op.
	atomic.StoreInt32(&chatCalls, 0)
	gen3, remaining3, err := ix.DrainPendingSummariesBatch(ctx, 0)
	if err != nil {
		t.Fatalf("Batch(empty): %v", err)
	}
	if gen3 != 0 || remaining3 != 0 {
		t.Errorf("empty batch: want 0/0, got %d/%d", gen3, remaining3)
	}
	if got := atomic.LoadInt32(&chatCalls); got != 0 {
		t.Errorf("empty batch must not call chat; got %d", got)
	}

	// Now cascade — package and repo summaries should appear.
	cascadeGen, err := ix.CascadePackageRepoSummaries(ctx)
	if err != nil {
		t.Fatalf("CascadePackageRepoSummaries: %v", err)
	}
	if cascadeGen == 0 {
		t.Error("cascade should generate at least one summary")
	}
	pkgs, _ = st.AllSummariesByKind(ctx, chunk.KindPackageSummary)
	if len(pkgs) == 0 {
		t.Error("cascade should have produced package_summary chunks")
	}
	repos, _ = st.AllSummariesByKind(ctx, chunk.KindRepoSummary)
	if len(repos) == 0 {
		t.Error("cascade should have produced a repo_summary chunk")
	}
}

// TestCascadePackageRepoSummariesEmpty verifies cascade is a safe
// no-op when no file_summary chunks exist (e.g. caller skipped the
// batch drain entirely).
func TestCascadePackageRepoSummariesEmpty(t *testing.T) {
	embedSrv := fakeEmbedServer(t)
	defer embedSrv.Close()
	// Chat is wired but should never be called — cascade has no inputs.
	chatSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("chat should not be called when there are no file summaries")
		http.Error(w, "no", 500)
	}))
	defer chatSrv.Close()

	projDir := t.TempDir()
	cacheDir := t.TempDir()

	ctx := context.Background()
	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, _ := store.Open(ctx, p.DBPath)
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(embedSrv.URL, "fake", 8, 5*time.Second)
	cc := chat.New(chatSrv.URL, "fake", 10*time.Second)

	ix := New(p, st, em, ig, Options{Chat: cc})
	gen, err := ix.CascadePackageRepoSummaries(ctx)
	if err != nil {
		t.Fatalf("CascadePackageRepoSummaries: %v", err)
	}
	if gen != 0 {
		t.Errorf("empty-state cascade: want 0 generated, got %d", gen)
	}
}

// TestDeferSummariesEnqueuesWithoutChat verifies that DeferSummaries=true
// makes the indexer queue summary jobs into pending_summaries instead of
// running them through the chat client. The chat endpoint is stubbed
// with a counter so we can assert it's never called. Package and repo
// summaries are skipped entirely in defer mode — they have cascading
// dependencies on file_summary chunks that don't exist yet, so the
// drainer (Phase 3) will handle them later.
func TestDeferSummariesEnqueuesWithoutChat(t *testing.T) {
	var chatCalls int32
	chatSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&chatCalls, 1)
		http.Error(w, "chat should not be called in defer mode", 500)
	}))
	defer chatSrv.Close()

	embedSrv := fakeEmbedServer(t)
	defer embedSrv.Close()

	projDir := t.TempDir()
	cacheDir := t.TempDir()

	// One short file (file_summary only) and one with a long structural
	// chunk (both file_summary and chunk_summary).
	writeFile(t, filepath.Join(projDir, "short.go"),
		"package main\n\nfunc S() string { return \"x\" }\n")

	long := "package main\n\nfunc LongFunc() {\n"
	for i := range chunkSummaryMinLines {
		long += fmt.Sprintf("\t// line %d\n", i+1)
	}
	long += "}\n"
	writeFile(t, filepath.Join(projDir, "long.go"), long)

	ctx := context.Background()
	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, _ := store.Open(ctx, p.DBPath)
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(embedSrv.URL, "fake", 8, 5*time.Second)
	cc := chat.New(chatSrv.URL, "fake", 10*time.Second)

	ix := New(p, st, em, ig, Options{
		Summarize:      true,
		DeferSummaries: true,
		Chat:           cc,
	})
	if err := ix.Run(ctx); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if got := atomic.LoadInt32(&chatCalls); got != 0 {
		t.Errorf("defer mode must not call chat; got %d calls", got)
	}

	pending, err := st.ListPendingSummaries(ctx, 0)
	if err != nil {
		t.Fatal(err)
	}

	var fileCount, chunkCount, pkgCount, repoCount int
	for _, p := range pending {
		switch p.Kind {
		case chunk.KindFileSummary:
			fileCount++
		case chunk.KindChunkSummary:
			chunkCount++
		case chunk.KindPackageSummary:
			pkgCount++
		case chunk.KindRepoSummary:
			repoCount++
		}
	}
	if fileCount != 2 {
		t.Errorf("expected 2 file_summary pending rows (short.go + long.go); got %d", fileCount)
	}
	if chunkCount != 1 {
		t.Errorf("expected 1 chunk_summary pending row (LongFunc); got %d", chunkCount)
	}
	if pkgCount != 0 {
		t.Errorf("package_summary should be skipped in defer mode; got %d", pkgCount)
	}
	if repoCount != 0 {
		t.Errorf("repo_summary should be skipped in defer mode; got %d", repoCount)
	}

	// Source chunks must still be indexed and embedded — defer mode only
	// changes the summary handling, not the chunk pipeline.
	stats, _ := st.Stats(ctx)
	if stats.Files < 2 {
		t.Errorf("source files should still be indexed; got %d", stats.Files)
	}

	// Re-running in defer mode is idempotent — same pending rows, no
	// chat calls.
	if err := ix.Run(ctx); err != nil {
		t.Fatalf("Run (2nd): %v", err)
	}
	if got := atomic.LoadInt32(&chatCalls); got != 0 {
		t.Errorf("2nd defer run still must not call chat; got %d", got)
	}
	pending2, _ := st.ListPendingSummaries(ctx, 0)
	if len(pending2) != len(pending) {
		t.Errorf("re-run should not duplicate pending rows: before=%d after=%d", len(pending), len(pending2))
	}
}

// TestSummarizeConcurrencyParallelizesAcrossFiles pins the perf invariant
// from Layer 1 of the indexing-perf plan: file_summary chat calls run in a
// single global pool across all slowFiles rather than serializing per file.
// With per-call latency dominating, an 8-file run at SummaryConcurrency=4
// should finish in roughly two waves (~2× call latency + overhead), not
// eight (~8× call latency).
//
// The bound is intentionally loose so the test isn't flaky on slow boxes:
// a fully serial implementation at 8 calls × 100 ms = 800 ms would blow
// past 500 ms; a properly parallel one at ~2 waves should land near 250
// ms even with goroutine + HTTP + sqlite overhead.
func TestSummarizeConcurrencyParallelizesAcrossFiles(t *testing.T) {
	const (
		nFiles    = 8
		callDelay = 100 * time.Millisecond
		// Serial would be ~nFiles*callDelay = 800ms. Concurrency=4 should
		// be ~2*callDelay + overhead ≈ 250ms. 500ms is the wedge.
		upperBound = 500 * time.Millisecond
	)

	var chatCalls int32
	chatSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&chatCalls, 1)
		time.Sleep(callDelay)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"role": "assistant", "content": "Stub summary."}},
			},
		})
	}))
	defer chatSrv.Close()

	embedSrv := fakeEmbedServer(t)
	defer embedSrv.Close()

	projDir := t.TempDir()
	cacheDir := t.TempDir()

	// Short files in distinct dirs: each produces exactly one
	// file_summary call and no chunk_summary calls (every function body
	// is under chunkSummaryMinLines). Distinct dirs would also produce
	// package_summary calls, which are themselves parallel in Layer 1 —
	// but to keep the perf invariant on just file summaries, group all
	// files in one dir.
	for i := range nFiles {
		content := fmt.Sprintf("package main\n\nfunc F%d() string { return \"x\" }\n", i)
		writeFile(t, filepath.Join(projDir, fmt.Sprintf("f%d.go", i)), content)
	}

	ctx := context.Background()
	p, _ := proj.Resolve(projDir, cacheDir)
	_ = p.EnsureCacheDir()
	st, _ := store.Open(ctx, p.DBPath)
	defer st.Close()
	ig, _ := ignore.New(p.Root)
	em := embed.New(embedSrv.URL, "fake", 8, 5*time.Second)
	cc := chat.New(chatSrv.URL, "fake", 10*time.Second)

	ix := New(p, st, em, ig, Options{
		Summarize:          true,
		Chat:               cc,
		SummaryConcurrency: 4,
	})

	start := time.Now()
	if err := ix.Run(ctx); err != nil {
		t.Fatalf("Run: %v", err)
	}
	elapsed := time.Since(start)

	gotCalls := atomic.LoadInt32(&chatCalls)
	// nFiles file summaries + 1 package summary + 1 repo summary.
	const wantCallsAtLeast = nFiles
	if gotCalls < wantCallsAtLeast {
		t.Errorf("expected ≥ %d chat calls; got %d", wantCallsAtLeast, gotCalls)
	}

	if elapsed > upperBound {
		t.Errorf("indexing %d files with SummaryConcurrency=4 took %v; "+
			"serial would be ~%v, concurrent should be ~%v. "+
			"This usually means file summaries are running serially across slowFiles.",
			nFiles, elapsed, time.Duration(nFiles)*callDelay, 2*callDelay)
	}
}
