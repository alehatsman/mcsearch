// Package mcp wires the dex toolset onto the official MCP Go SDK
// and runs it over stdio.
package mcp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/alehatsman/dex/internal/chat"
	"github.com/alehatsman/dex/internal/chunk"
	"github.com/alehatsman/dex/internal/embed"
	"github.com/alehatsman/dex/internal/ignore"
	"github.com/alehatsman/dex/internal/index"
	"github.com/alehatsman/dex/internal/proj"
	"github.com/alehatsman/dex/internal/rerank"
	"github.com/alehatsman/dex/internal/store"
	"github.com/alehatsman/dex/internal/watch"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// AutoWatchConfig configures the MCP server's lazy per-project watcher.
// Zero value (Enabled=false) disables auto-watching entirely; tools
// behave exactly as before.
type AutoWatchConfig struct {
	// Enabled toggles the per-project watcher. When true, the first
	// MCP request that resolves a project also spawns a `watch.Watcher`
	// goroutine that lives for the server's lifetime — keeping the
	// chunk index fresh as files change and (when Summarize is set)
	// filling pending summaries in the background.
	Enabled bool
	// Debounce is the quiet window between fs events before re-indexing
	// (default 500ms).
	Debounce time.Duration
	// Summarize, when true and the parent Server's ChatClient is
	// configured, enables per-flush summary queueing + the idle drainer.
	// When false the watcher only keeps the chunk index fresh.
	Summarize bool
	// OnIdleAfter is the quiet window after a flush before the summary
	// drainer fires (default 5s). Ignored when Summarize is false.
	OnIdleAfter time.Duration
	// BatchSize bounds rows per idle drain (default 10).
	BatchSize int
	// IndexConcurrency caps Pass 1 worker count (default 0 = GOMAXPROCS).
	IndexConcurrency int
	// SummaryConcurrency caps in-flight chat calls during the drain.
	SummaryConcurrency int
	// ChunkSummaryMinLines forwards to index.Options.ChunkSummaryMinLines.
	ChunkSummaryMinLines int
	// Logger receives spawn/teardown messages; nil = io.Discard.
	Logger *slog.Logger
}

// Server holds everything the MCP handlers need.
type Server struct {
	EmbedClient    *embed.Client
	ChatClient     *chat.Client         // optional — when nil, view_summarize is not registered
	SummaryClient  *chat.Client         // optional — used by the auto-watcher's background drainer; falls back to ChatClient if nil
	SummaryModels  index.SummaryModels  // optional — per-tier model overrides forwarded to the auto-watcher's indexer
	RerankClient   rerank.HealthChecker // optional — only consulted by `status` for health reporting; the actual rerank wiring goes through StoreOpts.Reranker
	CompressClient *chat.Client         // optional — health reported by status
	DraftClient    *chat.Client         // optional — health reported by status
	IndexDir       string               // base dir holding per-project index folders
	StoreOpts      store.Options        // applied to every Store opened by the server
	AutoWatch      AutoWatchConfig      // lazy per-project watcher; zero value disables

	// runCtx is set at the start of RunStdio and is used as the parent
	// context for spawned watcher goroutines. nil for non-stdio usage
	// (CLI helpers that build a Server for a single call) — ensureWatcher
	// checks this and bails so one-shot CLI tools never leak goroutines.
	runCtx context.Context
	// watchers tracks per-project watcher spawns so each project gets
	// exactly one watcher across the server's lifetime. Keyed by
	// proj.Project.ID; value is *struct{} (presence-only).
	watchers sync.Map
	// watcherWG lets RunStdio wait for all watcher goroutines to drain
	// before returning.
	watcherWG sync.WaitGroup
}

// Search, FindSymbol, Related, Summarize are thin exported wrappers
// around the unexported MCP handlers so the CLI can reuse the same
// logic that the stdio server exposes over JSON-RPC. The MCP SDK
// passes a *sdk.CallToolRequest into every handler; CLI callers
// don't have one, so the wrappers pass nil.
func (s *Server) Search(ctx context.Context, in SearchInput) (SearchOutput, error) {
	_, out, err := s.search(ctx, nil, in)
	return out, err
}

func (s *Server) FindSymbol(ctx context.Context, in FindSymbolInput) (FindSymbolOutput, error) {
	_, out, err := s.findSymbol(ctx, nil, in)
	return out, err
}

func (s *Server) Related(ctx context.Context, in RelatedInput) (RelatedOutput, error) {
	_, out, err := s.related(ctx, nil, in)
	return out, err
}

func (s *Server) Summarize(ctx context.Context, in SummarizeInput) (SummarizeOutput, error) {
	_, out, err := s.summarize(ctx, nil, in)
	return out, err
}

func (s *Server) Status(ctx context.Context) (StatusOutput, error) {
	_, out, err := s.status(ctx, nil, StatusInput{})
	return out, err
}

// ─── tool: search_semantic ────────────────────────────────────────────────

type SearchInput struct {
	Query       string   `json:"query" jsonschema:"natural-language or code query"`
	ProjectRoot string   `json:"project_root,omitempty" jsonschema:"absolute path to the project root; defaults to the server's working directory"`
	K           int      `json:"k,omitempty" jsonschema:"number of results to return (default 8, max 30)"`
	Exclude     []string `json:"exclude,omitempty" jsonschema:"path prefixes to skip (e.g. ['vendor/', 'internal/legacy/'])"`
}

type SearchHit struct {
	Path      string `json:"path"`
	Kind      string `json:"kind"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
	// Score is the cosine similarity in [-1, 1]. Always populated.
	Score float32 `json:"score"`
	// BM25Score is the lexical (FTS5) score when the hit surfaced
	// through the BM25 leg of hybrid search. Larger = better. Zero
	// for semantic-only hits.
	BM25Score float32 `json:"bm25_score,omitempty"`
	// RRFScore is the fused rank used for ordering when hybrid search
	// is active. Zero when search ran semantic-only.
	RRFScore float32 `json:"rrf_score,omitempty"`
	// RerankScore is the cross-encoder relevance score in [0, 1] when
	// rerank ran. Zero when no reranker was wired or it failed open.
	RerankScore float32 `json:"rerank_score,omitempty"`
	// Role is a compact tag describing how the symbol sits in the call
	// graph — e.g. "central:47/9pkg" (47 callers from 9 packages),
	// "leaf" (no callees), "exported-unused" (exported but no callers).
	// Empty when the symbol has no graph node (non-Go file, top-level
	// const, etc.) or sits in the unremarkable middle. See formatRole.
	Role    string `json:"role,omitempty"`
	Content string `json:"content"`
}

type SearchOutput struct {
	Status   string      `json:"status"`             // "ok" | "no-index" | "embedding-service-unreachable" | "error"
	Hint     string      `json:"hint,omitempty"`     // human-readable suggestion for the model
	Endpoint string      `json:"endpoint,omitempty"` // when unreachable
	Project  string      `json:"project,omitempty"`  // resolved project root
	Stale    bool        `json:"stale,omitempty"`    // last_indexed older than 24h
	Hits     []SearchHit `json:"hits,omitempty"`
}

// resolveProject canonicalizes projectRoot (falling back to cwd) and
// resolves it to a Project. On failure it returns a non-empty hint that
// callers can surface as a Status:"error" response.
//
// Side effect: on successful resolution under stdio mode, ensures a
// per-project watcher goroutine is running (no-op if AutoWatch is
// disabled or one is already spawned).
func (s *Server) resolveProject(projectRoot string) (*proj.Project, string) {
	root := projectRoot
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			return nil, "could not determine project root; pass project_root explicitly"
		}
		root = wd
	}
	p, err := proj.Resolve(root, s.IndexDir)
	if err != nil {
		return nil, fmt.Sprintf("resolve project: %v", err)
	}
	s.ensureWatcher(p)
	return p, ""
}

// ensureWatcher lazily spawns a Watcher goroutine for this project,
// at most once per server lifetime. No-op unless RunStdio set runCtx
// (i.e. only the stdio MCP path opts in) AND AutoWatch.Enabled is
// true. Concurrency-safe; the goroutine self-cleans when runCtx ends.
func (s *Server) ensureWatcher(p *proj.Project) {
	if s == nil || s.runCtx == nil || s.runCtx.Err() != nil {
		return
	}
	if !s.AutoWatch.Enabled {
		return
	}
	if _, loaded := s.watchers.LoadOrStore(p.ID, struct{}{}); loaded {
		return
	}
	s.watcherWG.Add(1)
	go s.runWatcher(p)
}

// runWatcher owns the lifecycle of a single project's Watcher inside
// the MCP server. Closes its store + ignores when the goroutine
// returns so RunStdio's defer s.watcherWG.Wait() drains cleanly.
func (s *Server) runWatcher(p *proj.Project) {
	defer s.watcherWG.Done()
	// On exit, free the slot — if the server is shutting down nothing
	// reads it again; if a future request hits the same project after
	// a watcher errored out, we can respawn.
	defer s.watchers.Delete(p.ID)

	logger := s.AutoWatch.Logger
	if logger == nil {
		logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}

	if err := proj.CheckIndexable(p, false); err != nil {
		logger.Info("mcp watch: skipping (not indexable)", "root", p.Root, "err", err)
		return
	}
	if err := p.EnsureCacheDir(); err != nil {
		logger.Warn("mcp watch: cache dir failed", "root", p.Root, "err", err)
		return
	}
	st, err := store.OpenWith(s.runCtx, p.DBPath, s.StoreOpts)
	if err != nil {
		logger.Warn("mcp watch: store open failed", "root", p.Root, "err", err)
		return
	}
	defer st.Close()
	ig, err := ignore.New(p.Root)
	if err != nil {
		logger.Warn("mcp watch: ignore init failed", "root", p.Root, "err", err)
		return
	}

	summaryChat := s.SummaryClient
	if summaryChat == nil {
		summaryChat = s.ChatClient
	}
	ixOpts := index.Options{
		Logger:      logger,
		Concurrency: s.AutoWatch.IndexConcurrency,
	}
	if s.AutoWatch.Summarize && summaryChat != nil {
		ixOpts.Summarize = true
		ixOpts.DeferSummaries = true
		ixOpts.Chat = summaryChat
		ixOpts.SummaryModels = s.SummaryModels
		ixOpts.SummaryConcurrency = s.AutoWatch.SummaryConcurrency
		ixOpts.ChunkSummaryMinLines = s.AutoWatch.ChunkSummaryMinLines
	}
	ix := index.New(p, st, s.EmbedClient, ig, ixOpts)

	wOpts := watch.Options{
		Debounce: s.AutoWatch.Debounce,
		Logger:   logger,
	}
	if ixOpts.Summarize {
		wOpts.OnIdle = ix.IdleSummaryDrainer(s.AutoWatch.BatchSize)
		wOpts.OnIdleAfter = s.AutoWatch.OnIdleAfter
	}
	w := watch.New(ix, ig, p.Root, wOpts)
	logger.Info("mcp watch: starting", "root", p.Root, "summarize", ixOpts.Summarize)
	if err := w.Run(s.runCtx); err != nil && !errors.Is(err, context.Canceled) {
		logger.Warn("mcp watch: exited with error", "root", p.Root, "err", err)
	}
}

func (s *Server) search(ctx context.Context, _ *sdk.CallToolRequest, in SearchInput) (*sdk.CallToolResult, SearchOutput, error) {
	out := SearchOutput{}
	if strings.TrimSpace(in.Query) == "" {
		return nil, SearchOutput{Status: "error", Hint: "query is empty — pass a natural-language description or code fragment"}, nil
	}
	p, hint := s.resolveProject(in.ProjectRoot)
	if hint != "" {
		return nil, SearchOutput{Status: "error", Hint: hint}, nil
	}
	out.Project = p.Root

	if _, err := os.Stat(p.DBPath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			out.Status = "no-index"
			out.Hint = fmt.Sprintf("no index for %s — run `dex index %s` first, then retry. Fall back to grep / Glob in the meantime.", p.Root, p.Root)
			return nil, out, nil
		}
		out.Status = "error"
		out.Hint = err.Error()
		return nil, out, nil
	}

	k := in.K
	if k <= 0 {
		k = 8
	}
	if k > 30 {
		k = 30
	}

	em := s.EmbedClient
	vecs, err := em.Embed(ctx, []string{in.Query})
	if err != nil {
		if errors.Is(err, embed.ErrUnreachable) {
			out.Status = "embedding-service-unreachable"
			out.Endpoint = em.Endpoint()
			out.Hint = "the local embedding service is offline — fall back to grep / Glob / ripgrep for this query."
			return nil, out, nil
		}
		out.Status = "error"
		out.Hint = fmt.Sprintf("embed: %v", err)
		return nil, out, nil
	}

	st, err := store.OpenWith(ctx, p.DBPath, s.StoreOpts)
	if err != nil {
		out.Status = "error"
		out.Hint = fmt.Sprintf("open index: %v", err)
		return nil, out, nil
	}
	defer st.Close()

	stats, err := st.Stats(ctx)
	if err == nil && !stats.LastIndex.IsZero() && time.Since(stats.LastIndex) > 24*time.Hour {
		out.Stale = true
		out.Hint = fmt.Sprintf("index is %s old — results may be stale; run `dex index %s` to refresh.",
			time.Since(stats.LastIndex).Round(time.Hour), p.Root)
	}

	hits, err := st.Search(ctx, vecs[0], in.Query, k)
	if err != nil {
		out.Status = "error"
		out.Hint = fmt.Sprintf("search: %v", err)
		return nil, out, nil
	}
	out.Status = "ok"
	for _, h := range hits {
		if excluded(h.Path, in.Exclude) {
			continue
		}
		out.Hits = append(out.Hits, SearchHit{
			Path:        h.Path,
			Kind:        h.Kind,
			StartLine:   h.StartLine,
			EndLine:     h.EndLine,
			Score:       h.Score,
			BM25Score:   h.BM25Score,
			RRFScore:    h.RRFScore,
			RerankScore: h.RerankScore,
			Content:     h.Content,
		})
	}
	return nil, out, nil
}

// excluded returns true when path matches any entry in the exclude list.
// An exclude entry matches if path equals it or path has it as a prefix
// (treating it as a directory prefix).
func excluded(path string, exclude []string) bool {
	for _, ex := range exclude {
		if ex == "" {
			continue
		}
		if path == ex || strings.HasPrefix(path, ex) {
			return true
		}
	}
	return false
}

// ─── tool: search_symbol ──────────────────────────────────────────────────

type FindSymbolInput struct {
	Name        string `json:"name" jsonschema:"exact identifier name to look up (case-sensitive, e.g. 'MyFunc', 'HTTPHandler')"`
	ProjectRoot string `json:"project_root,omitempty" jsonschema:"absolute path to the project root; defaults to the server's working directory"`
	K           int    `json:"k,omitempty" jsonschema:"max results to return (default 10)"`
}

type FindSymbolOutput struct {
	Status  string      `json:"status"` // "ok" | "no-index" | "not-found" | "error"
	Hint    string      `json:"hint,omitempty"`
	Project string      `json:"project,omitempty"`
	Hits    []SearchHit `json:"hits,omitempty"`
}

func (s *Server) findSymbol(ctx context.Context, _ *sdk.CallToolRequest, in FindSymbolInput) (*sdk.CallToolResult, FindSymbolOutput, error) {
	if strings.TrimSpace(in.Name) == "" {
		return nil, FindSymbolOutput{Status: "error", Hint: "name is empty"}, nil
	}
	p, hint := s.resolveProject(in.ProjectRoot)
	if hint != "" {
		return nil, FindSymbolOutput{Status: "error", Hint: hint}, nil
	}
	if _, err := os.Stat(p.DBPath); errors.Is(err, os.ErrNotExist) {
		return nil, FindSymbolOutput{Status: "no-index", Project: p.Root,
			Hint: fmt.Sprintf("no index for %s — run `dex index %s` first.", p.Root, p.Root)}, nil
	}
	st, err := store.OpenWith(ctx, p.DBPath, s.StoreOpts)
	if err != nil {
		return nil, FindSymbolOutput{Status: "error", Hint: fmt.Sprintf("open index: %v", err)}, nil
	}
	defer st.Close()
	hits, err := st.FindSymbol(ctx, in.Name, in.K)
	if err != nil {
		return nil, FindSymbolOutput{Status: "error", Hint: fmt.Sprintf("search_symbol: %v", err)}, nil
	}
	out := FindSymbolOutput{Status: "ok", Project: p.Root}
	if len(hits) == 0 {
		out.Status = "not-found"
		hint := fmt.Sprintf("no chunk with name=%q in the index; check spelling or re-index if recently added.", in.Name)
		// Near-miss surface: substring matches give the agent something
		// real to retry with instead of guessing. Errors are non-fatal —
		// the original "not-found" hint is still useful on its own.
		if cands, candErr := st.FindSymbolCandidates(ctx, in.Name, 5); candErr == nil && len(cands) > 0 {
			hint += " Did you mean: " + strings.Join(cands, ", ") + "?"
		}
		out.Hint = hint
		return nil, out, nil
	}
	for _, h := range hits {
		out.Hits = append(out.Hits, SearchHit{
			Path:      h.Path,
			Kind:      h.Kind,
			StartLine: h.StartLine,
			EndLine:   h.EndLine,
			Score:     1.0,
			Role:      formatRole(h.Name, h.InDegree, h.OutDegree, h.CrossPkgCallers),
			Content:   h.Content,
		})
	}
	return nil, out, nil
}

// ─── tool: graph_neighbors ────────────────────────────────────────────────

type RelatedInput struct {
	Path        string `json:"path" jsonschema:"relative file path of the source chunk (e.g. 'internal/store/store.go')"`
	StartLine   int    `json:"start_line" jsonschema:"start line of the source chunk (1-indexed)"`
	ProjectRoot string `json:"project_root,omitempty" jsonschema:"absolute path to the project root; defaults to the server's working directory"`
	K           int    `json:"k,omitempty" jsonschema:"number of related chunks to return (default 8, max 30)"`
}

type RelatedOutput struct {
	Status  string      `json:"status"` // "ok" | "no-index" | "not-found" | "embedding-service-unreachable" | "error"
	Hint    string      `json:"hint,omitempty"`
	Project string      `json:"project,omitempty"`
	Hits    []SearchHit `json:"hits,omitempty"`
}

func (s *Server) related(ctx context.Context, _ *sdk.CallToolRequest, in RelatedInput) (*sdk.CallToolResult, RelatedOutput, error) {
	if strings.TrimSpace(in.Path) == "" {
		return nil, RelatedOutput{Status: "error", Hint: "path is empty"}, nil
	}
	if in.StartLine <= 0 {
		return nil, RelatedOutput{Status: "error", Hint: "start_line must be ≥ 1"}, nil
	}
	p, hint := s.resolveProject(in.ProjectRoot)
	if hint != "" {
		return nil, RelatedOutput{Status: "error", Hint: hint}, nil
	}
	if _, err := os.Stat(p.DBPath); errors.Is(err, os.ErrNotExist) {
		return nil, RelatedOutput{Status: "no-index", Project: p.Root,
			Hint: fmt.Sprintf("no index for %s — run `dex index %s` first.", p.Root, p.Root)}, nil
	}
	k := in.K
	if k <= 0 {
		k = 8
	}
	if k > 30 {
		k = 30
	}
	st, err := store.OpenWith(ctx, p.DBPath, s.StoreOpts)
	if err != nil {
		return nil, RelatedOutput{Status: "error", Hint: fmt.Sprintf("open index: %v", err)}, nil
	}
	defer st.Close()
	hits, err := st.RelatedChunks(ctx, in.Path, in.StartLine, k)
	if err != nil {
		if strings.Contains(err.Error(), "no chunk at") {
			return nil, RelatedOutput{Status: "not-found", Project: p.Root,
				Hint: err.Error() + " — check that path and start_line match an indexed chunk exactly."}, nil
		}
		return nil, RelatedOutput{Status: "error", Hint: fmt.Sprintf("related: %v", err)}, nil
	}
	out := RelatedOutput{Status: "ok", Project: p.Root}
	for _, h := range hits {
		out.Hits = append(out.Hits, SearchHit{
			Path:      h.Path,
			Kind:      h.Kind,
			StartLine: h.StartLine,
			EndLine:   h.EndLine,
			Score:     h.Score,
			Content:   h.Content,
		})
	}
	return nil, out, nil
}

// ─── tool: view_summarize ─────────────────────────────────────────────────

type SummarizeInput struct {
	Path        string  `json:"path" jsonschema:"file path to summarize; relative paths are resolved against project_root"`
	ProjectRoot string  `json:"project_root,omitempty" jsonschema:"absolute path to the project root; defaults to the server's working directory"`
	StartLine   int     `json:"start_line,omitempty" jsonschema:"first line to summarize (1-indexed, inclusive); 0 = beginning of file"`
	EndLine     int     `json:"end_line,omitempty" jsonschema:"last line to summarize (1-indexed, inclusive); 0 = end of file"`
	Focus       string  `json:"focus,omitempty" jsonschema:"optional steering — e.g. 'public API surface', 'side effects', 'error handling'"`
	Temperature float32 `json:"temperature,omitempty" jsonschema:"sampling temperature (0 = server default)"`
	MaxTokens   int     `json:"max_tokens,omitempty" jsonschema:"maximum tokens to generate (0 = server default)"`
}

type SummarizeOutput struct {
	Status       string `json:"status"` // "ok" | "chat-service-unreachable" | "error"
	Hint         string `json:"hint,omitempty"`
	Project      string `json:"project,omitempty"`
	Path         string `json:"path,omitempty"` // resolved path, relative to project root
	StartLine    int    `json:"start_line,omitempty"`
	EndLine      int    `json:"end_line,omitempty"`
	Bytes        int    `json:"bytes,omitempty"`     // how many bytes were sent to the model
	Truncated    bool   `json:"truncated,omitempty"` // true if the slice was cut to fit the cap
	Model        string `json:"model,omitempty"`
	Endpoint     string `json:"endpoint,omitempty"`
	Content      string `json:"content,omitempty"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// maxSummarizeBytes caps the slice we send to the chat endpoint. Above
// this the local model's quality drops sharply and latency spikes;
// callers wanting a whole-repo overview should use ask_codebase with
// RAG instead. Tuned to fit comfortably in a 32B-coder context window
// alongside the system prompt and the summary itself.
const maxSummarizeBytes = 64 * 1024

func (s *Server) summarize(ctx context.Context, _ *sdk.CallToolRequest, in SummarizeInput) (*sdk.CallToolResult, SummarizeOutput, error) {
	out := SummarizeOutput{}
	if s.ChatClient == nil {
		return nil, SummarizeOutput{Status: "error", Hint: "chat client not configured on this server"}, nil
	}
	if strings.TrimSpace(in.Path) == "" {
		return nil, SummarizeOutput{Status: "error", Hint: "path is empty"}, nil
	}
	root := in.ProjectRoot
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			return nil, SummarizeOutput{Status: "error", Hint: "could not determine project root; pass project_root explicitly"}, nil
		}
		root = wd
	}
	p, err := proj.Resolve(root, s.IndexDir)
	if err != nil {
		return nil, SummarizeOutput{Status: "error", Hint: fmt.Sprintf("resolve project: %v", err)}, nil
	}
	out.Project = p.Root
	out.Endpoint = s.ChatClient.Endpoint()
	out.Model = s.ChatClient.ModelName()

	// Resolve path under the project root. Reject anything that
	// escapes it (so an MCP caller can't read /etc/passwd by passing
	// "/etc/passwd" or "../../etc/passwd").
	target := in.Path
	if !filepath.IsAbs(target) {
		target = filepath.Join(p.Root, target)
	}
	realTarget, err := filepath.EvalSymlinks(target)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, SummarizeOutput{Status: "error", Hint: fmt.Sprintf("file does not exist: %s", target)}, nil
		}
		return nil, SummarizeOutput{Status: "error", Hint: fmt.Sprintf("resolve path: %v", err)}, nil
	}
	relTarget, err := filepath.Rel(p.Root, realTarget)
	if err != nil || strings.HasPrefix(relTarget, "..") || relTarget == ".." {
		return nil, SummarizeOutput{Status: "error", Hint: fmt.Sprintf("path %s is outside project root %s", target, p.Root)}, nil
	}
	st, err := os.Stat(realTarget)
	if err != nil {
		return nil, SummarizeOutput{Status: "error", Hint: fmt.Sprintf("stat: %v", err)}, nil
	}
	if st.IsDir() {
		return nil, SummarizeOutput{Status: "error", Hint: fmt.Sprintf("%s is a directory — pass a file path", relTarget)}, nil
	}
	out.Path = relTarget

	data, err := os.ReadFile(realTarget)
	if err != nil {
		return nil, SummarizeOutput{Status: "error", Hint: fmt.Sprintf("read: %v", err)}, nil
	}

	slice, sliceStart, sliceEnd := sliceLines(data, in.StartLine, in.EndLine)
	out.StartLine = sliceStart
	out.EndLine = sliceEnd

	if len(slice) > maxSummarizeBytes {
		slice = slice[:maxSummarizeBytes]
		out.Truncated = true
	}
	out.Bytes = len(slice)

	system := buildSummarizeSystem(in.Focus)
	userContent := fmt.Sprintf("FILE: %s (lines %d-%d)\n\n```\n%s\n```",
		relTarget, sliceStart, sliceEnd, slice)

	resp, err := s.ChatClient.Generate(ctx, []chat.Message{
		{Role: "system", Content: system},
		{Role: "user", Content: userContent},
	}, chat.Options{
		Temperature: in.Temperature,
		MaxTokens:   in.MaxTokens,
	})
	if err != nil {
		if errors.Is(err, chat.ErrUnreachable) {
			out.Status = "chat-service-unreachable"
			out.Hint = "the local chat-completion service is offline."
			return nil, out, nil
		}
		out.Status = "error"
		out.Hint = fmt.Sprintf("chat: %v", err)
		return nil, out, nil
	}

	out.Status = "ok"
	out.Content = resp.Content
	out.FinishReason = resp.FinishReason
	if resp.Model != "" {
		out.Model = resp.Model
	}
	return nil, out, nil
}

// sliceLines returns the byte slice of `data` between lines start and
// end (both 1-indexed, inclusive). Zero values mean "from start of
// file" / "to end of file". Returned start/end are clamped to the
// actual file extents so the caller can echo back what was used.
func sliceLines(data []byte, start, end int) ([]byte, int, int) {
	if start <= 0 && end <= 0 {
		return data, 1, chunk.LineCount(data)
	}
	if start <= 0 {
		start = 1
	}
	// Walk newlines once. Cheap and avoids splitting the whole file.
	var (
		startByte = -1
		endByte   = len(data)
		line      = 1
	)
	if start == 1 {
		startByte = 0
	}
	for i := range data {
		if data[i] != '\n' {
			continue
		}
		line++
		if startByte < 0 && line == start {
			startByte = i + 1
		}
		if end > 0 && line > end {
			endByte = i + 1
			break
		}
	}
	if startByte < 0 {
		// `start` is past EOF — return empty slice but record extents.
		return nil, start, start - 1
	}
	if end <= 0 || end > line {
		end = line
	}
	return data[startByte:endByte], start, end
}

func buildSummarizeSystem(focus string) string {
	base := "You are a file summarizer. Given a single file (or slice), produce a tight, factual summary the reader can use as a substitute for opening the file. " +
		"Lead with one sentence on what the file is for. Then a short bulleted list of the central items the file defines or exposes — picking the framing that fits the file kind: " +
		"exported types/functions for source code, targets and variables for Makefiles, top-level keys for config (YAML/TOML/JSON), section headings for docs, etc. " +
		"Also note key invariants, side effects, or constraints, and any non-obvious dependencies or cross-references. " +
		"Quote identifiers and names verbatim. No prose padding, no apologies, no restating the prompt. " +
		"Keep under 200 words. For trivial files (license, .gitignore, simple stubs) a single sentence is fine."
	if strings.TrimSpace(focus) != "" {
		base += " Focus specifically on: " + strings.TrimSpace(focus) + "."
	}
	return base
}

// ─── tool: index_status ───────────────────────────────────────────────────

type StatusInput struct{}

type ProjectStatus struct {
	ID               string `json:"id"`
	Root             string `json:"root,omitempty"`
	Chunks           int    `json:"chunks"`
	Files            int    `json:"files"`
	Dim              int    `json:"dim"`
	LastIndexed      string `json:"last_indexed,omitempty"`
	PendingSummaries int    `json:"pending_summaries,omitempty"`
	LastSummarized   string `json:"last_summarized,omitempty"`
}

type StatusOutput struct {
	Endpoint          string          `json:"endpoint"`
	Reachable         bool            `json:"reachable"`
	Model             string          `json:"model"`
	ChatEndpoint      string          `json:"chat_endpoint,omitempty"`
	ChatReachable     bool            `json:"chat_reachable,omitempty"`
	ChatModel         string          `json:"chat_model,omitempty"`
	RerankEndpoint    string          `json:"rerank_endpoint,omitempty"`
	RerankReachable   bool            `json:"rerank_reachable,omitempty"`
	RerankModel       string          `json:"rerank_model,omitempty"`
	CompressEndpoint  string          `json:"compress_endpoint,omitempty"`
	CompressReachable bool            `json:"compress_reachable,omitempty"`
	CompressModel     string          `json:"compress_model,omitempty"`
	DraftEndpoint     string          `json:"draft_endpoint,omitempty"`
	DraftReachable    bool            `json:"draft_reachable,omitempty"`
	DraftModel        string          `json:"draft_model,omitempty"`
	Version           string          `json:"version"`
	IndexDir          string          `json:"index_dir"`
	Projects          []ProjectStatus `json:"projects,omitempty"`
	Error             string          `json:"error,omitempty"`
}

// healthChecker abstracts a client that can report reachability.
type healthChecker interface {
	Health(ctx context.Context) error
}

func (s *Server) status(ctx context.Context, _ *sdk.CallToolRequest, _ StatusInput) (*sdk.CallToolResult, StatusOutput, error) {
	out := StatusOutput{
		Endpoint: s.EmbedClient.Endpoint(),
		Model:    s.EmbedClient.ModelName(),
		Version:  Version,
		IndexDir: s.IndexDir,
	}

	// Populate optional endpoint metadata before probing (read-only).
	if s.ChatClient != nil {
		out.ChatEndpoint = s.ChatClient.Endpoint()
		out.ChatModel = s.ChatClient.ModelName()
	}
	if s.RerankClient != nil {
		out.RerankEndpoint = s.RerankClient.Endpoint()
		out.RerankModel = s.RerankClient.ModelName()
	}
	if s.CompressClient != nil {
		out.CompressEndpoint = s.CompressClient.Endpoint()
		out.CompressModel = s.CompressClient.ModelName()
	}
	if s.DraftClient != nil {
		out.DraftEndpoint = s.DraftClient.Endpoint()
		out.DraftModel = s.DraftClient.ModelName()
	}

	// Probe all clients concurrently — each has a 3 s timeout; running
	// them in parallel keeps the total wall-clock cost at ~3 s instead
	// of up to 15 s when clients are unreachable.
	probe := func(wg *sync.WaitGroup, client healthChecker, setResult func(bool, string)) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			pctx, cancel := context.WithTimeout(ctx, 3*time.Second)
			defer cancel()
			if err := client.Health(pctx); err != nil {
				setResult(false, err.Error())
			} else {
				setResult(true, "")
			}
		}()
	}

	var wg sync.WaitGroup
	probe(&wg, s.EmbedClient, func(ok bool, errMsg string) {
		out.Reachable = ok
		out.Error = errMsg
	})
	if s.ChatClient != nil {
		probe(&wg, s.ChatClient, func(ok bool, _ string) { out.ChatReachable = ok })
	}
	if s.RerankClient != nil {
		probe(&wg, s.RerankClient, func(ok bool, _ string) { out.RerankReachable = ok })
	}
	if s.CompressClient != nil {
		probe(&wg, s.CompressClient, func(ok bool, _ string) { out.CompressReachable = ok })
	}
	if s.DraftClient != nil {
		probe(&wg, s.DraftClient, func(ok bool, _ string) { out.DraftReachable = ok })
	}
	wg.Wait()

	if entries, err := os.ReadDir(s.IndexDir); err == nil {
		type result struct {
			ps ProjectStatus
			ok bool
		}
		results := make([]result, len(entries))
		sem := make(chan struct{}, 8)
		var pwg sync.WaitGroup
		for i, e := range entries {
			if !e.IsDir() {
				continue
			}
			dbPath := filepath.Join(s.IndexDir, e.Name(), "index.db")
			if _, err := os.Stat(dbPath); err != nil {
				continue
			}
			pwg.Add(1)
			go func(idx int, id, path string) {
				defer pwg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()
				st, err := store.OpenWith(ctx, path, s.StoreOpts)
				if err != nil {
					return
				}
				stats, _ := st.Stats(ctx)
				root, _ := st.ProjectRoot(ctx)
				st.Close()
				ps := ProjectStatus{
					ID:               id,
					Root:             root,
					Chunks:           stats.Chunks,
					Files:            stats.Files,
					Dim:              stats.Dim,
					PendingSummaries: stats.PendingSummaries,
				}
				if !stats.LastIndex.IsZero() {
					ps.LastIndexed = stats.LastIndex.Format(time.RFC3339)
				}
				if !stats.LastSummarized.IsZero() {
					ps.LastSummarized = stats.LastSummarized.Format(time.RFC3339)
				}
				results[idx] = result{ps: ps, ok: true}
			}(i, e.Name(), dbPath)
		}
		pwg.Wait()
		for _, r := range results {
			if r.ok {
				out.Projects = append(out.Projects, r.ps)
			}
		}
	}
	return nil, out, nil
}

// RunStdio starts the MCP server bound to stdin/stdout. Sets runCtx
// so per-project Watcher goroutines spawned during the session share
// this ctx and exit cleanly when it ends. Blocks until ctx is
// cancelled or the transport closes, then waits for any spawned
// watchers to drain.
func (s *Server) RunStdio(ctx context.Context) error {
	s.runCtx = ctx
	defer s.watcherWG.Wait()

	srv := sdk.NewServer(&sdk.Implementation{
		Name:    "dex",
		Version: Version,
	}, nil)

	sdk.AddTool(srv, &sdk.Tool{
		Name: "search_semantic",
		Description: "Prefer `ask` for general code-understanding questions — it composes this " +
			"tool with symbol lookup and graph expansion. Use search_semantic directly only when you specifically " +
			"want raw semantic ranking without intent routing. " +
			"Embeds the query and returns top-k matching chunks. Supports exclude list to skip paths. " +
			"On error, returns a structured status: 'no-index' (run dex index first), " +
			"'embedding-service-unreachable' (fall back to grep), or 'ok'.",
	}, s.search)

	sdk.AddTool(srv, &sdk.Tool{
		Name: "search_symbol",
		Description: "Prefer `ask` — it detects identifiers in your question and runs this " +
			"lookup automatically as part of a fused response. Use search_symbol directly only when you " +
			"already have the exact identifier name and want nothing else. " +
			"Fast SQL lookup — no embedding required. Returns 'not-found' when no chunk with that name exists.",
	}, s.findSymbol)

	sdk.AddTool(srv, &sdk.Tool{
		Name: "graph_neighbors",
		Description: "Prefer `ask` — it includes neighborhood expansion as part of routing. " +
			"Use graph_neighbors directly only when you already have the exact (path, start_line) of a chunk " +
			"and want its cosine neighbors. " +
			"Finds code that is semantically related even without keyword overlap.",
	}, s.related)

	sdk.AddTool(srv, &sdk.Tool{
		Name: "graph_deps",
		Description: "Return the `imports` edges for a file or package — the package the file belongs to, " +
			"and the list of packages it depends on. Sourced from the static graph (no embedding, no chat). " +
			"Pass `path` (relative file inside the project) OR `package` (full package path). " +
			"Returns 'no-index' / 'no-graph' / 'not-found' when the project, graph, or symbol is missing.",
	}, s.graphDeps)

	sdk.AddTool(srv, &sdk.Tool{
		Name: "graph_callers",
		Description: "Return functions that CALL the given symbol, from the static graph's `calls` edges. " +
			"Go-only for now (Python/JS/Rust callers fall back to ripgrep via `ask`). " +
			"Accepts a bare name (`Foo`), a qualified method (`(*Server).RunStdio`), or a package-qualified " +
			"name (`mcp.NewServer`). Multiple matches are returned with their package paths so the agent can " +
			"disambiguate. Returns 'no-graph' when calls edges haven't been indexed yet.",
	}, s.graphCallers)

	sdk.AddTool(srv, &sdk.Tool{
		Name: "graph_callees",
		Description: "Return functions that the given symbol CALLS, from the static graph's `calls` edges. " +
			"Go-only for now. Same name resolution as graph_callers. " +
			"Returns 'no-graph' when calls edges haven't been indexed yet.",
	}, s.graphCallees)

	sdk.AddTool(srv, &sdk.Tool{
		Name:        "index_status",
		Description: "Report dex endpoint health and the list of indexed projects with their chunk counts and last-indexed times.",
	}, s.status)

	sdk.AddTool(srv, &sdk.Tool{
		Name: "ask",
		Description: "PRIMARY ENTRY POINT for code-understanding questions. Call this BEFORE Grep/Glob/Read fan-out. " +
			"Given a free-text question (and optional intent override), it picks a strategy, composes search_semantic " +
			"+ search_symbol + graph expansion, and returns a compact bundle: `semantic_hits`, `symbols`, `suggested_reads` " +
			"(both lanes carry their CONTENTS inlined by default — no follow-up Read needed in the common case), a prose " +
			"`next_action` directive you can execute verbatim, and an `avoid` line telling you what NOT to do. Each " +
			"SymbolHit carries `signature` (declaration line) and `doc` (leading comment block) so you can see the API " +
			"without reading the body. `annotations` is a per-path map populated by intent: always-on entries include " +
			"sibling `tests` (foo.go ↔ foo_test.go) and `nearest_doc` (closest CLAUDE.md / doc.go / README.md walking " +
			"up); editing_context adds `last_commit` / `last_author` (git blame) and `owners` (CODEOWNERS); architecture " +
			"and editing_context add `build_tags` and `package`. `references` carries the `calls` graph edges for " +
			"callers/callees intents (Go-only; other languages fall back to a ripgrep usage list). Inline content " +
			"shares ONE per-intent byte pool across both lanes: targeted intents budget ~60 lines / 4 KB per range " +
			"and ~20 KB total; exploration intents (architecture, package_topology) widen to ~120 lines / 8 KB per " +
			"range and ~40 KB total. Suggested_reads (~2 targeted / ~5 exploration) are filled first as the curated " +
			"cut; semantic_hits use the remaining budget. A range that appears in both lanes is read once and " +
			"charged once. Oversize ranges arrive with `truncated: true` and the original line range, so the caller " +
			"can Read the rest if needed. Pass `no_inline: true` to omit content payloads when you already have the " +
			"files open. Intent is inferred automatically " +
			"(behavior_search/symbol_lookup/callers/callees/architecture/package_topology/editing_context) — pass `intent` " +
			"only to override. Returns 'no-index' / 'embedding-service-unreachable' for graceful fallback to grep.",
	}, s.contextRouter)

	if s.ChatClient != nil {
		sdk.AddTool(srv, &sdk.Tool{
			Name: "view_summarize",
			Description: "Prefer `ask` first — its `suggested_reads` will name the file worth " +
				"summarizing. Use view_summarize directly only when you already know which file you need digested. " +
				"Sends the file slice directly to the chat model. Pass `focus` to steer (e.g. 'public API surface'). " +
				"Path must resolve inside project_root. Files larger than 64 KB are truncated. " +
				"On error, returns 'chat-service-unreachable' or 'error'.",
		}, s.summarize)
	}

	return srv.Run(ctx, &sdk.StdioTransport{})
}

// Version is set at build time via -ldflags.
var Version = "dev"
