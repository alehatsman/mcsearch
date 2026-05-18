// Package mcp wires the mcsearch toolset onto the official MCP Go SDK
// and runs it over stdio.
package mcp

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/alehatsman/mcsearch/internal/chat"
	"github.com/alehatsman/mcsearch/internal/chunk"
	"github.com/alehatsman/mcsearch/internal/embed"
	"github.com/alehatsman/mcsearch/internal/proj"
	"github.com/alehatsman/mcsearch/internal/rerank"
	"github.com/alehatsman/mcsearch/internal/store"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Server holds everything the MCP handlers need.
type Server struct {
	EmbedClient    *embed.Client
	ChatClient     *chat.Client         // optional — when nil, summarize_path is not registered
	RerankClient   rerank.HealthChecker // optional — only consulted by `status` for health reporting; the actual rerank wiring goes through StoreOpts.Reranker
	CompressClient *chat.Client         // optional — health reported by status
	DraftClient    *chat.Client         // optional — health reported by status
	IndexDir       string               // base dir holding per-project index folders
	StoreOpts      store.Options        // applied to every Store opened by the server
}

// ─── tool: semantic_search ────────────────────────────────────────────────

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
	Content     string  `json:"content"`
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
	return p, ""
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
			out.Hint = fmt.Sprintf("no index for %s — run `mcsearch index %s` first, then retry. Fall back to grep / Glob in the meantime.", p.Root, p.Root)
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
		out.Hint = fmt.Sprintf("index is %s old — results may be stale; run `mcsearch index %s` to refresh.",
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

// ─── tool: find_symbol ────────────────────────────────────────────────────

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
			Hint: fmt.Sprintf("no index for %s — run `mcsearch index %s` first.", p.Root, p.Root)}, nil
	}
	st, err := store.OpenWith(ctx, p.DBPath, s.StoreOpts)
	if err != nil {
		return nil, FindSymbolOutput{Status: "error", Hint: fmt.Sprintf("open index: %v", err)}, nil
	}
	defer st.Close()
	hits, err := st.FindSymbol(ctx, in.Name, in.K)
	if err != nil {
		return nil, FindSymbolOutput{Status: "error", Hint: fmt.Sprintf("find_symbol: %v", err)}, nil
	}
	out := FindSymbolOutput{Status: "ok", Project: p.Root}
	if len(hits) == 0 {
		out.Status = "not-found"
		out.Hint = fmt.Sprintf("no chunk with name=%q in the index; check spelling or re-index if recently added.", in.Name)
		return nil, out, nil
	}
	for _, h := range hits {
		out.Hits = append(out.Hits, SearchHit{
			Path:      h.Path,
			Kind:      h.Kind,
			StartLine: h.StartLine,
			EndLine:   h.EndLine,
			Score:     1.0,
			Content:   h.Content,
		})
	}
	return nil, out, nil
}

// ─── tool: related_chunks ─────────────────────────────────────────────────

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
			Hint: fmt.Sprintf("no index for %s — run `mcsearch index %s` first.", p.Root, p.Root)}, nil
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

// ─── tool: summarize_path ─────────────────────────────────────────────────

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

// ─── tool: mcsearch_status ────────────────────────────────────────────────

type StatusInput struct{}

type ProjectStatus struct {
	ID          string `json:"id"`
	Chunks      int    `json:"chunks"`
	Files       int    `json:"files"`
	Dim         int    `json:"dim"`
	LastIndexed string `json:"last_indexed,omitempty"`
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
				st.Close()
				ps := ProjectStatus{
					ID:     id,
					Chunks: stats.Chunks,
					Files:  stats.Files,
					Dim:    stats.Dim,
				}
				if !stats.LastIndex.IsZero() {
					ps.LastIndexed = stats.LastIndex.Format(time.RFC3339)
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

// RunStdio starts the MCP server bound to stdin/stdout.
func (s *Server) RunStdio(ctx context.Context) error {
	srv := sdk.NewServer(&sdk.Implementation{
		Name:    "mcsearch",
		Version: Version,
	}, nil)

	sdk.AddTool(srv, &sdk.Tool{
		Name: "semantic_search",
		Description: "Search a project's code semantically by embedding the query and returning the top-k matching chunks. " +
			"Use this instead of fanning out grep when the user's intent is described in natural language. " +
			"Supports exclude list to skip paths. " +
			"On error, the tool returns a structured status: 'no-index' (run mcsearch index first), " +
			"'embedding-service-unreachable' (fall back to grep), or 'ok'.",
	}, s.search)

	sdk.AddTool(srv, &sdk.Tool{
		Name: "find_symbol",
		Description: "Look up chunks by exact identifier name (function, method, type, class). " +
			"Fast SQL lookup — no embedding required. " +
			"Use when you already know the exact name of a symbol and want to find its definition(s). " +
			"Returns 'not-found' when no chunk with that name exists in the index.",
	}, s.findSymbol)

	sdk.AddTool(srv, &sdk.Tool{
		Name: "related_chunks",
		Description: "Return chunks most similar to a specific chunk at (path, start_line). " +
			"Uses vector cosine similarity — finds code that is semantically related even without keyword overlap. " +
			"Use after semantic_search or find_symbol to explore the neighbourhood of a known chunk.",
	}, s.related)

	sdk.AddTool(srv, &sdk.Tool{
		Name:        "mcsearch_status",
		Description: "Report mcsearch endpoint health and the list of indexed projects with their chunk counts and last-indexed times.",
	}, s.status)

	if s.ChatClient != nil {
		sdk.AddTool(srv, &sdk.Tool{
			Name: "summarize_path",
			Description: "Summarize a single file (or line range) into a tight factual digest the caller can read instead " +
				"of the full file. No retrieval — sends the file slice directly to the chat model. Pass `focus` to steer " +
				"(e.g. 'public API surface', 'side effects'). Path must resolve inside project_root. Files larger than " +
				"64 KB are truncated. " +
				"On error, returns a structured status: 'chat-service-unreachable' or 'error'.",
		}, s.summarize)
	}

	return srv.Run(ctx, &sdk.StdioTransport{})
}

// Version is set at build time via -ldflags.
var Version = "dev"
