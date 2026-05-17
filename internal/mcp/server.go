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
	"time"

	"github.com/alehatsman/mcsearch/internal/chat"
	"github.com/alehatsman/mcsearch/internal/embed"
	"github.com/alehatsman/mcsearch/internal/proj"
	"github.com/alehatsman/mcsearch/internal/rerank"
	"github.com/alehatsman/mcsearch/internal/store"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Server holds everything the MCP handlers need.
type Server struct {
	EmbedClient  *embed.Client
	ChatClient   *chat.Client   // optional — when nil, generate_code is not registered
	RerankClient *rerank.Client // optional — only consulted by `status` for health reporting; the actual rerank wiring goes through StoreOpts.Reranker
	IndexDir     string         // base dir holding per-project index folders
	StoreOpts    store.Options  // applied to every Store opened by the server
}

// ─── tool: semantic_search ────────────────────────────────────────────────

type SearchInput struct {
	Query       string `json:"query" jsonschema:"natural-language or code query"`
	ProjectRoot string `json:"project_root,omitempty" jsonschema:"absolute path to the project root; defaults to the server's working directory"`
	K           int    `json:"k,omitempty" jsonschema:"number of results to return (default 8, max 30)"`
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

func (s *Server) search(ctx context.Context, req *sdk.CallToolRequest, in SearchInput) (*sdk.CallToolResult, SearchOutput, error) {
	out := SearchOutput{}
	if strings.TrimSpace(in.Query) == "" {
		return nil, SearchOutput{Status: "error", Hint: "query is empty — pass a natural-language description or code fragment"}, nil
	}
	root := in.ProjectRoot
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			return nil, SearchOutput{Status: "error", Hint: "could not determine project root; pass project_root explicitly"}, nil
		}
		root = wd
	}
	p, err := proj.Resolve(root, s.IndexDir)
	if err != nil {
		return nil, SearchOutput{Status: "error", Hint: fmt.Sprintf("resolve project: %v", err)}, nil
	}
	out.Project = p.Root

	if _, err := os.Stat(p.DBPath); err != nil {
		if os.IsNotExist(err) {
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
			out.Endpoint = em.BaseURL
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

// ─── tool: generate_code ──────────────────────────────────────────────────

type GenerateInput struct {
	Prompt      string  `json:"prompt" jsonschema:"natural-language description of the code to generate / change / explain"`
	ProjectRoot string  `json:"project_root,omitempty" jsonschema:"absolute path to the project root; defaults to the server's working directory"`
	K           int     `json:"k,omitempty" jsonschema:"number of RAG chunks to retrieve as context (default 8, max 30; ignored when use_index=false)"`
	UseIndex    *bool   `json:"use_index,omitempty" jsonschema:"prepend top-k chunks from the project's mcsearch index as context (default true)"`
	System      string  `json:"system,omitempty" jsonschema:"override the system prompt; default is a concise code-assistant prompt"`
	Temperature float32 `json:"temperature,omitempty" jsonschema:"sampling temperature (0 = server default)"`
	MaxTokens   int     `json:"max_tokens,omitempty" jsonschema:"maximum tokens to generate (0 = server default)"`
}

type GenerateContextChunk struct {
	Path      string  `json:"path"`
	Kind      string  `json:"kind"`
	StartLine int     `json:"start_line"`
	EndLine   int     `json:"end_line"`
	Score     float32 `json:"score"`
}

type GenerateOutput struct {
	Status       string                 `json:"status"` // "ok" | "no-index" | "embedding-service-unreachable" | "chat-service-unreachable" | "error"
	Hint         string                 `json:"hint,omitempty"`
	Endpoint     string                 `json:"endpoint,omitempty"`
	Project      string                 `json:"project,omitempty"`
	Model        string                 `json:"model,omitempty"`
	Content      string                 `json:"content,omitempty"`
	FinishReason string                 `json:"finish_reason,omitempty"`
	Context      []GenerateContextChunk `json:"context,omitempty"` // chunks fed to the model
}

const defaultGenerateSystem = "You are a precise coding assistant. " +
	"When CONTEXT chunks from the user's project are provided, ground your answer in them — " +
	"reference real symbols, paths, and conventions rather than inventing names. " +
	"Output code in fenced blocks; keep prose minimal."

// defaultAskSystem steers the model toward answering questions about
// the project rather than emitting code. The retrieval pipeline is
// identical to generate_code; only the framing differs — code chunks
// in, prose out.
const defaultAskSystem = "You are a precise repository analyst answering questions about the user's codebase. " +
	"Ground every claim in the CONTEXT chunks: cite file paths and symbol names verbatim, " +
	"never invent identifiers or guess at code you weren't shown. " +
	"If the CONTEXT is insufficient, say so plainly and name what's missing. " +
	"When multiple chunks bear on the question — for example, several gates in a filter chain, " +
	"several layers in a pipeline, several call sites of a function — enumerate ALL of them. " +
	"Do not stop at the first plausible answer when the CONTEXT shows more. " +
	"Prefer short, structured answers (bullets or numbered steps) over long prose. " +
	"Quote code only when the literal text matters; otherwise describe it."

func (s *Server) generate(ctx context.Context, req *sdk.CallToolRequest, in GenerateInput) (*sdk.CallToolResult, GenerateOutput, error) {
	return s.runRAGChat(ctx, in, defaultGenerateSystem)
}

func (s *Server) ask(ctx context.Context, req *sdk.CallToolRequest, in AskInput) (*sdk.CallToolResult, GenerateOutput, error) {
	// AskInput and GenerateInput are shape-compatible; the only
	// reason ask_codebase has its own type is to advertise different
	// jsonschema descriptions to MCP clients (Q&A vs. code gen).
	return s.runRAGChat(ctx, GenerateInput(in), defaultAskSystem)
}

// runRAGChat is the shared semantic-search → chat pipeline behind
// both generate_code and ask_codebase. Caller picks the default system
// prompt; everything else (retrieval, context formatting, error
// translation) is identical.
func (s *Server) runRAGChat(ctx context.Context, in GenerateInput, defaultSystem string) (*sdk.CallToolResult, GenerateOutput, error) {
	out := GenerateOutput{}
	if s.ChatClient == nil {
		return nil, GenerateOutput{Status: "error", Hint: "chat client not configured on this server"}, nil
	}
	if strings.TrimSpace(in.Prompt) == "" {
		return nil, GenerateOutput{Status: "error", Hint: "prompt is empty"}, nil
	}
	root := in.ProjectRoot
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			return nil, GenerateOutput{Status: "error", Hint: "could not determine project root; pass project_root explicitly"}, nil
		}
		root = wd
	}
	p, err := proj.Resolve(root, s.IndexDir)
	if err != nil {
		return nil, GenerateOutput{Status: "error", Hint: fmt.Sprintf("resolve project: %v", err)}, nil
	}
	out.Project = p.Root
	out.Endpoint = s.ChatClient.BaseURL
	out.Model = s.ChatClient.Model

	// RAG is on by default; caller can opt out with use_index=false.
	useIndex := true
	if in.UseIndex != nil {
		useIndex = *in.UseIndex
	}

	var contextChunks []store.Hit
	if useIndex {
		if _, err := os.Stat(p.DBPath); err != nil {
			if os.IsNotExist(err) {
				out.Status = "no-index"
				out.Hint = fmt.Sprintf("no index for %s — run `mcsearch index %s` first, or retry with use_index=false to generate without project context.", p.Root, p.Root)
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

		vecs, err := s.EmbedClient.Embed(ctx, []string{in.Prompt})
		if err != nil {
			if errors.Is(err, embed.ErrUnreachable) {
				out.Status = "embedding-service-unreachable"
				out.Endpoint = s.EmbedClient.BaseURL
				out.Hint = "the local embedding service is offline — retry later, or pass use_index=false to skip RAG."
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
		contextChunks, err = st.Search(ctx, vecs[0], in.Prompt, k)
		st.Close()
		if err != nil {
			out.Status = "error"
			out.Hint = fmt.Sprintf("search: %v", err)
			return nil, out, nil
		}
	}

	system := in.System
	if strings.TrimSpace(system) == "" {
		system = defaultSystem
	}

	messages := []chat.Message{{Role: "system", Content: system}}
	userContent := in.Prompt
	if len(contextChunks) > 0 {
		userContent = formatContext(contextChunks) + "\n\n---\n\n" + in.Prompt
	}
	messages = append(messages, chat.Message{Role: "user", Content: userContent})

	resp, err := s.ChatClient.Generate(ctx, messages, chat.Options{
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
	for _, h := range contextChunks {
		out.Context = append(out.Context, GenerateContextChunk{
			Path:      h.Path,
			Kind:      h.Kind,
			StartLine: h.StartLine,
			EndLine:   h.EndLine,
			Score:     h.Score,
		})
	}
	return nil, out, nil
}

// formatContext renders retrieved chunks as a fenced CONTEXT block. Each
// chunk gets a path:line header so the model can cite locations back to
// the caller without us inventing a schema the model has to follow.
func formatContext(hits []store.Hit) string {
	var b strings.Builder
	b.WriteString("CONTEXT — relevant chunks from the project's mcsearch index:\n\n")
	for i, h := range hits {
		fmt.Fprintf(&b, "--- chunk %d: %s:%d-%d (%s, score=%.4f) ---\n",
			i+1, h.Path, h.StartLine, h.EndLine, h.Kind, h.Score)
		b.WriteString(h.Content)
		if !strings.HasSuffix(h.Content, "\n") {
			b.WriteByte('\n')
		}
		b.WriteByte('\n')
	}
	return strings.TrimRight(b.String(), "\n")
}

// ─── tool: ask_codebase ───────────────────────────────────────────────────

// AskInput is shape-compatible with GenerateInput — same retrieval
// pipeline backs both — but exposes a Q&A-tuned jsonschema so MCP
// clients pick the right tool for "explain how X works" vs. "write me
// code that does X".
type AskInput struct {
	Prompt      string  `json:"prompt" jsonschema:"a question about the project (how does X work? where is Y handled? what calls Z?)"`
	ProjectRoot string  `json:"project_root,omitempty" jsonschema:"absolute path to the project root; defaults to the server's working directory"`
	K           int     `json:"k,omitempty" jsonschema:"number of RAG chunks to retrieve as context (default 8, max 30; ignored when use_index=false)"`
	UseIndex    *bool   `json:"use_index,omitempty" jsonschema:"retrieve top-k chunks from the mcsearch index and feed them as context (default true)"`
	System      string  `json:"system,omitempty" jsonschema:"override the system prompt; default is a Q&A-tuned repo-analyst prompt"`
	Temperature float32 `json:"temperature,omitempty" jsonschema:"sampling temperature (0 = server default)"`
	MaxTokens   int     `json:"max_tokens,omitempty" jsonschema:"maximum tokens to generate (0 = server default)"`
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

func (s *Server) summarize(ctx context.Context, req *sdk.CallToolRequest, in SummarizeInput) (*sdk.CallToolResult, SummarizeOutput, error) {
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
	out.Endpoint = s.ChatClient.BaseURL
	out.Model = s.ChatClient.Model

	// Resolve path under the project root. Reject anything that
	// escapes it (so an MCP caller can't read /etc/passwd by passing
	// "/etc/passwd" or "../../etc/passwd").
	target := in.Path
	if !filepath.IsAbs(target) {
		target = filepath.Join(p.Root, target)
	}
	realTarget, err := filepath.EvalSymlinks(target)
	if err != nil {
		if os.IsNotExist(err) {
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
		return data, 1, countLines(data)
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

func countLines(data []byte) int {
	if len(data) == 0 {
		return 0
	}
	n := 1
	for _, b := range data {
		if b == '\n' {
			n++
		}
	}
	// A trailing newline doesn't add a "next" line for the user.
	if data[len(data)-1] == '\n' {
		n--
	}
	return n
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
	Endpoint        string          `json:"endpoint"`
	Reachable       bool            `json:"reachable"`
	Model           string          `json:"model"`
	ChatEndpoint    string          `json:"chat_endpoint,omitempty"`
	ChatReachable   bool            `json:"chat_reachable,omitempty"`
	ChatModel       string          `json:"chat_model,omitempty"`
	RerankEndpoint  string          `json:"rerank_endpoint,omitempty"`
	RerankReachable bool            `json:"rerank_reachable,omitempty"`
	RerankModel     string          `json:"rerank_model,omitempty"`
	Version         string          `json:"version"`
	IndexDir        string          `json:"index_dir"`
	Projects        []ProjectStatus `json:"projects,omitempty"`
	Error           string          `json:"error,omitempty"`
}

func (s *Server) status(ctx context.Context, req *sdk.CallToolRequest, _ StatusInput) (*sdk.CallToolResult, StatusOutput, error) {
	out := StatusOutput{
		Endpoint: s.EmbedClient.BaseURL,
		Model:    s.EmbedClient.Model,
		Version:  Version,
		IndexDir: s.IndexDir,
	}
	hctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	if err := s.EmbedClient.Health(hctx); err != nil {
		out.Reachable = false
		out.Error = err.Error()
	} else {
		out.Reachable = true
	}
	if s.ChatClient != nil {
		out.ChatEndpoint = s.ChatClient.BaseURL
		out.ChatModel = s.ChatClient.Model
		cctx, ccancel := context.WithTimeout(ctx, 3*time.Second)
		if err := s.ChatClient.Health(cctx); err == nil {
			out.ChatReachable = true
		}
		ccancel()
	}
	if s.RerankClient != nil {
		out.RerankEndpoint = s.RerankClient.BaseURL
		out.RerankModel = s.RerankClient.Model
		rctx, rcancel := context.WithTimeout(ctx, 3*time.Second)
		if err := s.RerankClient.Health(rctx); err == nil {
			out.RerankReachable = true
		}
		rcancel()
	}

	entries, err := os.ReadDir(s.IndexDir)
	if err == nil {
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			dbPath := filepath.Join(s.IndexDir, e.Name(), "index.db")
			if _, err := os.Stat(dbPath); err != nil {
				continue
			}
			st, err := store.OpenWith(ctx, dbPath, s.StoreOpts)
			if err != nil {
				continue
			}
			stats, _ := st.Stats(ctx)
			st.Close()
			ps := ProjectStatus{
				ID:     e.Name(),
				Chunks: stats.Chunks,
				Files:  stats.Files,
				Dim:    stats.Dim,
			}
			if !stats.LastIndex.IsZero() {
				ps.LastIndexed = stats.LastIndex.Format(time.RFC3339)
			}
			out.Projects = append(out.Projects, ps)
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
			"On error, the tool returns a structured status: 'no-index' (run mcsearch index first), " +
			"'embedding-service-unreachable' (fall back to grep), or 'ok'.",
	}, s.search)

	sdk.AddTool(srv, &sdk.Tool{
		Name:        "mcsearch_status",
		Description: "Report mcsearch endpoint health and the list of indexed projects with their chunk counts and last-indexed times.",
	}, s.status)

	if s.ChatClient != nil {
		sdk.AddTool(srv, &sdk.Tool{
			Name: "generate_code",
			Description: "Generate code (or edit/explain it) grounded in the project's mcsearch index. " +
				"By default, retrieves the top-k chunks semantically relevant to the prompt and feeds them as CONTEXT " +
				"to a chat-completion model — so the output references real symbols and paths from the project. " +
				"Pass use_index=false to skip retrieval. Returns the generated content plus the chunks fed as context. " +
				"On error, returns a structured status: 'no-index', 'embedding-service-unreachable', " +
				"'chat-service-unreachable', or 'ok'.",
		}, s.generate)

		sdk.AddTool(srv, &sdk.Tool{
			Name: "ask_codebase",
			Description: "Answer a natural-language question about the project, grounded in the mcsearch index. " +
				"Same retrieval pipeline as generate_code but tuned for Q&A: cites paths and symbols, returns prose " +
				"rather than code blocks. Use this instead of fanning out Read calls when the user wants to understand " +
				"how something works ('how does indexing handle deletions?', 'where is config loaded?'). " +
				"On error, returns the same structured statuses as generate_code.",
		}, s.ask)

		sdk.AddTool(srv, &sdk.Tool{
			Name: "summarize_path",
			Description: "Summarize a single file (or line range) into a tight factual digest the caller can read instead " +
				"of the full file. No retrieval — sends the file slice directly to the chat model. Pass `focus` to steer " +
				"(e.g. 'public API surface', 'side effects'). Path must resolve inside project_root. Files larger than " +
				"64 KB are truncated; for bigger overviews use ask_codebase. " +
				"On error, returns a structured status: 'chat-service-unreachable' or 'error'.",
		}, s.summarize)
	}

	return srv.Run(ctx, &sdk.StdioTransport{})
}

// Version is set at build time via -ldflags.
var Version = "dev"
