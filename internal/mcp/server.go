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
	RerankClient rerank.HealthChecker // optional — only consulted by `status` for health reporting; the actual rerank wiring goes through StoreOpts.Reranker
	// CompressClient, when set, distils retrieved chunks into a dense
	// prose summary before they are injected into the chat model's
	// context window. Reduces the token cost of each ask_codebase /
	// generate_code call — typically 4–5× — at the cost of one extra
	// round-trip to a fast local model. Disabled when nil.
	CompressClient *chat.Client
	// DraftClient, when set, pre-generates a local code draft for
	// generate_code. The main ChatClient then validates and refines the
	// draft rather than writing from scratch, cutting generation tokens
	// by 3–10×. Falls back to standard RAG generation on draft failure.
	// Disabled when nil.
	DraftClient  *chat.Client
	IndexDir     string        // base dir holding per-project index folders
	StoreOpts    store.Options // applied to every Store opened by the server
	// AskChatModel optionally overrides ChatClient.Model for the
	// ask_codebase tool. Empty = use the same model as generate_code.
	// Intended for swapping in an instruction-tuned model for Q&A while
	// keeping a coder-tuned model for code generation — coder models
	// resist citation-only prompts and produce ungrounded "hypothetical"
	// answers, instruct models follow the contract more reliably.
	AskChatModel string
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

func (s *Server) findSymbol(ctx context.Context, req *sdk.CallToolRequest, in FindSymbolInput) (*sdk.CallToolResult, FindSymbolOutput, error) {
	if strings.TrimSpace(in.Name) == "" {
		return nil, FindSymbolOutput{Status: "error", Hint: "name is empty"}, nil
	}
	root := in.ProjectRoot
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			return nil, FindSymbolOutput{Status: "error", Hint: "could not determine project root; pass project_root explicitly"}, nil
		}
		root = wd
	}
	p, err := proj.Resolve(root, s.IndexDir)
	if err != nil {
		return nil, FindSymbolOutput{Status: "error", Hint: fmt.Sprintf("resolve project: %v", err)}, nil
	}
	if _, err := os.Stat(p.DBPath); os.IsNotExist(err) {
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

func (s *Server) related(ctx context.Context, req *sdk.CallToolRequest, in RelatedInput) (*sdk.CallToolResult, RelatedOutput, error) {
	if strings.TrimSpace(in.Path) == "" {
		return nil, RelatedOutput{Status: "error", Hint: "path is empty"}, nil
	}
	if in.StartLine <= 0 {
		return nil, RelatedOutput{Status: "error", Hint: "start_line must be ≥ 1"}, nil
	}
	root := in.ProjectRoot
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			return nil, RelatedOutput{Status: "error", Hint: "could not determine project root; pass project_root explicitly"}, nil
		}
		root = wd
	}
	p, err := proj.Resolve(root, s.IndexDir)
	if err != nil {
		return nil, RelatedOutput{Status: "error", Hint: fmt.Sprintf("resolve project: %v", err)}, nil
	}
	if _, err := os.Stat(p.DBPath); os.IsNotExist(err) {
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
	// Draft is the local model's speculative code draft before the main
	// model refined it. Populated only when DraftClient is wired and
	// the draft call succeeded. Exposed so callers can diff draft vs.
	// final to see what the main model corrected.
	Draft      string `json:"draft,omitempty"`
	DraftModel string `json:"draft_model,omitempty"`
}

const defaultGenerateSystem = "You are a code assistant. Your output MUST be grounded in the CONTEXT chunks provided.\n" +
	"\n" +
	"HARD RULES — any violation makes the output invalid:\n" +
	"1. Every symbol, type, function, method, field, package name, and file path you write MUST appear verbatim in the CONTEXT chunks. If it is not in CONTEXT, do not write it.\n" +
	"2. Do not invent wrapper functions, helper types, interfaces, or packages that are not in CONTEXT.\n" +
	"3. If CONTEXT is insufficient to implement the request, output ONLY: \"INSUFFICIENT CONTEXT: <list what symbols or files are missing>\" — no speculative code.\n" +
	"4. Output code in a single fenced block. No step-by-step guides, no version-control instructions."

// defaultAskSystem steers the model toward answering questions about
// the project rather than emitting code. The retrieval pipeline is
// identical to generate_code; only the framing differs — code chunks
// in, prose out.
//
// Output is forced into a two-section CITATIONS / ANSWER format:
// commit to evidence before opining. This pattern (chain-of-grounding)
// dramatically reduces hallucination across models — even ones that
// otherwise ignore "don't invent" instructions, because they have to
// physically write down which chunks they're using first, and then
// any claim in ANSWER without a [n] tag is visibly wrong.
//
// Code blocks are banned because qwen2.5-coder and similar coder-tuned
// models otherwise wrap invented signatures in real file paths. Reach
// for an instruct-tuned model via MCSEARCH_ASK_MODEL if the coder
// model keeps cheating on the format.
const defaultAskSystem = "You are a repository analyst. Answer ONLY from the CONTEXT chunks provided — not from training data or general knowledge.\n" +
	"\n" +
	"OUTPUT FORMAT — follow this exactly or the response is invalid:\n" +
	"\n" +
	"CITATIONS:\n" +
	"[1] <path>:<start>-<end> — <one clause: what this chunk contains>\n" +
	"[2] <path>:<start>-<end> — <one clause: what this chunk contains>\n" +
	"(one line per chunk you rely on; skip irrelevant ones)\n" +
	"\n" +
	"ANSWER:\n" +
	"<prose answering the question; every concrete claim ends with a [n] tag citing a CITATIONS entry>\n" +
	"\n" +
	"HARD RULES:\n" +
	"1. CITATIONS first, always. Writing ANSWER before CITATIONS = invalid.\n" +
	"2. Every path:line in CITATIONS must be copied verbatim from a chunk header. Never invent a path.\n" +
	"3. Every claim in ANSWER must end with [n]. A claim with no tag is guessing — delete it or move it to \"UNCLEAR:\".\n" +
	"4. No fenced code blocks, no indented code. Inline backticks for symbol names only.\n" +
	"5. If CONTEXT does not contain the answer, end ANSWER with: \"UNCLEAR: <what's missing>.\"\n" +
	"6. Banned words/phrases (each signals guessing — replace with a citation or delete): 'hypothetically', 'for example', 'for instance', 'typically', 'usually', 'likely', 'probably', 'you might', 'one approach', 'something like', 'pseudo-code', 'simplified'."

func (s *Server) generate(ctx context.Context, req *sdk.CallToolRequest, in GenerateInput) (*sdk.CallToolResult, GenerateOutput, error) {
	if s.DraftClient != nil {
		return s.generateWithLocalDraft(ctx, in)
	}
	return s.runRAGChat(ctx, in, defaultGenerateSystem, "", true)
}

func (s *Server) ask(ctx context.Context, req *sdk.CallToolRequest, in AskInput) (*sdk.CallToolResult, GenerateOutput, error) {
	// AskInput and GenerateInput are shape-compatible; the only
	// reason ask_codebase has its own type is to advertise different
	// jsonschema descriptions to MCP clients (Q&A vs. code gen).
	//
	// Compression is disabled for ask_codebase: the citations-first
	// contract requires exact path:line references in each chunk so the
	// model can emit accurate [n] citations. Distilled prose loses that
	// granularity and breaks the CITATIONS / ANSWER template.
	return s.runRAGChat(ctx, GenerateInput(in), defaultAskSystem, s.AskChatModel, false)
}

// retrieveContext runs the shared RAG retrieval step: checks index
// existence, embeds the prompt, and returns the top-k context chunks.
//
// Returns ok=false and a ready-to-return GenerateOutput when retrieval
// cannot proceed (no index, embed unreachable, etc.). When use_index=false
// it returns an empty slice with ok=true immediately.
func (s *Server) retrieveContext(ctx context.Context, in GenerateInput, p *proj.Project) (chunks []store.Hit, errOut GenerateOutput, ok bool) {
	useIndex := true
	if in.UseIndex != nil {
		useIndex = *in.UseIndex
	}
	if !useIndex {
		return nil, GenerateOutput{}, true
	}

	if _, err := os.Stat(p.DBPath); err != nil {
		if os.IsNotExist(err) {
			return nil, GenerateOutput{
				Status: "no-index",
				Hint:   fmt.Sprintf("no index for %s — run `mcsearch index %s` first, or retry with use_index=false to generate without project context.", p.Root, p.Root),
			}, false
		}
		return nil, GenerateOutput{Status: "error", Hint: err.Error()}, false
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
			return nil, GenerateOutput{
				Status:   "embedding-service-unreachable",
				Endpoint: s.EmbedClient.BaseURL,
				Hint:     "the local embedding service is offline — retry later, or pass use_index=false to skip RAG.",
			}, false
		}
		return nil, GenerateOutput{Status: "error", Hint: fmt.Sprintf("embed: %v", err)}, false
	}

	st, err := store.OpenWith(ctx, p.DBPath, s.StoreOpts)
	if err != nil {
		return nil, GenerateOutput{Status: "error", Hint: fmt.Sprintf("open index: %v", err)}, false
	}
	hits, err := st.Search(ctx, vecs[0], in.Prompt, k)
	st.Close()
	if err != nil {
		return nil, GenerateOutput{Status: "error", Hint: fmt.Sprintf("search: %v", err)}, false
	}
	return hits, GenerateOutput{}, true
}

// runRAGChat is the shared semantic-search → chat pipeline behind
// both generate_code and ask_codebase. Caller picks the default system
// prompt; everything else (retrieval, context formatting, error
// translation) is identical.
func (s *Server) runRAGChat(ctx context.Context, in GenerateInput, defaultSystem string, modelOverride string, allowCompress bool) (*sdk.CallToolResult, GenerateOutput, error) {
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

	contextChunks, errOut, ok := s.retrieveContext(ctx, in, p)
	if !ok {
		return nil, errOut, nil
	}

	system := in.System
	if strings.TrimSpace(system) == "" {
		system = defaultSystem
	}

	messages := []chat.Message{{Role: "system", Content: system}}
	userContent := in.Prompt
	if len(contextChunks) > 0 {
		ctxText := store.FormatHits(contextChunks)
		if allowCompress && s.CompressClient != nil {
			if compressed, cerr := compressHits(ctx, s.CompressClient, in.Prompt, contextChunks); cerr == nil && compressed != "" {
				ctxText = fmt.Sprintf("DISTILLED CONTEXT (from %d retrieved chunks):\n\n%s", len(contextChunks), compressed)
			}
		}
		userContent = ctxText + "\n\n---\nCONSTRAINT: Every symbol, type, function, import path, and file name you use in your response MUST appear verbatim in the CONTEXT above. Do not use any knowledge from outside these chunks.\n---\n\n" + in.Prompt
	}
	messages = append(messages, chat.Message{Role: "user", Content: userContent})

	resp, err := s.ChatClient.Generate(ctx, messages, chat.Options{
		Temperature: in.Temperature,
		MaxTokens:   in.MaxTokens,
		Model:       modelOverride,
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

// compressHits calls CompressClient to distil the retrieved chunks into a
// dense, query-focused prose summary. The caller replaces the raw
// store.FormatHits block in the user message with this output, reducing the
// token cost of a generate_code call at the price of one extra round-trip
// to a fast local model.
//
// On any error the caller falls back to the raw store.FormatHits output so
// an offline or slow CompressClient never breaks generation.
func compressHits(ctx context.Context, cc *chat.Client, query string, hits []store.Hit) (string, error) {
	const system = "You are a code context distiller. Given retrieved code chunks and a query, write a concise, dense summary (≤350 words) of what the chunks collectively reveal that is relevant to the query.\n\n" +
		"Structure your response:\n" +
		"1. One direct-answer sentence (or \"The chunks do not answer this directly.\").\n" +
		"2. Bullet list — one bullet per relevant chunk: `path:start-end` — key symbol or behavior.\n" +
		"3. One or two sentences on cross-chunk patterns, call chains, or invariants (if any).\n" +
		"4. UNCLEAR: <what is missing> — only if the query is not fully answered.\n\n" +
		"Rules: quote identifiers verbatim; no code blocks; no prose padding; skip irrelevant chunks entirely."

	var b strings.Builder
	fmt.Fprintf(&b, "QUERY: %s\n\nCHUNKS:\n\n", query)
	for i, h := range hits {
		fmt.Fprintf(&b, "--- chunk %d: %s:%d-%d (%s) ---\n", i+1, h.Path, h.StartLine, h.EndLine, h.Kind)
		b.WriteString(h.Content)
		if !strings.HasSuffix(h.Content, "\n") {
			b.WriteByte('\n')
		}
		b.WriteByte('\n')
	}

	resp, err := cc.Generate(ctx, []chat.Message{
		{Role: "system", Content: system},
		{Role: "user", Content: b.String()},
	}, chat.Options{MaxTokens: 600})
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(resp.Content), nil
}

// defaultDraftSystem steers the local draft model toward producing a
// complete working implementation grounded in the retrieved CONTEXT.
const defaultDraftSystem = "You are a code generator. Write an implementation using ONLY the CONTEXT chunks.\n" +
	"RULES:\n" +
	"1. Every symbol, type, function, import, and file path you write MUST appear verbatim in CONTEXT.\n" +
	"2. Never invent a name, path, or type not shown in CONTEXT. If something is missing, write a TODO comment naming the missing symbol.\n" +
	"3. Output a single fenced code block. No explanation."

// defaultRefineSystem steers the main model to validate a local draft
// rather than generating from scratch — review is 3–10× cheaper in tokens
// than fresh generation.
const defaultRefineSystem = "You are a code reviewer. A local model produced [LOCAL_DRAFT]. Review it against the CONTEXT chunks.\n" +
	"RULES:\n" +
	"1. Delete any symbol, type, function, or file path in [LOCAL_DRAFT] that does NOT appear verbatim in CONTEXT.\n" +
	"2. Fix bugs, wrong signatures, and type mismatches using only what CONTEXT provides.\n" +
	"3. Output the corrected code in a single fenced block.\n" +
	"4. After the block, list any symbols you deleted because they were absent from CONTEXT."

// generateWithLocalDraft runs the two-phase draft → refine pipeline:
//  1. Retrieve context chunks (same path as runRAGChat).
//  2. Optionally compress them via CompressClient.
//  3. Call DraftClient to generate a speculative local draft.
//  4. Call ChatClient to validate and refine the draft.
//
// On draft failure the function falls back to a direct ChatClient call
// (no [LOCAL_DRAFT] block), so a stale or offline DraftClient never
// breaks generate_code.
func (s *Server) generateWithLocalDraft(ctx context.Context, in GenerateInput) (*sdk.CallToolResult, GenerateOutput, error) {
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

	contextChunks, errOut, ok := s.retrieveContext(ctx, in, p)
	if !ok {
		return nil, errOut, nil
	}

	// The draft model always gets the raw code chunks — it needs actual
	// symbol names, types, and file paths to generate working code.
	// Compression (if configured) is applied only for the refine step,
	// where the expensive main model benefits most from token reduction.
	rawCtxText := ""
	if len(contextChunks) > 0 {
		rawCtxText = store.FormatHits(contextChunks)
	}

	// Phase 1: speculative draft from the local model.
	draftUserContent := in.Prompt
	if rawCtxText != "" {
		draftUserContent = rawCtxText + "\n\n---\nCONSTRAINT: Every symbol, type, function, import path, and file name you use MUST appear verbatim in the CONTEXT above. Do not use any knowledge from outside these chunks.\n---\n\n" + in.Prompt
	}
	draftResp, draftErr := s.DraftClient.Generate(ctx, []chat.Message{
		{Role: "system", Content: defaultDraftSystem},
		{Role: "user", Content: draftUserContent},
	}, chat.Options{Temperature: in.Temperature, MaxTokens: in.MaxTokens})
	if draftErr == nil {
		out.Draft = strings.TrimSpace(draftResp.Content)
		out.DraftModel = draftResp.Model
		if out.DraftModel == "" {
			out.DraftModel = s.DraftClient.Model
		}
	}
	// draftErr != nil → out.Draft stays ""; refine step falls back to
	// direct generation using defaultGenerateSystem instead.

	// Phase 2: validation / refinement by the main model. The refine
	// step can use compressed context — this is where token savings matter.
	refineCtxText := rawCtxText
	if rawCtxText != "" && s.CompressClient != nil {
		if compressed, cerr := compressHits(ctx, s.CompressClient, in.Prompt, contextChunks); cerr == nil && compressed != "" {
			refineCtxText = fmt.Sprintf("DISTILLED CONTEXT (from %d retrieved chunks):\n\n%s", len(contextChunks), compressed)
		}
	}

	system := in.System
	if strings.TrimSpace(system) == "" {
		if out.Draft != "" {
			system = defaultRefineSystem
		} else {
			system = defaultGenerateSystem
		}
	}

	var parts []string
	if refineCtxText != "" {
		parts = append(parts, refineCtxText)
		parts = append(parts, "CONSTRAINT: Every symbol, type, function, import path, and file name in your output MUST appear verbatim in the CONTEXT above. Delete anything not found there.")
	}
	if out.Draft != "" {
		parts = append(parts, "[LOCAL_DRAFT]\n"+out.Draft+"\n[/LOCAL_DRAFT]")
	}
	parts = append(parts, in.Prompt)
	refineUserContent := in.Prompt
	if len(parts) > 1 {
		refineUserContent = strings.Join(parts, "\n\n---\n\n")
	}

	resp, err := s.ChatClient.Generate(ctx, []chat.Message{
		{Role: "system", Content: system},
		{Role: "user", Content: refineUserContent},
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
		out.RerankEndpoint = s.RerankClient.Endpoint()
		out.RerankModel = s.RerankClient.ModelName()
		rctx, rcancel := context.WithTimeout(ctx, 3*time.Second)
		if err := s.RerankClient.Health(rctx); err == nil {
			out.RerankReachable = true
		}
		rcancel()
	}
	if s.CompressClient != nil {
		out.CompressEndpoint = s.CompressClient.BaseURL
		out.CompressModel = s.CompressClient.Model
		cmpctx, cmpcancel := context.WithTimeout(ctx, 3*time.Second)
		if err := s.CompressClient.Health(cmpctx); err == nil {
			out.CompressReachable = true
		}
		cmpcancel()
	}
	if s.DraftClient != nil {
		out.DraftEndpoint = s.DraftClient.BaseURL
		out.DraftModel = s.DraftClient.Model
		dctx, dcancel := context.WithTimeout(ctx, 3*time.Second)
		if err := s.DraftClient.Health(dctx); err == nil {
			out.DraftReachable = true
		}
		dcancel()
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
