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
	"github.com/alehatsman/mcsearch/internal/store"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Server holds everything the MCP handlers need.
type Server struct {
	EmbedClient *embed.Client
	ChatClient  *chat.Client  // optional — when nil, generate_code is not registered
	IndexDir    string        // base dir holding per-project index folders
	StoreOpts   store.Options // applied to every Store opened by the server
}

// ─── tool: semantic_search ────────────────────────────────────────────────

type SearchInput struct {
	Query       string `json:"query" jsonschema:"natural-language or code query"`
	ProjectRoot string `json:"project_root,omitempty" jsonschema:"absolute path to the project root; defaults to the server's working directory"`
	K           int    `json:"k,omitempty" jsonschema:"number of results to return (default 8, max 30)"`
}

type SearchHit struct {
	Path      string  `json:"path"`
	Kind      string  `json:"kind"`
	StartLine int     `json:"start_line"`
	EndLine   int     `json:"end_line"`
	// Score is the cosine similarity in [-1, 1]. Always populated.
	Score float32 `json:"score"`
	// BM25Score is the lexical (FTS5) score when the hit surfaced
	// through the BM25 leg of hybrid search. Larger = better. Zero
	// for semantic-only hits.
	BM25Score float32 `json:"bm25_score,omitempty"`
	// RRFScore is the fused rank used for ordering when hybrid search
	// is active. Zero when search ran semantic-only.
	RRFScore float32 `json:"rrf_score,omitempty"`
	Content  string  `json:"content"`
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
			Path:      h.Path,
			Kind:      h.Kind,
			StartLine: h.StartLine,
			EndLine:   h.EndLine,
			Score:     h.Score,
			BM25Score: h.BM25Score,
			RRFScore:  h.RRFScore,
			Content:   h.Content,
		})
	}
	return nil, out, nil
}

// ─── tool: generate_code ──────────────────────────────────────────────────

type GenerateInput struct {
	Prompt      string `json:"prompt" jsonschema:"natural-language description of the code to generate / change / explain"`
	ProjectRoot string `json:"project_root,omitempty" jsonschema:"absolute path to the project root; defaults to the server's working directory"`
	K           int    `json:"k,omitempty" jsonschema:"number of RAG chunks to retrieve as context (default 8, max 30; ignored when use_index=false)"`
	UseIndex    *bool  `json:"use_index,omitempty" jsonschema:"prepend top-k chunks from the project's mcsearch index as context (default true)"`
	System      string `json:"system,omitempty" jsonschema:"override the system prompt; default is a concise code-assistant prompt"`
	Temperature float32 `json:"temperature,omitempty" jsonschema:"sampling temperature (0 = server default)"`
	MaxTokens   int    `json:"max_tokens,omitempty" jsonschema:"maximum tokens to generate (0 = server default)"`
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

func (s *Server) generate(ctx context.Context, req *sdk.CallToolRequest, in GenerateInput) (*sdk.CallToolResult, GenerateOutput, error) {
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
		system = defaultGenerateSystem
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
	Endpoint      string          `json:"endpoint"`
	Reachable     bool            `json:"reachable"`
	Model         string          `json:"model"`
	ChatEndpoint  string          `json:"chat_endpoint,omitempty"`
	ChatReachable bool            `json:"chat_reachable,omitempty"`
	ChatModel     string          `json:"chat_model,omitempty"`
	Version       string          `json:"version"`
	IndexDir      string          `json:"index_dir"`
	Projects      []ProjectStatus `json:"projects,omitempty"`
	Error         string          `json:"error,omitempty"`
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
	}

	return srv.Run(ctx, &sdk.StdioTransport{})
}

// Version is set at build time via -ldflags.
var Version = "dev"
