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

	"github.com/alehatsman/mcsearch/internal/embed"
	"github.com/alehatsman/mcsearch/internal/proj"
	"github.com/alehatsman/mcsearch/internal/store"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Server holds everything the MCP handlers need.
type Server struct {
	EmbedClient *embed.Client
	IndexDir    string // base dir holding per-project index folders
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
	Score     float32 `json:"score"`
	Content   string  `json:"content"`
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

	st, err := store.Open(ctx, p.DBPath)
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

	hits, err := st.Search(ctx, vecs[0], k)
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
			Content:   h.Content,
		})
	}
	return nil, out, nil
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
	Endpoint  string          `json:"endpoint"`
	Reachable bool            `json:"reachable"`
	Model     string          `json:"model"`
	Version   string          `json:"version"`
	IndexDir  string          `json:"index_dir"`
	Projects  []ProjectStatus `json:"projects,omitempty"`
	Error     string          `json:"error,omitempty"`
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
			st, err := store.Open(ctx, dbPath)
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

	return srv.Run(ctx, &sdk.StdioTransport{})
}

// Version is set at build time via -ldflags.
var Version = "dev"
