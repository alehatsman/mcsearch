package mcp

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"

	"github.com/alehatsman/dex/internal/store"
)

// handleSummaries serves GET /v1/projects/{id}/summaries — every summary chunk
// dex composed for the project, enumerated directly from the store.
func (s *Server) handleSummaries(projects map[string]string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		root := resolveProjectFromURL(w, r, projects)
		if root == "" {
			return
		}
		writeJSON(w, http.StatusOK, s.Summaries(r.Context(), root))
	}
}

// SummaryItem is one enumerated summary: the path it describes, its kind
// (file_summary | package_summary | repo_summary), and the prose.
type SummaryItem struct {
	Path    string `json:"path"`
	Kind    string `json:"kind"`
	Content string `json:"content"`
}

// SummariesOutput is the response for the enumerate-summaries endpoint.
type SummariesOutput struct {
	Status    string        `json:"status"` // "ok" | "no-index" | "error"
	Hint      string        `json:"hint,omitempty"`
	Project   string        `json:"project,omitempty"`
	Summaries []SummaryItem `json:"summaries"`
}

// Summaries enumerates every summary chunk dex composed for a project (repo,
// per-package, per-file) in a single direct query — no embedding or rerank,
// so the result is complete and stable regardless of index size. This is the
// reliable alternative to recalling summaries through semantic search, where
// the top-k cutoff and reranker drop summaries in favor of code chunks.
func (s *Server) Summaries(ctx context.Context, projectRoot string) SummariesOutput {
	p, hint := s.resolveProject(projectRoot)
	if hint != "" {
		return SummariesOutput{Status: "error", Hint: hint}
	}
	if _, err := os.Stat(p.DBPath); errors.Is(err, os.ErrNotExist) {
		return SummariesOutput{Status: "no-index", Project: p.Root,
			Hint: fmt.Sprintf("no index for %s — run `dex index %s` first.", p.Root, p.Root)}
	}
	st, err := store.OpenWith(ctx, p.DBPath, s.StoreOpts)
	if err != nil {
		return SummariesOutput{Status: "error", Hint: fmt.Sprintf("open index: %v", err)}
	}
	defer func() { _ = st.Close() }()

	chunks, err := st.AllSummaryChunks(ctx)
	if err != nil {
		return SummariesOutput{Status: "error", Hint: fmt.Sprintf("query summaries: %v", err)}
	}
	items := make([]SummaryItem, 0, len(chunks))
	for _, c := range chunks {
		items = append(items, SummaryItem{Path: c.Path, Kind: c.Kind, Content: c.Content})
	}
	return SummariesOutput{Status: "ok", Project: p.Root, Summaries: items}
}
