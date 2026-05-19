// Output rendering for the CLI subcommands. Splitting these out keeps
// main.go focused on dispatch + env wiring, and makes it obvious which
// pieces are "presentation only" vs "real work".
package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/alehatsman/mcsearch/internal/mcp"
	"github.com/alehatsman/mcsearch/internal/store"
)

// queryJSONHit is the wire shape for `mcsearch query --format=json`.
// Mirrors mcp.SearchHit so the two CLI/MCP surfaces stay aligned.
type queryJSONHit struct {
	Path        string  `json:"path"`
	Kind        string  `json:"kind"`
	StartLine   int     `json:"start_line"`
	EndLine     int     `json:"end_line"`
	Score       float32 `json:"score"`
	BM25Score   float32 `json:"bm25_score,omitempty"`
	RRFScore    float32 `json:"rrf_score,omitempty"`
	RerankScore float32 `json:"rerank_score,omitempty"`
	Content     string  `json:"content"`
}

func hitsToJSON(hits []store.Hit) []queryJSONHit {
	out := make([]queryJSONHit, len(hits))
	for i, h := range hits {
		out[i] = queryJSONHit{
			Path:        h.Path,
			Kind:        h.Kind,
			StartLine:   h.StartLine,
			EndLine:     h.EndLine,
			Score:       h.Score,
			BM25Score:   h.BM25Score,
			RRFScore:    h.RRFScore,
			RerankScore: h.RerankScore,
			Content:     h.Content,
		}
	}
	return out
}

// truncate clips s to n bytes, snapping back to a UTF-8 boundary so we
// don't emit a half-rune sequence to the terminal.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	cut := n
	for cut > 0 && (s[cut]&0xC0) == 0x80 {
		cut--
	}
	return s[:cut] + "\n…(truncated)"
}

// relativeTime formats a timestamp as a human-friendly relative string
// ("just now", "5m ago", "2h ago", "3d ago", or a date for old entries).
func relativeTime(t time.Time) string {
	if t.IsZero() {
		return "never"
	}
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	case d < 7*24*time.Hour:
		return fmt.Sprintf("%dd ago", int(d.Hours()/24))
	default:
		return t.Format("2006-01-02")
	}
}

// printContextText emits a human-readable rendering of a ContextOutput.
// Mirrors the layout cmdQuery uses for hits so the two surfaces feel
// like the same tool.
func printContextText(out mcp.ContextOutput) {
	if out.Status != "ok" {
		fmt.Fprintf(os.Stderr, "status: %s\n", out.Status)
		if out.Hint != "" {
			fmt.Fprintf(os.Stderr, "hint:   %s\n", out.Hint)
		}
		if out.Endpoint != "" {
			fmt.Fprintf(os.Stderr, "endpoint: %s\n", out.Endpoint)
		}
		return
	}
	fmt.Printf("intent: %s  project: %s\n", out.Intent, out.Project)
	if out.Stale {
		fmt.Println("⚠ index is stale — refresh recommended")
	}
	if out.Hint != "" {
		fmt.Printf("hint: %s\n", out.Hint)
	}
	fmt.Println()

	if len(out.SuggestedReads) > 0 {
		fmt.Println("Suggested reads:")
		for i, r := range out.SuggestedReads {
			loc := r.Path
			if r.StartLine > 0 || r.EndLine > 0 {
				loc = fmt.Sprintf("%s:%d-%d", r.Path, r.StartLine, r.EndLine)
			}
			fmt.Printf("  %d. %s\n     reason: %s\n", i+1, loc, r.Reason)
			if r.Content != "" {
				for line := range strings.SplitSeq(strings.TrimRight(r.Content, "\n"), "\n") {
					fmt.Printf("     │ %s\n", line)
				}
				if r.Truncated {
					fmt.Println("     │ … (truncated; Read the file for the rest)")
				}
			}
		}
		fmt.Println()
	}

	if len(out.Symbols) > 0 {
		fmt.Println("Relevant symbols:")
		for _, sym := range out.Symbols {
			loc := sym.Path
			if sym.StartLine > 0 {
				loc = fmt.Sprintf("%s:%d", sym.Path, sym.StartLine)
			}
			fmt.Printf("  - %s  (%s)  %s\n", sym.QualifiedName, sym.Kind, loc)
			if sym.Signature != "" {
				fmt.Printf("      sig: %s\n", sym.Signature)
			}
			if sym.Doc != "" {
				for line := range strings.SplitSeq(sym.Doc, "\n") {
					fmt.Printf("      doc: %s\n", line)
				}
			}
		}
		fmt.Println()
	}

	if len(out.References) > 0 {
		fmt.Println("References:")
		for _, r := range out.References {
			fmt.Printf("  - %s:%d  %s\n", r.Path, r.Line, r.Snippet)
		}
		fmt.Println()
	}

	if len(out.Annotations) > 0 {
		fmt.Println("Annotations:")
		for path, meta := range out.Annotations {
			fmt.Printf("  %s\n", path)
			if meta.LastCommit != "" {
				fmt.Printf("    last:    %s  %s\n", meta.LastCommit, meta.LastAuthor)
			}
			if len(meta.Owners) > 0 {
				fmt.Printf("    owners:  %s\n", strings.Join(meta.Owners, " "))
			}
			if meta.NearestDoc != "" {
				fmt.Printf("    doc:     %s\n", meta.NearestDoc)
			}
			if len(meta.Tests) > 0 {
				fmt.Printf("    tests:   %s\n", strings.Join(meta.Tests, " "))
			}
			if meta.BuildTags != "" {
				fmt.Printf("    build:   %s\n", meta.BuildTags)
			}
			if meta.Package != "" {
				fmt.Printf("    package: %s\n", meta.Package)
			}
		}
		fmt.Println()
	}

	if len(out.SemanticHits) > 0 {
		fmt.Println("Semantic hits:")
		for i, h := range out.SemanticHits {
			loc := fmt.Sprintf("%s:%d-%d", h.Path, h.StartLine, h.EndLine)
			fmt.Printf("  %d. %s  score=%.4f", i+1, loc, h.Score)
			if h.Reason != "" {
				fmt.Printf("  (%s)", h.Reason)
			}
			fmt.Println()
		}
		fmt.Println()
	}

	if out.Graph != nil && (len(out.Graph.Nodes) > 0 || len(out.Graph.Edges) > 0) {
		fmt.Println("Graph:")
		for _, n := range out.Graph.Nodes {
			fmt.Printf("  node  %-12s  %s\n", n.Kind, n.ID)
		}
		for _, e := range out.Graph.Edges {
			fmt.Printf("  edge  %-12s  %s → %s\n", e.Kind, e.From, e.To)
		}
		fmt.Println()
	}

	if out.NextAction != "" {
		fmt.Printf("Next action:\n  %s\n\n", out.NextAction)
	}
	if out.Avoid != "" {
		fmt.Printf("Avoid:\n  %s\n", out.Avoid)
	}
}
