// Output rendering for the CLI subcommands. Splitting these out keeps
// main.go focused on dispatch + env wiring, and makes it obvious which
// pieces are "presentation only" vs "real work".
package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/alehatsman/dex/internal/mcp"
	"github.com/alehatsman/dex/internal/store"
)

// endpointProbe captures one configured backend for `dex index status`
// to report. health is nil for probes that aren't reachable to begin with
// (unset opt-in URL, summary inheriting chat) — those skip the HTTP call
// and use the pre-set status string.
type endpointProbe struct {
	name   string
	url    string
	model  string
	health func(context.Context) error
	status string // ok | UNREACHABLE | not configured | inherits chat
}

// collectEndpoints builds the probe list the status command displays.
// Mirrors the env wiring in main.go: embed/chat always present (they
// have defaults); rerank/compress/draft are opt-in; summary falls back
// to chat when DEX_SUMMARY_URL is unset.
func collectEndpoints() []endpointProbe {
	probes := []endpointProbe{}

	em := newEmbedClient()
	probes = append(probes, endpointProbe{name: "embed", url: em.BaseURL, model: em.Model, health: em.Health})

	cc := newChatClient()
	probes = append(probes, endpointProbe{name: "chat", url: cc.BaseURL, model: cc.Model, health: cc.Health})

	if rc := newRerankClient(); rc != nil {
		probes = append(probes, endpointProbe{name: "rerank", url: rc.Endpoint(), model: rc.ModelName(), health: rc.Health})
	} else {
		probes = append(probes, endpointProbe{name: "rerank", status: "not configured"})
	}

	if c := newCompressClient(); c != nil {
		probes = append(probes, endpointProbe{name: "compress", url: c.BaseURL, model: c.Model, health: c.Health})
	} else {
		probes = append(probes, endpointProbe{name: "compress", status: "not configured"})
	}

	if c := newDraftClient(); c != nil {
		probes = append(probes, endpointProbe{name: "draft", url: c.BaseURL, model: c.Model, health: c.Health})
	} else {
		probes = append(probes, endpointProbe{name: "draft", status: "not configured"})
	}

	// summary inherits chat unless DEX_SUMMARY_URL is set explicitly;
	// we report it as its own row so users can see which leg indexing uses.
	if os.Getenv("DEX_SUMMARY_URL") == "" {
		probes = append(probes, endpointProbe{name: "summary", status: "inherits chat"})
	} else {
		sc := newSummaryClient()
		probes = append(probes, endpointProbe{name: "summary", url: sc.BaseURL, model: sc.Model, health: sc.Health})
	}

	return probes
}

// printEndpoints fans out concurrent health checks for every probe with
// a configured URL, then renders an aligned table under a section
// header.
//
// Column order is (NAME, STATUS, MODEL, URL) — status sits next to
// the name so a quick glance scans down a single column to spot any
// failures, instead of having to skip past two wide columns first.
func printEndpoints(ctx context.Context) {
	probes := collectEndpoints()

	var wg sync.WaitGroup
	for i := range probes {
		if probes[i].health == nil {
			continue
		}
		wg.Add(1)
		go func(p *endpointProbe) {
			defer wg.Done()
			if err := p.health(ctx); err != nil {
				p.status = "UNREACHABLE"
			} else {
				p.status = "ok"
			}
		}(&probes[i])
	}
	wg.Wait()

	// Count reachable for the section heading so users can spot a
	// degraded backend without reading the full table.
	reachable := 0
	for _, p := range probes {
		if p.status == "ok" {
			reachable++
		}
	}

	// Column widths derived from the data PLUS the literal header
	// labels so the heading row aligns under the data even when the
	// widest data cell is narrower than the label.
	headers := struct{ name, status, model, url string }{"NAME", "STATUS", "MODEL", "URL"}
	nameW := len(headers.name)
	statusW := len(headers.status)
	modelW := len(headers.model)
	urlW := len(headers.url)
	for _, p := range probes {
		nameW = max(nameW, len(p.name))
		statusW = max(statusW, len(p.status))
		modelW = max(modelW, len(displayCell(p.model)))
		urlW = max(urlW, len(displayCell(p.url)))
	}

	fmt.Printf("endpoints (%d reachable)\n", reachable)
	fmt.Printf("  %-*s  %-*s  %-*s  %s\n",
		nameW, headers.name,
		statusW, headers.status,
		modelW, headers.model,
		headers.url)
	for _, p := range probes {
		fmt.Printf("  %-*s  %-*s  %-*s  %s\n",
			nameW, p.name,
			statusW, p.status,
			modelW, displayCell(p.model),
			displayCell(p.url))
	}
}

func displayCell(s string) string {
	if s == "" {
		return "—"
	}
	return s
}

// formatProjectAge renders the AGE column for the project list.
// Zero timestamps return "—" so the column is still aligned. Stale
// entries (>24h since last index) carry an explicit " stale" suffix
// instead of relying on the ⚠ symbol — easier to scan, no glyph
// dependency.
func formatProjectAge(t time.Time) string {
	if t.IsZero() {
		return "—"
	}
	rel := relativeTime(t)
	if time.Since(t) > 24*time.Hour {
		return rel + "  stale"
	}
	return rel
}

// formatSummaryStatus condenses the two summary-related Stats fields
// into one line. Examples:
//
//	formatSummaryStatus(0, last)          → "last 4h ago"
//	formatSummaryStatus(2, time.Time{})   → "2 queued"
//	formatSummaryStatus(2, last)          → "2 queued · last 4h ago"
//	formatSummaryStatus(0, time.Time{})   → ""  (caller skips the row)
func formatSummaryStatus(pending int, lastSummarized time.Time) string {
	var parts []string
	if pending > 0 {
		parts = append(parts, fmt.Sprintf("%d queued", pending))
	}
	if !lastSummarized.IsZero() {
		parts = append(parts, "last "+relativeTime(lastSummarized))
	}
	return strings.Join(parts, " · ")
}

// queryJSONHit is the wire shape for `dex search semantic --format=json`.
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

// printSearchHitResult renders the shared status/hint/hits shape used
// by the search-style MCP tools (search_semantic, search_symbol,
// graph_neighbors). Single helper keeps the CLI's text output for all
// three surfaces visually identical.
func printSearchHitResult(status, hint, project string, hits []mcp.SearchHit) {
	if status != "" && status != "ok" {
		fmt.Fprintf(os.Stderr, "status: %s\n", status)
		if hint != "" {
			fmt.Fprintf(os.Stderr, "hint:   %s\n", hint)
		}
		return
	}
	if project != "" {
		fmt.Printf("project: %s\n", project)
	}
	if hint != "" {
		fmt.Printf("hint: %s\n", hint)
	}
	if len(hits) == 0 {
		fmt.Fprintln(os.Stderr, "no results")
		return
	}
	for i, h := range hits {
		loc := fmt.Sprintf("%s:%d-%d", h.Path, h.StartLine, h.EndLine)
		header := fmt.Sprintf("─── #%d %s  (%s)  score=%.4f", i+1, loc, h.Kind, h.Score)
		if h.RerankScore > 0 {
			header += fmt.Sprintf("  rerank=%.4f", h.RerankScore)
		}
		if h.Role != "" {
			header += "  role=" + h.Role
		}
		fmt.Println(header)
		if h.Content != "" {
			fmt.Println(truncate(h.Content, 1500))
		}
		fmt.Println()
	}
}

// printContextText emits a human-readable rendering of a ContextOutput.
// Mirrors the layout cmdQuery uses for hits so the two surfaces feel
// like the same tool. The per-section helpers below take the relevant
// slice/map directly so each one is independently testable.
func printContextText(out mcp.ContextOutput) {
	if out.Status != "ok" {
		printContextError(out)
		return
	}
	printContextHeader(out)
	printSuggestedReads(out.SuggestedReads)
	printSymbols(out.Symbols)
	printReferences(out.References)
	printAnnotations(out.Annotations)
	printSemanticHits(out.SemanticHits)
	printGraph(out.Graph)
	printNextActionAndAvoid(out)
}

func printContextError(out mcp.ContextOutput) {
	fmt.Fprintf(os.Stderr, "status: %s\n", out.Status)
	if out.Hint != "" {
		fmt.Fprintf(os.Stderr, "hint:   %s\n", out.Hint)
	}
	if out.Endpoint != "" {
		fmt.Fprintf(os.Stderr, "endpoint: %s\n", out.Endpoint)
	}
}

func printContextHeader(out mcp.ContextOutput) {
	fmt.Printf("intent: %s  project: %s\n", out.Intent, out.Project)
	if out.Stale {
		fmt.Println("⚠ index is stale — refresh recommended")
	}
	if out.Hint != "" {
		fmt.Printf("hint: %s\n", out.Hint)
	}
	fmt.Println()
}

func printSuggestedReads(reads []mcp.SuggestedRead) {
	if len(reads) == 0 {
		return
	}
	fmt.Println("Suggested reads:")
	for i, r := range reads {
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

func printSymbols(symbols []mcp.SymbolHit) {
	if len(symbols) == 0 {
		return
	}
	fmt.Println("Relevant symbols:")
	for _, sym := range symbols {
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

func printReferences(refs []mcp.RefHit) {
	if len(refs) == 0 {
		return
	}
	fmt.Println("References:")
	for _, r := range refs {
		fmt.Printf("  - %s:%d  %s\n", r.Path, r.Line, r.Snippet)
	}
	fmt.Println()
}

func printAnnotations(anns map[string]mcp.PathMeta) {
	if len(anns) == 0 {
		return
	}
	fmt.Println("Annotations:")
	for path, meta := range anns {
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

func printSemanticHits(hits []mcp.SemHit) {
	if len(hits) == 0 {
		return
	}
	fmt.Println("Semantic hits:")
	for i, h := range hits {
		loc := fmt.Sprintf("%s:%d-%d", h.Path, h.StartLine, h.EndLine)
		fmt.Printf("  %d. %s  (%s)  score=%.4f", i+1, loc, h.Kind, h.Score)
		if h.Reason != "" {
			fmt.Printf("  (%s)", h.Reason)
		}
		fmt.Println()
		// Summary-kind hits carry synthesized prose in Content; the
		// line range points at source that wouldn't match if re-read,
		// so inline the body here.
		if strings.HasSuffix(h.Kind, "_summary") && h.Content != "" {
			for line := range strings.SplitSeq(strings.TrimRight(h.Content, "\n"), "\n") {
				fmt.Printf("     │ %s\n", line)
			}
		}
	}
	fmt.Println()
}

func printGraph(gr *mcp.GraphResult) {
	if gr == nil || (len(gr.Nodes) == 0 && len(gr.Edges) == 0) {
		return
	}
	fmt.Println("Graph:")
	for _, n := range gr.Nodes {
		fmt.Printf("  node  %-12s  %s\n", n.Kind, n.ID)
	}
	for _, e := range gr.Edges {
		fmt.Printf("  edge  %-12s  %s → %s\n", e.Kind, e.From, e.To)
	}
	fmt.Println()
}

func printNextActionAndAvoid(out mcp.ContextOutput) {
	if out.NextAction != "" {
		fmt.Printf("Next action:\n  %s\n\n", out.NextAction)
	}
	if out.Avoid != "" {
		fmt.Printf("Avoid:\n  %s\n", out.Avoid)
	}
}
