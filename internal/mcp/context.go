// Package mcp — mcsearch_context tool.
//
// context.go wires up `mcsearch_context`, a query planner for code
// understanding. The goal is to be the single entry point an agent
// reaches for instead of fanning out to grep / Read / semantic_search
// loops. Given a project and a free-text question (plus optional
// intent override), the router picks a strategy, runs the right
// combination of legs (semantic_search, find_symbol, related_chunks,
// and — when it lands — graph queries), and returns a compact bundle
// with `suggested_reads`, a prose `next_action`, and an `avoid` line.
//
// Schema, field names, and intent vocabulary follow issue #5.
//
// Graph integration: when internal/graph lands, plug a graphExpander
// into Server (or pass via StoreOpts). Until then `callers`,
// `callees`, and `package_topology` degrade to a semantic + symbol
// fallback with an `avoid` line flagging the missing capability.
package mcp

import (
	"context"
	"errors"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/alehatsman/mcsearch/internal/embed"
	"github.com/alehatsman/mcsearch/internal/store"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// ─── intent vocabulary (issue #5) ─────────────────────────────────────────

const (
	IntentAuto            = "auto"
	IntentBehaviorSearch  = "behavior_search"
	IntentSymbolLookup    = "symbol_lookup"
	IntentCallers         = "callers"
	IntentCallees         = "callees"
	IntentArchitecture    = "architecture"
	IntentPackageTopology = "package_topology"
	IntentEditingContext  = "editing_context"
)

var validIntents = map[string]struct{}{
	IntentAuto: {}, IntentBehaviorSearch: {}, IntentSymbolLookup: {},
	IntentCallers: {}, IntentCallees: {}, IntentArchitecture: {},
	IntentPackageTopology: {}, IntentEditingContext: {},
}

// graphDeferredIntents tags intents whose primary leg requires graph
// edges the layer 1 extractor doesn't yet emit. Layer 1 ships
// contains/imports/has_method/has_field/embeds — enough for
// symbol_lookup, editing_context, architecture, package_topology.
// callers/callees still need `calls` edges (deferred to a follow-up
// layer per internal/graph). The router emits an `avoid` line so the
// agent doesn't trust the symbols list as exhaustive for those.
var graphDeferredIntents = map[string]struct{}{
	IntentCallers: {}, IntentCallees: {},
}

// Identifier detection patterns. Conservative — false positives are
// cheap (we just run find_symbol and get nothing) but false negatives
// mean we miss the structural fast path.
var (
	// (*Type).Method or Type.Method — receiver-qualified Go-style names.
	reQualifiedSymbol = regexp.MustCompile(`\(\*?[A-Z][A-Za-z0-9_]*\)\.[A-Za-z_][A-Za-z0-9_]*|\b[A-Z][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b`)
	// Bare PascalCase identifier of length ≥ 3 (skip "I", "Go", noise).
	reBarePascal = regexp.MustCompile(`\b[A-Z][A-Za-z0-9_]{2,}\b`)
	// snake_case_with_underscores — at least one underscore so we don't
	// flag plain words.
	reSnake = regexp.MustCompile(`\b[a-z][a-z0-9_]*_[a-z0-9_]+\b`)

	// Intent keyword regexes for auto routing.
	reCallers      = regexp.MustCompile(`\b(callers?|who calls|what calls|called by|usage of|usages of|references? to|where is .* used|where is .* called)\b`)
	reCallees      = regexp.MustCompile(`\b(callees?|what does .* call|calls from|outgoing calls|dependencies of)\b`)
	reArchitecture = regexp.MustCompile(`\b(architecture|how does .* work|overview|big picture|design of|walk me through|how is .* organized)\b`)
	rePackages     = regexp.MustCompile(`\b(packages?|modules?|topology|dependency graph|import graph|package layout)\b`)
	reEditing      = regexp.MustCompile(`\b(edit|modify|change|update|refactor|rename|extend|fix|patch|implement|add)\b`)
)

// ─── tool: mcsearch_context ───────────────────────────────────────────────

type ContextInput struct {
	Project  string `json:"project,omitempty" jsonschema:"absolute path to the project root; defaults to the server's working directory"`
	Question string `json:"question" jsonschema:"free-text question about the codebase (e.g. 'where is filesystem event debouncing handled?', 'how does indexing work?', 'callers of (*Store).Search')"`
	Intent   string `json:"intent,omitempty" jsonschema:"force a strategy: auto|behavior_search|symbol_lookup|callers|callees|architecture|package_topology|editing_context (default: auto)"`
	K        int    `json:"k,omitempty" jsonschema:"max hits per lane (default 8, max 30)"`
}

// SemHit is a semantic-search result reduced to the wire shape the
// issue specifies. Full chunk content is not included — `suggested_reads`
// tells the caller which file/range to open in full.
type SemHit struct {
	Path      string  `json:"path"`
	StartLine int     `json:"start_line"`
	EndLine   int     `json:"end_line"`
	Score     float32 `json:"score"`
	Reason    string  `json:"reason,omitempty"`
}

type SymbolHit struct {
	QualifiedName string `json:"qualified_name"`
	Path          string `json:"path,omitempty"`
	StartLine     int    `json:"start_line,omitempty"`
	EndLine       int    `json:"end_line,omitempty"`
	Kind          string `json:"kind,omitempty"`
}

// GraphResult is the placeholder for the deferred graph layer. Always
// emitted (even when empty) so the caller can rely on the field
// existing — when graph lands the wire shape grows but the field
// presence doesn't change.
type GraphResult struct {
	Nodes []GraphNode `json:"nodes"`
	Edges []GraphEdge `json:"edges"`
}

type GraphNode struct {
	ID            string `json:"id"`
	QualifiedName string `json:"qualified_name,omitempty"`
	Kind          string `json:"kind,omitempty"`
}

type GraphEdge struct {
	From string `json:"from"`
	To   string `json:"to"`
	Kind string `json:"kind,omitempty"`
}

type SuggestedRead struct {
	Path      string `json:"path"`
	StartLine int    `json:"start_line,omitempty"`
	EndLine   int    `json:"end_line,omitempty"`
	Reason    string `json:"reason"`
}

type ContextOutput struct {
	Status        string          `json:"status"` // ok | no-index | embedding-service-unreachable | error
	Hint          string          `json:"hint,omitempty"`
	Endpoint      string          `json:"endpoint,omitempty"` // populated when embed is unreachable
	Project       string          `json:"project,omitempty"`
	Intent        string          `json:"intent,omitempty"`
	Stale         bool            `json:"stale,omitempty"`
	SemanticHits  []SemHit        `json:"semantic_hits,omitempty"`
	Symbols       []SymbolHit     `json:"symbols,omitempty"`
	Graph         *GraphResult    `json:"graph,omitempty"`
	SuggestedReads []SuggestedRead `json:"suggested_reads,omitempty"`
	NextAction    string          `json:"next_action,omitempty"`
	Avoid         string          `json:"avoid,omitempty"`
}

// ContextRouter is the exported entry point used by the CLI
// (`mcsearch context`). It delegates to the MCP-registered handler.
func (s *Server) ContextRouter(ctx context.Context, in ContextInput) (*sdk.CallToolResult, ContextOutput, error) {
	return s.contextRouter(ctx, nil, in)
}

func (s *Server) contextRouter(ctx context.Context, _ *sdk.CallToolRequest, in ContextInput) (*sdk.CallToolResult, ContextOutput, error) {
	if strings.TrimSpace(in.Question) == "" {
		return nil, ContextOutput{Status: "error", Hint: "question is empty — pass a natural-language question about the codebase"}, nil
	}
	p, hint := s.resolveProject(in.Project)
	if hint != "" {
		return nil, ContextOutput{Status: "error", Hint: hint}, nil
	}

	intent, candidates := resolveIntent(in)
	out := ContextOutput{Project: p.Root, Intent: intent}

	if _, err := os.Stat(p.DBPath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			out.Status = "no-index"
			out.Hint = fmt.Sprintf("no index for %s — run `mcsearch index %s` first; fall back to grep until then.", p.Root, p.Root)
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
	k = min(k, 30)

	st, err := store.OpenWith(ctx, p.DBPath, s.StoreOpts)
	if err != nil {
		out.Status = "error"
		out.Hint = fmt.Sprintf("open index: %v", err)
		return nil, out, nil
	}
	defer st.Close()

	if stats, statsErr := st.Stats(ctx); statsErr == nil && !stats.LastIndex.IsZero() && time.Since(stats.LastIndex) > 24*time.Hour {
		out.Stale = true
		out.Hint = fmt.Sprintf("index is %s old — run `mcsearch index %s` to refresh.",
			time.Since(stats.LastIndex).Round(time.Hour), p.Root)
	}

	// Always emit the graph field (issue requires structural presence).
	// enrichGraph may replace it with a populated view per intent.
	out.Graph = &GraphResult{Nodes: []GraphNode{}, Edges: []GraphEdge{}}

	// Load the graph view once per request. Nil view = no graph
	// indexed; intents that need it will note this in `avoid`.
	graphView, _ := loadGraphView(ctx, st)

	// Symbol lane — exact identifier lookups. Cheap, no embed required.
	// Runs whenever the question contains identifier-shaped tokens, even
	// for non-symbol intents (a behavior_search question that mentions
	// `(*Store).Search` benefits from the structural lane too).
	symbols, symbolPaths := s.runSymbolLane(ctx, st, candidates, k)
	out.Symbols = symbols

	// Semantic lane — runs unless embed is offline. We always run it
	// for recall even when the symbol lane has exact hits.
	semHits, embedFailed := s.runSemanticLane(ctx, st, in.Question, k)
	if embedFailed {
		out.Endpoint = s.EmbedClient.Endpoint()
	}
	out.SemanticHits = semHits

	if len(out.Symbols) == 0 && len(out.SemanticHits) == 0 {
		if embedFailed {
			out.Status = "embedding-service-unreachable"
			out.Hint = "the local embedding service is offline — fall back to grep / Glob / ripgrep for this query."
			return nil, out, nil
		}
		out.Status = "ok"
		out.Hint = "no matches; try broader phrasing or a more specific identifier."
		out.NextAction = "Try rephrasing the question with concrete keywords from the codebase, or fall back to grep."
		return nil, out, nil
	}

	enrichGraph(&out, intent, graphView, out.SemanticHits, out.Symbols)
	out.SuggestedReads = pickSuggestedReads(intent, out.SemanticHits, out.Symbols, symbolPaths)
	out.NextAction = buildNextAction(intent, out.SuggestedReads, out.Symbols)
	out.Avoid = buildAvoid(intent, out.SemanticHits, out.Symbols, graphView != nil)
	out.Status = "ok"
	if embedFailed && out.Hint == "" {
		out.Hint = "embed offline; results from symbol lane only."
	}
	return nil, out, nil
}

// ─── intent classification ────────────────────────────────────────────────

// intentCandidates carries side data the lanes consume: identifiers
// detected in the question that should feed find_symbol.
type intentCandidates struct {
	identifiers []string // ranked best-first (qualified before bare)
}

// resolveIntent picks an intent and surfaces side data (detected
// identifiers). Priority:
//
//  1. Explicit Intent field (issue spec) when valid and not "auto".
//  2. Keyword regex on Question.
//  3. Identifier-shaped tokens → symbol_lookup.
//  4. Default: behavior_search.
func resolveIntent(in ContextInput) (string, intentCandidates) {
	cand := intentCandidates{identifiers: extractIdentifiers(in.Question)}

	explicit := strings.ToLower(strings.TrimSpace(in.Intent))
	if explicit != "" && explicit != IntentAuto {
		if _, ok := validIntents[explicit]; ok {
			return explicit, cand
		}
		// Invalid override falls through to auto routing.
	}

	q := strings.ToLower(in.Question)
	switch {
	case reCallers.MatchString(q):
		return IntentCallers, cand
	case reCallees.MatchString(q):
		return IntentCallees, cand
	case rePackages.MatchString(q):
		return IntentPackageTopology, cand
	case reArchitecture.MatchString(q):
		return IntentArchitecture, cand
	case reEditing.MatchString(q):
		return IntentEditingContext, cand
	}

	if len(cand.identifiers) > 0 && looksLikeBareIdentifierQuery(in.Question) {
		return IntentSymbolLookup, cand
	}
	return IntentBehaviorSearch, cand
}

// looksLikeBareIdentifierQuery returns true when the question is short
// enough and identifier-dominated that the user likely wants a symbol
// lookup rather than a behavior search. Heuristic, but keeps
// "(*Store).Search" from being routed to behavior_search.
func looksLikeBareIdentifierQuery(q string) bool {
	q = strings.TrimSpace(q)
	if q == "" {
		return false
	}
	words := strings.Fields(q)
	// 1-3 words AND at least one identifier-shaped token.
	return len(words) <= 3
}

func extractIdentifiers(q string) []string {
	seen := map[string]struct{}{}
	var out []string
	add := func(s string) {
		if _, ok := seen[s]; ok {
			return
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}

	// Pass 1: qualified symbols. Track their byte spans so we can skip
	// bare-Pascal matches that fall inside (e.g. "Store" and "Search"
	// inside "(*Store).Search" are noise once the qualified form is
	// recorded).
	type span struct{ lo, hi int }
	var taken []span
	for _, idx := range reQualifiedSymbol.FindAllStringIndex(q, -1) {
		add(q[idx[0]:idx[1]])
		taken = append(taken, span{idx[0], idx[1]})
	}
	inside := func(lo, hi int) bool {
		for _, sp := range taken {
			if lo >= sp.lo && hi <= sp.hi {
				return true
			}
		}
		return false
	}

	for _, idx := range reBarePascal.FindAllStringIndex(q, -1) {
		if inside(idx[0], idx[1]) {
			continue
		}
		add(q[idx[0]:idx[1]])
	}
	for _, idx := range reSnake.FindAllStringIndex(q, -1) {
		if inside(idx[0], idx[1]) {
			continue
		}
		add(q[idx[0]:idx[1]])
	}
	return out
}

// ─── lanes ────────────────────────────────────────────────────────────────

// runSymbolLane runs find_symbol for each detected identifier and
// returns deduplicated symbol hits plus a set of file paths the lane
// touched (used by pickSuggestedReads). At most `k` hits returned.
func (s *Server) runSymbolLane(ctx context.Context, st *store.Store, cand intentCandidates, k int) ([]SymbolHit, map[string]struct{}) {
	if len(cand.identifiers) == 0 {
		return nil, nil
	}
	paths := map[string]struct{}{}
	seen := map[string]struct{}{}
	var out []SymbolHit
	for _, id := range cand.identifiers {
		// find_symbol expects the bare name; strip a "(*T)." prefix.
		bare := id
		if i := strings.LastIndex(bare, "."); i >= 0 {
			bare = bare[i+1:]
		}
		hits, err := st.FindSymbol(ctx, bare, k)
		if err != nil {
			continue
		}
		for _, h := range hits {
			key := h.Path + ":" + h.Name
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			qual := h.Name
			if h.Name == "" {
				qual = bare
			}
			out = append(out, SymbolHit{
				QualifiedName: qual,
				Path:          h.Path,
				StartLine:     h.StartLine,
				EndLine:       h.EndLine,
				Kind:          h.Kind,
			})
			paths[h.Path] = struct{}{}
			if len(out) >= k {
				return out, paths
			}
		}
	}
	return out, paths
}

// runSemanticLane embeds the question and runs Search. Returns
// (hits, embedUnreachable). When embedUnreachable is true hits is nil
// and the caller should surface the failure.
func (s *Server) runSemanticLane(ctx context.Context, st *store.Store, question string, k int) ([]SemHit, bool) {
	em := s.EmbedClient
	vecs, err := em.Embed(ctx, []string{question})
	if err != nil {
		if errors.Is(err, embed.ErrUnreachable) {
			return nil, true
		}
		return nil, false
	}
	hits, err := st.Search(ctx, vecs[0], question, k)
	if err != nil {
		return nil, false
	}
	out := make([]SemHit, 0, len(hits))
	for _, h := range hits {
		out = append(out, SemHit{
			Path:      h.Path,
			StartLine: h.StartLine,
			EndLine:   h.EndLine,
			Score:     h.Score,
			Reason:    h.Name,
		})
	}
	return out, false
}

// ─── suggested_reads ──────────────────────────────────────────────────────

// pickSuggestedReads merges the top results from both lanes into a
// short, deduplicated list of file ranges the caller should open in
// full. Strategy by intent:
//
//   - symbol_lookup, callers, callees: prefer symbol-lane definition
//     sites; one read per definition.
//   - architecture, package_topology: top 2-3 semantic hits across
//     distinct files, widened to surrounding chunk extents.
//   - behavior_search, editing_context: top 2 semantic hits, prefer
//     paths that also appear in the symbol lane (cross-lane agreement
//     bumps confidence).
func pickSuggestedReads(intent string, semHits []SemHit, symbols []SymbolHit, symbolPaths map[string]struct{}) []SuggestedRead {
	maxReads := 2
	switch intent {
	case IntentArchitecture, IntentPackageTopology:
		maxReads = 3
	case IntentSymbolLookup, IntentCallers, IntentCallees:
		maxReads = 2
	}

	seen := map[string]bool{}
	out := make([]SuggestedRead, 0, maxReads)

	// Pass 1: symbol definitions for symbol-driven intents.
	if intent == IntentSymbolLookup || intent == IntentCallers || intent == IntentCallees {
		for _, sym := range symbols {
			if sym.Path == "" || seen[sym.Path] {
				continue
			}
			seen[sym.Path] = true
			out = append(out, SuggestedRead{
				Path:      sym.Path,
				StartLine: sym.StartLine,
				EndLine:   sym.EndLine,
				Reason:    "definition of " + sym.QualifiedName,
			})
			if len(out) >= maxReads {
				return out
			}
		}
	}

	// Pass 2: semantic hits, biased toward cross-lane agreement.
	type ranked struct {
		hit       SemHit
		crossLane bool
	}
	rs := make([]ranked, 0, len(semHits))
	for _, h := range semHits {
		_, cross := symbolPaths[h.Path]
		rs = append(rs, ranked{hit: h, crossLane: cross})
	}
	sort.SliceStable(rs, func(i, j int) bool {
		if rs[i].crossLane != rs[j].crossLane {
			return rs[i].crossLane // cross-lane agreement first
		}
		return rs[i].hit.Score > rs[j].hit.Score
	})
	for _, r := range rs {
		if seen[r.hit.Path] {
			continue
		}
		seen[r.hit.Path] = true
		reason := "top semantic match"
		if r.crossLane {
			reason = "semantic match + symbol agreement"
		}
		out = append(out, SuggestedRead{
			Path:      r.hit.Path,
			StartLine: r.hit.StartLine,
			EndLine:   r.hit.EndLine,
			Reason:    reason,
		})
		if len(out) >= maxReads {
			return out
		}
	}
	return out
}

// ─── next_action / avoid (prose) ──────────────────────────────────────────

// buildNextAction returns an imperative sentence the agent can execute
// directly. The issue is explicit that prose outperforms structured
// args for agent compliance. Always concrete — names paths and line
// ranges — never "do more research."
func buildNextAction(intent string, reads []SuggestedRead, symbols []SymbolHit) string {
	if len(reads) == 0 && len(symbols) == 0 {
		return "Rephrase the question with concrete keywords or fall back to grep."
	}
	switch intent {
	case IntentSymbolLookup:
		if len(reads) > 0 {
			return fmt.Sprintf("Read %s lines %d-%d to see the definition.", reads[0].Path, reads[0].StartLine, reads[0].EndLine)
		}
	case IntentCallers, IntentCallees:
		if len(symbols) > 0 {
			rel := "callers"
			if intent == IntentCallees {
				rel = "callees"
			}
			return fmt.Sprintf("Call-graph edges are not yet extracted — start from %s (%s) and grep for %s.",
				symbols[0].Path, symbols[0].QualifiedName, rel)
		}
	case IntentArchitecture, IntentPackageTopology:
		if len(reads) > 0 {
			parts := make([]string, 0, len(reads))
			for _, r := range reads {
				parts = append(parts, fmt.Sprintf("%s lines %d-%d", r.Path, r.StartLine, r.EndLine))
			}
			return "Skim " + strings.Join(parts, "; ") + " for the structural overview before editing."
		}
	case IntentEditingContext:
		if len(reads) > 0 {
			return fmt.Sprintf("Read %s lines %d-%d before editing — this is the primary site.", reads[0].Path, reads[0].StartLine, reads[0].EndLine)
		}
	}
	// behavior_search and fallback.
	if len(reads) > 0 {
		return fmt.Sprintf("Read %s lines %d-%d to ground your answer.", reads[0].Path, reads[0].StartLine, reads[0].EndLine)
	}
	if len(symbols) > 0 {
		return fmt.Sprintf("Inspect %s in %s.", symbols[0].QualifiedName, symbols[0].Path)
	}
	return ""
}

// buildAvoid emits a "what not to do" hint. Strong claims when we
// have strong signals (exact symbol found → don't grep); softer
// otherwise. `graphIndexed` is true when the project has a graph
// available — `callers`/`callees` still warn because layer 1 doesn't
// emit `calls` edges, regardless.
func buildAvoid(intent string, semHits []SemHit, symbols []SymbolHit, graphIndexed bool) string {
	if _, deferred := graphDeferredIntents[intent]; deferred {
		return "Do not trust the symbols list as exhaustive — `calls` edges are not yet extracted, so caller/callee coverage is best-effort. Verify with grep on the symbol name."
	}
	if !graphIndexed && graphSupportsIntent(intent) {
		return "Graph not indexed for this project — results from semantic + symbol lanes only. Run `mcsearch graph index <project>` for richer structural context."
	}
	if len(symbols) > 0 && len(semHits) > 0 {
		return "Do not grep for the identifier; it is already located. Read the suggested ranges instead of opening whole files."
	}
	if len(symbols) > 0 {
		return "Do not grep for the identifier; it is already located."
	}
	if len(semHits) > 0 {
		return "Do not read entire files; the suggested ranges cover the relevant context."
	}
	return ""
}

