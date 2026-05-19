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
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/alehatsman/mcsearch/internal/chunk"
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
	// camelCase — lowercase start with an internal uppercase transition
	// (e.g. `inlineContent`, `markDirty`). Required for Go unexported
	// identifiers; the uppercase transition keeps plain English words
	// out (no English word has a mid-word capital).
	reCamel = regexp.MustCompile(`\b[a-z][a-z0-9]*[A-Z][A-Za-z0-9_]*\b`)
	// snake_case_with_underscores — at least one underscore so we don't
	// flag plain words.
	reSnake = regexp.MustCompile(`\b[a-z][a-z0-9_]*_[a-z0-9_]+\b`)

	// Intent keyword regexes for auto routing.
	reCallers      = regexp.MustCompile(`\b(callers?|who calls|what calls|called by|usage of|usages of|references? to|where is .* used|where is .* called)\b`)
	reCallees      = regexp.MustCompile(`\b(callees?|what does .* call|calls from|outgoing calls|dependencies of)\b`)
	reArchitecture = regexp.MustCompile(`\b(architecture|how does .* work|overview|big picture|design of|walk me through|how is .* organized)\b`)
	rePackages     = regexp.MustCompile(`\b(packages?|modules?|topology|dependency graph|import graph|package layout)\b`)
	// `change` / `update` deliberately omitted — they fire on questions
	// like "when X changes" or "update the timestamp on Y" that are
	// really behavior_search, not editing_context.
	reEditing = regexp.MustCompile(`\b(edit|modify|refactor|rename|extend|fix|patch|implement|add)\b`)
)

// ─── tool: mcsearch_context ───────────────────────────────────────────────

type ContextInput struct {
	Project  string `json:"project,omitempty" jsonschema:"absolute path to the project root; defaults to the server's working directory"`
	Question string `json:"question" jsonschema:"free-text question about the codebase (e.g. 'where is filesystem event debouncing handled?', 'how does indexing work?', 'callers of (*Store).Search')"`
	Intent   string `json:"intent,omitempty" jsonschema:"force a strategy: auto|behavior_search|symbol_lookup|callers|callees|architecture|package_topology|editing_context (default: auto)"`
	K        int    `json:"k,omitempty" jsonschema:"max hits per lane (default 8, max 30)"`
	NoInline bool   `json:"no_inline,omitempty" jsonschema:"skip inlining file contents into suggested_reads and semantic_hits. Default off: both lanes carry their line-range content from one shared per-intent byte pool (per-range cap ~60 lines / 4 KB; total cap ~12 KB targeted / ~40 KB exploration; oversize ranges are clipped with truncated=true). Set true if you already have the files open."`
}

// SemHit is a semantic-search result reduced to the wire shape the
// issue specifies. Content is inlined by default so the caller doesn't
// have to issue a follow-up Read for hits below the suggested_reads
// cut; the same per-intent budget pool covers both lanes (see
// inlineCapsFor / inlineContent). Empty when no_inline=true, when the
// file cannot be opened, or when the shared byte budget was exhausted
// before this hit.
type SemHit struct {
	Path      string  `json:"path"`
	StartLine int     `json:"start_line"`
	EndLine   int     `json:"end_line"`
	Score     float32 `json:"score"`
	Kind      string  `json:"kind,omitempty"`
	Reason    string  `json:"reason,omitempty"`
	Content   string  `json:"content,omitempty"`
	Truncated bool    `json:"truncated,omitempty"`
}

type SymbolHit struct {
	QualifiedName string `json:"qualified_name"`
	Path          string `json:"path,omitempty"`
	StartLine     int    `json:"start_line,omitempty"`
	EndLine       int    `json:"end_line,omitempty"`
	Kind          string `json:"kind,omitempty"`
	// Signature is the declaration line (e.g. `func (s *Store) Search(q
	// string) ([]Hit, error)`). Cheap: one file line at StartLine. Lets
	// the caller see the API contract without reading the body.
	Signature string `json:"signature,omitempty"`
	// Doc is the contiguous comment block immediately above StartLine
	// (Go `//` lines, Python `#` lines). Capped at ~10 lines / 600 B.
	Doc string `json:"doc,omitempty"`
}

// RefHit is a lexical reference produced by the references lane
// (callers/callees intents). Stand-in for the deferred `calls` graph
// edges — ripgrep over the bare symbol name, capped to a few dozen
// hits. The definition line is filtered out so the list is genuinely
// "uses of" rather than "appearances of".
type RefHit struct {
	Path    string `json:"path"`
	Line    int    `json:"line"`
	Snippet string `json:"snippet,omitempty"` // single-line excerpt
	Symbol  string `json:"symbol,omitempty"`  // which symbol this is a ref to
}

// PathMeta is the per-file annotation bundle keyed by relative path in
// ContextOutput.Annotations. Fields are populated conditionally based
// on intent and may individually be empty. Designed so all data about
// a single file lives in one place — the caller joins by path.
type PathMeta struct {
	// LastCommit / LastAuthor are populated for editing_context. Short
	// SHA + short date + author; e.g. "5a79083 2026-05-19 Aleh Atsman".
	LastCommit string `json:"last_commit,omitempty"`
	LastAuthor string `json:"last_author,omitempty"`
	// Owners from the project's CODEOWNERS file, matched by glob.
	// Populated for editing_context only.
	Owners []string `json:"owners,omitempty"`
	// NearestDoc is the closest documentation file walking up from the
	// path's directory — CLAUDE.md > doc.go > README.md, stopping at
	// projectRoot. Always-on (cheap dir walk).
	NearestDoc string `json:"nearest_doc,omitempty"`
	// Tests are sibling test files paired by language convention
	// (foo.go ↔ foo_test.go; foo.py ↔ test_foo.py; foo.ts ↔
	// foo.test.ts). Always-on (pure path heuristic).
	Tests []string `json:"tests,omitempty"`
	// BuildTags is the //go:build or // +build constraint line plus the
	// package clause for Go files; populated for editing_context,
	// architecture, and package_topology.
	BuildTags string `json:"build_tags,omitempty"`
	// Package is the `package x` clause for Go files; populated
	// alongside BuildTags.
	Package string `json:"package,omitempty"`
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
	// Content is the file slice for [StartLine, EndLine], inlined by
	// default so the caller doesn't need a follow-up Read for the
	// common case. Capped per-read and totaled across reads — see
	// inlineSuggestedReads. Empty when no_inline=true, when the file
	// cannot be opened, or when the caller hit the total byte budget.
	Content string `json:"content,omitempty"`
	// Truncated is set when the per-read line/byte cap clipped the
	// content before reaching EndLine. The caller can still issue a
	// regular Read for the rest if needed.
	Truncated bool `json:"truncated,omitempty"`
}

type ContextOutput struct {
	Status         string          `json:"status"` // ok | no-index | embedding-service-unreachable | error
	Hint           string          `json:"hint,omitempty"`
	Endpoint       string          `json:"endpoint,omitempty"` // populated when embed is unreachable
	Project        string          `json:"project,omitempty"`
	Intent         string          `json:"intent,omitempty"`
	Stale          bool            `json:"stale,omitempty"`
	SemanticHits   []SemHit        `json:"semantic_hits,omitempty"`
	Symbols        []SymbolHit     `json:"symbols,omitempty"`
	Graph          *GraphResult    `json:"graph,omitempty"`
	SuggestedReads []SuggestedRead `json:"suggested_reads,omitempty"`
	NextAction     string          `json:"next_action,omitempty"`
	Avoid          string          `json:"avoid,omitempty"`
	// References is the ripgrep-backed reference list. Populated for
	// callers/callees intents when at least one SymbolHit is present.
	// Stand-in for the deferred `calls` graph edges.
	References []RefHit `json:"references,omitempty"`
	// Annotations is per-file metadata keyed by the same relative path
	// used in SuggestedReads / Symbols / SemanticHits. Which sub-fields
	// are populated depends on intent (see enrich.go for the gating
	// matrix). Callers join by path.
	Annotations map[string]PathMeta `json:"annotations,omitempty"`
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
	defer func() { _ = st.Close() }()

	if stats, statsErr := st.Stats(ctx); statsErr == nil && !stats.LastIndex.IsZero() && time.Since(stats.LastIndex) > 24*time.Hour {
		out.Stale = true
		out.Hint = fmt.Sprintf("index is %s old — run `mcsearch index %s` to refresh.",
			time.Since(stats.LastIndex).Round(time.Hour), p.Root)
	}

	// enrichGraph sets out.Graph only when it has something to emit.
	// An absent `graph` key signals "no graph indexed, or this intent
	// surfaced no structural context" — saves bytes over shipping
	// `{nodes:[], edges:[]}` on every response.

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
	if intent == IntentArchitecture || intent == IntentPackageTopology {
		summaryHits := s.runSummaryLane(ctx, st, in.Question, k)
		semHits = mergeSummaryHits(summaryHits, semHits, k)
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

	// Near-miss surface for symbol_lookup whiffs: when the user
	// clearly asked for an identifier (intent is symbol_lookup AND we
	// extracted identifier candidates) but the symbols lane found
	// nothing, scan the chunks table for substring matches and surface
	// them in the hint. Mirrors find_symbol's behavior so the agent
	// gets candidate names without a follow-up tool call.
	if intent == IntentSymbolLookup && len(out.Symbols) == 0 && len(candidates.identifiers) > 0 {
		var cands []string
		for _, id := range candidates.identifiers {
			bare := id
			if i := strings.LastIndex(bare, "."); i >= 0 {
				bare = bare[i+1:]
			}
			names, err := st.FindSymbolCandidates(ctx, bare, 5)
			if err != nil {
				continue
			}
			cands = append(cands, names...)
			if len(cands) >= 5 {
				cands = cands[:5]
				break
			}
		}
		if len(cands) > 0 {
			out.Hint = "no exact symbol match — did you mean: " + strings.Join(cands, ", ") + "?"
		}
	}

	enrichGraph(&out, intent, graphView, out.SemanticHits, out.Symbols)
	out.SuggestedReads = pickSuggestedReads(intent, out.SemanticHits, out.Symbols, symbolPaths)
	if !in.NoInline {
		inlineContent(p.Root, intent, out.SuggestedReads, out.SemanticHits)
	}
	enrich(ctx, p.Root, intent, k, &out)
	topSem := maxSemanticScore(out.SemanticHits)
	var graphEdgeCount int
	if out.Graph != nil {
		graphEdgeCount = len(out.Graph.Edges)
	}
	out.NextAction = buildNextAction(intent, out.SuggestedReads, out.Symbols, topSem,
		graphEdgeCount, hasBlameAnnotations(out.Annotations))
	// If the directive's primary read was truncated at inline time,
	// flag that so the agent knows the inlined Content isn't the full
	// chunk and can Read the original line range for the rest.
	if !in.NoInline && len(out.SuggestedReads) > 0 && out.SuggestedReads[0].Truncated {
		out.NextAction += " The inlined content is truncated at inline-budget caps — Read the full line range if you need the tail."
	}
	out.Avoid = buildAvoid(intent, out.SemanticHits, out.Symbols, graphView != nil, len(out.References) > 0)
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
	for _, idx := range reCamel.FindAllStringIndex(q, -1) {
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

	// Fallback for single-word lowercase queries (e.g. `rerank`,
	// `index`, `embed`). None of the regexes above pick these up —
	// they require camelCase, PascalCase, or underscore shape — but
	// they're a perfectly valid form for Go's unexported identifiers
	// and short package names. When the question is literally one
	// short token and we have nothing yet, treat the token as the
	// identifier to look up. Guarded by length and content so a single
	// English word like "fix" or "bug" doesn't dominate.
	if len(out) == 0 {
		trimmed := strings.TrimSpace(q)
		if len(trimmed) >= 3 && len(trimmed) <= 32 && isAllIdentChars(trimmed) {
			out = append(out, trimmed)
		}
	}
	return out
}

// isAllIdentChars reports whether every byte in s is a valid Go
// identifier character (letter, digit, or underscore). Used by the
// single-token fallback in extractIdentifiers to avoid passing
// punctuation/whitespace to find_symbol.
func isAllIdentChars(s string) bool {
	for i := 0; i < len(s); i++ {
		if !isIdentChar(s[i]) {
			return false
		}
	}
	return true
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

// runSummaryLane runs a summary-only semantic search (file_summary +
// package_summary chunks). Used by architecture/package_topology intents to
// surface prose overviews that may not win the general top-k race against
// higher-scoring code chunks.
func (s *Server) runSummaryLane(ctx context.Context, st *store.Store, question string, k int) []SemHit {
	vecs, err := s.EmbedClient.Embed(ctx, []string{question})
	if err != nil {
		return nil
	}
	hits, err := st.SearchSummaries(ctx, vecs[0], question, k)
	if err != nil || len(hits) == 0 {
		return nil
	}
	out := make([]SemHit, 0, len(hits))
	for _, h := range hits {
		// Summary chunks store synthesized prose in Content. Use it
		// directly — the line range points at the underlying source
		// file (or a directory for package_summary) and would yield
		// raw source or nothing if re-read from disk.
		out = append(out, SemHit{
			Path:      h.Path,
			StartLine: h.StartLine,
			EndLine:   h.EndLine,
			Score:     h.Score,
			Kind:      h.Kind,
			Reason:    h.Name,
			Content:   h.Content,
		})
	}
	return out
}

// mergeSummaryHits prepends summary hits before code hits, filling up to k
// total slots. Summaries lead so agents see the prose overview first.
func mergeSummaryHits(summaries, code []SemHit, k int) []SemHit {
	out := make([]SemHit, 0, k)
	out = append(out, summaries...)
	for _, h := range code {
		if len(out) >= k {
			break
		}
		out = append(out, h)
	}
	return out
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
		// In hybrid mode, Hit.Score is raw cosine — zero for hits
		// that came in via BM25 only (the FTS leg of the RRF fusion).
		// Surfacing 0 here misleads the agent into thinking it's
		// looking at irrelevant content. Fall back to the RRF
		// score so every returned hit has a positive ranking signal.
		// Scales differ (cosine ~0-1, RRF ~0-0.03) but ordering
		// within the list is what matters.
		score := h.Score
		if score == 0 && h.RRFScore > 0 {
			score = h.RRFScore
		}
		// Summary-kind rows hold synthesized prose in Content; surface it
		// directly so the inliner doesn't re-read the underlying file
		// and clobber the summary with raw source.
		var content string
		if chunk.IsSummaryKind(h.Kind) {
			content = h.Content
		}
		out = append(out, SemHit{
			Path:      h.Path,
			StartLine: h.StartLine,
			EndLine:   h.EndLine,
			Kind:      h.Kind,
			Score:     score,
			Reason:    h.Name,
			Content:   content,
		})
	}
	return out, false
}

// isDocPath returns true for plain-text documentation files. Used by
// pickSuggestedReads to keep code wins over a near-tied README hit on
// non-architecture intents — the rerank stage *should* sort this out
// but in practice docs sometimes outscore the code they describe.
func isDocPath(p string) bool {
	switch {
	case strings.HasSuffix(p, ".md"),
		strings.HasSuffix(p, ".rst"),
		strings.HasSuffix(p, ".txt"),
		strings.HasSuffix(p, ".adoc"),
		strings.HasSuffix(p, ".mdx"):
		return true
	}
	return false
}

// isBuildOrConfigPath returns true for build/CI/config files that
// rarely contain the implementation a caller is asking about for
// editing_context or behavior_search intents. Same demotion mechanic
// as isDocPath: when the rerank stage lets a Taskfile.yml outscore the
// .go file it's wrapping, the tiebreaker should pick the code. Kept
// narrow on purpose — go.mod / package.json are sometimes the right
// answer ("bump version"), so they stay out.
func isBuildOrConfigPath(p string) bool {
	base := filepath.Base(p)
	switch {
	case strings.HasSuffix(p, ".yml"),
		strings.HasSuffix(p, ".yaml"),
		strings.HasSuffix(p, ".toml"):
		return true
	}
	switch base {
	case "Dockerfile", "Makefile", "Taskfile.yml", "Taskfile.yaml":
		return true
	}
	return false
}

// isTestPath returns true for test files across the languages we
// index. Demoted in pickSuggestedReads Pass 2 so a bare-noun
// symbol_lookup query (e.g. "Executor") doesn't surface
// `executor_test.go` above the type definition. Sibling-test
// annotations still link the matching test from each suggested
// implementation read — demotion only affects ranking, not
// availability.
func isTestPath(p string) bool {
	base := filepath.Base(p)
	switch {
	case strings.HasSuffix(base, "_test.go"),
		strings.HasSuffix(base, ".test.ts"),
		strings.HasSuffix(base, ".test.tsx"),
		strings.HasSuffix(base, ".test.js"),
		strings.HasSuffix(base, ".test.jsx"),
		strings.HasSuffix(base, ".spec.ts"),
		strings.HasSuffix(base, ".spec.tsx"),
		strings.HasSuffix(base, ".spec.js"),
		strings.HasSuffix(base, ".spec.jsx"),
		strings.HasSuffix(base, "_test.py"),
		strings.HasPrefix(base, "test_") && strings.HasSuffix(base, ".py"),
		strings.HasSuffix(base, "_spec.rb"),
		strings.HasSuffix(base, "_test.rs"):
		return true
	}
	return false
}

// isNonImplPath unifies the doc + build/config + test demotion checks.
func isNonImplPath(p string) bool {
	return isDocPath(p) || isBuildOrConfigPath(p) || isTestPath(p)
}

// ─── suggested_reads ──────────────────────────────────────────────────────

// maxSemanticScore returns the highest Score across all semantic
// hits. semantic_hits isn't strictly score-sorted (summary merging
// and rerank-driven re-ordering permute it), so using [0] for the
// "weak match" decision mis-classifies strong responses whenever a
// low-score symbol-driven entry gets promoted to the front.
func maxSemanticScore(hits []SemHit) float32 {
	var top float32
	for _, h := range hits {
		if h.Score > top {
			top = h.Score
		}
	}
	return top
}

// isReadableRange reports whether a SemHit points at a concrete file
// slice the agent can actually `Read`. Rollup chunks (package_summary,
// repo_summary) have Path set to a directory; they're useful context
// in semantic_hits but should not land in suggested_reads where
// "lines 0-0" reads as a Read directive the agent can't execute.
func isReadableRange(h SemHit) bool {
	switch h.Kind {
	case "package_summary", "repo_summary":
		return false
	}
	return true
}

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
		// Exploration intents — the caller is forming a mental model,
		// so a denser bundle (more files, more lines, see
		// inlineCapsFor) pays off more than a slim one.
		maxReads = 5
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
	// For code-oriented intents we also demote non-implementation paths
	// (docs and build/CI config) as a tiebreaker, so a README or
	// Taskfile.yml doesn't beat the .go file that implements the
	// feature when scores are close. Architecture is the exception —
	// the README often IS the right read, and build files can reveal
	// structure.
	preferCode := intent != IntentArchitecture
	type ranked struct {
		hit       SemHit
		crossLane bool
		nonImpl   bool
	}
	rs := make([]ranked, 0, len(semHits))
	for _, h := range semHits {
		// Skip rollup hits (package_summary / repo_summary) — their
		// "path" is a directory and StartLine/EndLine are 0, so they
		// produce bogus "lines 0-0" directives downstream. They still
		// live in semantic_hits as informational context; they just
		// don't belong in suggested_reads.
		if !isReadableRange(h) {
			continue
		}
		_, cross := symbolPaths[h.Path]
		rs = append(rs, ranked{hit: h, crossLane: cross, nonImpl: isNonImplPath(h.Path)})
	}
	sort.SliceStable(rs, func(i, j int) bool {
		if rs[i].crossLane != rs[j].crossLane {
			return rs[i].crossLane // cross-lane agreement first
		}
		if preferCode && rs[i].nonImpl != rs[j].nonImpl {
			return !rs[i].nonImpl // implementation beats doc/build
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
			Content:   r.hit.Content,
		})
		if len(out) >= maxReads {
			return out
		}
	}
	return out
}

// ─── next_action / avoid (prose) ──────────────────────────────────────────

// noiseFloorScore is the per-hit cutoff applied to semantic_hits
// inlining when the top score is already below lowConfidenceScore.
// On a genuine no-signal query (gibberish, very rare phrase) the whole
// pool tends to cluster in the 0.35-0.40 band; inlining all of them
// burns the byte budget on hits the agent will rightly ignore. The
// path+range still ships, just without Content, so the caller can
// follow up with a manual Read if a low-score path turns out to be
// relevant after all.
const noiseFloorScore = 0.40

// lowConfidenceScore is the cosine-fused top-score threshold below
// which we treat semantic results as noise rather than signal. Picked
// empirically: real matches on this index cluster ≥0.5; nonsense
// queries ("frobnicate the quux gizmo") tend to score ≤0.4 on whatever
// chunk happens to share a token.
const lowConfidenceScore = 0.45

// buildNextAction returns an imperative sentence the agent can execute
// directly. The issue is explicit that prose outperforms structured
// args for agent compliance. Always concrete — names paths and line
// ranges — never "do more research."
//
// The "weak semantic" fallback fires only when the intent's *primary*
// payload is also empty. For graph-driven intents (package_topology /
// architecture) a populated graph counts as confidence even when the
// semantic-hit scores are low; for editing_context, populated blame
// annotations count likewise. This prevents the misleading
// "rephrase or grep" message on calls that actually returned useful
// structural data.
func buildNextAction(intent string, reads []SuggestedRead, symbols []SymbolHit, topSemScore float32, graphEdgeCount int, hasBlame bool) string {
	if len(reads) == 0 && len(symbols) == 0 && graphEdgeCount == 0 {
		return "Rephrase the question with concrete keywords or fall back to grep."
	}
	// Confidence comes from any of: symbol hits, strong semantic score,
	// or an intent-specific structural payload.
	intentPayloadStrong := false
	switch intent {
	case IntentPackageTopology, IntentArchitecture:
		intentPayloadStrong = graphEdgeCount > 0
	case IntentEditingContext:
		intentPayloadStrong = hasBlame
	}
	weakSemantic := topSemScore > 0 && topSemScore < lowConfidenceScore
	if len(symbols) == 0 && weakSemantic && !intentPayloadStrong {
		return "Top semantic match is weak — rephrase with concrete keywords or fall back to grep."
	}
	switch intent {
	case IntentSymbolLookup:
		// Only claim "the definition" when a symbol actually matched —
		// reads[0] without symbols is a semantic neighbor, not the
		// definition the user asked about.
		if len(symbols) > 0 && len(reads) > 0 {
			// Multiple definitions across distinct paths is a real
			// shape for ambiguous names (`Options` exists in chat,
			// graph, index, store, watch). Signal that — singular
			// "the definition" hides matches the agent should know
			// about.
			if distinctSymbolPaths(symbols) > 1 {
				return fmt.Sprintf("%d definitions across files — closest is %s lines %d-%d; consult the full `symbols` array for the rest.",
					distinctSymbolPaths(symbols), reads[0].Path, reads[0].StartLine, reads[0].EndLine)
			}
			return fmt.Sprintf("Read %s lines %d-%d to see the definition.", reads[0].Path, reads[0].StartLine, reads[0].EndLine)
		}
		if len(symbols) == 0 && len(reads) > 0 {
			return fmt.Sprintf("No exact symbol match — the closest semantic neighbor is %s lines %d-%d. Verify there before assuming the identifier exists.",
				reads[0].Path, reads[0].StartLine, reads[0].EndLine)
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
	case IntentPackageTopology:
		if graphEdgeCount > 0 {
			return fmt.Sprintf("Read the `graph.edges` list (%d imports) to see package dependencies, then call with intent=symbol_lookup on a specific package to drill in.", graphEdgeCount)
		}
		if len(reads) > 0 {
			return readsSkimDirective(reads)
		}
	case IntentArchitecture:
		if len(reads) > 0 {
			return readsSkimDirective(reads)
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

// distinctSymbolPaths counts the number of unique paths across a
// SymbolHit slice. Used by buildNextAction to signal when a single
// identifier resolves to multiple definitions (e.g. `Options` exists
// in chat, graph, index, store, watch packages) so the agent reads
// the full symbols array rather than stopping at the first read.
func distinctSymbolPaths(syms []SymbolHit) int {
	seen := make(map[string]struct{}, len(syms))
	for _, s := range syms {
		if s.Path == "" {
			continue
		}
		seen[s.Path] = struct{}{}
	}
	return len(seen)
}

// readsSkimDirective renders the multi-file skim hint used by
// architecture / package_topology when the graph isn't the headline.
func readsSkimDirective(reads []SuggestedRead) string {
	parts := make([]string, 0, len(reads))
	for _, r := range reads {
		parts = append(parts, fmt.Sprintf("%s lines %d-%d", r.Path, r.StartLine, r.EndLine))
	}
	return "Skim " + strings.Join(parts, "; ") + " for the structural overview, then re-call with intent=symbol_lookup to drill into specific types, or intent=editing_context for files you want to modify."
}

// hasBlameAnnotations reports whether any path in the annotations map
// carries blame metadata — the signal that buildNextAction uses to
// avoid emitting "weak match" on editing_context responses that have
// concrete authorship data.
func hasBlameAnnotations(anns map[string]PathMeta) bool {
	for _, m := range anns {
		if m.LastCommit != "" || m.LastAuthor != "" {
			return true
		}
	}
	return false
}

// buildAvoid emits a "what not to do" hint. Strong claims when we
// have strong signals (exact symbol found → don't grep); softer
// otherwise. `graphIndexed` is true when the project has a graph
// available — `callers`/`callees` still warn because layer 1 doesn't
// emit `calls` edges, regardless. `hasRefs` softens the callers/callees
// message: when ripgrep already populated a reference list the agent
// has the surface it needs, so the message shifts from "verify with
// grep" to "do not re-grep, the list is here."
func buildAvoid(intent string, semHits []SemHit, symbols []SymbolHit, graphIndexed, hasRefs bool) string {
	if _, deferred := graphDeferredIntents[intent]; deferred {
		if hasRefs {
			return "Do not grep for the identifier — the `references` field already lists usages. Treat it as a best-effort lexical list (no `calls` graph yet); rely on it for navigation, verify edge cases by reading the snippets."
		}
		return "Do not trust the symbols list as exhaustive — `calls` edges are not yet extracted, so caller/callee coverage is best-effort. Verify with grep on the symbol name."
	}
	if !graphIndexed {
		return "Graph not indexed for this project — results from semantic + symbol lanes only. Run `mcsearch index <project>` to refresh both layers (graph extraction is part of the default index run)."
	}
	// Exploration intents — the user is forming a mental model, so
	// the failure mode to discourage is breadth (enumerating files,
	// re-deriving the topology) rather than depth (reading whole files).
	switch intent {
	case IntentArchitecture:
		return "Do not enumerate the file tree — the graph nodes and suggested reads ARE the structural overview. Start there before broader exploration."
	case IntentPackageTopology:
		return "Do not infer imports by grepping — the graph edges encode them. Use the topology, don't rebuild it."
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

// ─── inline file contents into suggested_reads ────────────────────────────

// inlineCaps are the per-intent budgets for inlineSuggestedReads.
// Exploration intents (architecture, package_topology) get a denser
// bundle than targeted ones — the caller is forming a mental model,
// so giving them more files / longer slices saves multiple round-trips
// vs. saving a few KB of response.
type inlineCaps struct {
	maxLinesPerRead int
	maxBytesPerRead int
	totalBytesCap   int
}

func inlineCapsFor(intent string) inlineCaps {
	switch intent {
	case IntentArchitecture, IntentPackageTopology:
		return inlineCaps{maxLinesPerRead: 120, maxBytesPerRead: 8 * 1024, totalBytesCap: 40 * 1024}
	default:
		return inlineCaps{maxLinesPerRead: 60, maxBytesPerRead: 4 * 1024, totalBytesCap: 12 * 1024}
	}
}

// inlineContent fills the Content/Truncated fields on suggested_reads
// and semantic_hits from a single per-intent byte budget, so the
// caller gets a usable bundle without follow-up Reads. Suggested_reads
// are filled first (they are the curated cut); remaining budget then
// covers semantic_hits in order. A small read cache means a range
// that appears in both lanes is loaded once and charged once against
// the budget.
//
// Bounds are enforced at two levels: per-read (lines + bytes) and
// total bytes across both arrays. Caps scale with intent (see
// inlineCapsFor). Failures (missing file, unreadable, scanner error)
// leave Content empty and the caller still has Path/StartLine/EndLine
// to fall back on a manual Read.
func inlineContent(projectRoot, intent string, reads []SuggestedRead, sem []SemHit) {
	caps := inlineCapsFor(intent)
	budget := caps.totalBytesCap

	type key struct {
		path           string
		start, end     int
		maxLines, maxB int
	}
	type cached struct {
		content   string
		truncated bool
	}
	cache := map[key]cached{}

	fetch := func(path string, start, end int) (string, bool, bool) {
		// Returns (content, truncated, charged) where charged=true
		// means we drew from the budget on this call (cache miss).
		perBytes := min(caps.maxBytesPerRead, budget)
		k := key{path, start, end, caps.maxLinesPerRead, perBytes}
		if c, ok := cache[k]; ok {
			return c.content, c.truncated, false
		}
		if budget <= 0 {
			return "", false, false
		}
		abs := path
		if !filepath.IsAbs(abs) {
			abs = filepath.Join(projectRoot, abs)
		}
		content, truncated, err := readLineRange(abs, start, end, caps.maxLinesPerRead, perBytes)
		if err != nil {
			return "", false, false
		}
		cache[k] = cached{content, truncated}
		return content, truncated, true
	}

	for i := range reads {
		if budget <= 0 {
			return
		}
		// Entries with pre-populated Content (summary kinds carrying
		// synthesized prose) skip disk I/O but still charge the byte
		// budget so the total response size stays bounded.
		if reads[i].Content != "" {
			budget -= len(reads[i].Content)
			continue
		}
		content, truncated, charged := fetch(reads[i].Path, reads[i].StartLine, reads[i].EndLine)
		if content == "" && !truncated {
			continue
		}
		reads[i].Content = content
		reads[i].Truncated = truncated
		if charged {
			budget -= len(content)
		}
	}
	// On a no-signal query (top semantic score below the confidence
	// threshold) the whole pool is likely noise. Skip inlining hits
	// whose individual score is also below the noise floor — the agent
	// keeps the path/range pointer but we don't burn bytes on a Content
	// blob that won't pay off.
	var topScore float32
	if len(sem) > 0 {
		topScore = sem[0].Score
	}
	suppressLowScore := topScore > 0 && topScore < lowConfidenceScore
	for i := range sem {
		if budget <= 0 {
			return
		}
		if sem[i].Content != "" {
			budget -= len(sem[i].Content)
			continue
		}
		if suppressLowScore && sem[i].Score < noiseFloorScore {
			continue
		}
		content, truncated, charged := fetch(sem[i].Path, sem[i].StartLine, sem[i].EndLine)
		if content == "" && !truncated {
			continue
		}
		sem[i].Content = content
		sem[i].Truncated = truncated
		if charged {
			budget -= len(content)
		}
	}
}

// readLineRange returns the 1-indexed [start, end] line slice of a
// file, clipped at maxLines and maxBytes. truncated reports whether
// either cap fired before reaching end.
func readLineRange(path string, start, end, maxLines, maxBytes int) (string, bool, error) {
	if maxLines <= 0 || maxBytes <= 0 {
		return "", false, nil
	}
	if start <= 0 {
		start = 1
	}
	if end < start {
		end = start
	}
	f, err := os.Open(path)
	if err != nil {
		return "", false, err
	}
	defer func() { _ = f.Close() }()

	sc := bufio.NewScanner(f)
	// Lift the default 64 KB line cap so minified files don't bail —
	// we still bound the output via maxBytes below.
	sc.Buffer(make([]byte, 64*1024), 1024*1024)

	var buf strings.Builder
	lineNum := 0
	included := 0
	truncated := false
	for sc.Scan() {
		lineNum++
		if lineNum < start {
			continue
		}
		if lineNum > end {
			break
		}
		if included >= maxLines {
			truncated = true
			break
		}
		line := sc.Bytes()
		if buf.Len()+len(line)+1 > maxBytes {
			truncated = true
			break
		}
		buf.Write(line)
		buf.WriteByte('\n')
		included++
	}
	if err := sc.Err(); err != nil {
		return "", false, err
	}
	// If we exited the loop because the file ended before EndLine,
	// that's not truncation by the cap — leave truncated as-is.
	return buf.String(), truncated, nil
}
