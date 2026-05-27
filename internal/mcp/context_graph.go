// Package mcp provides graph integration for the `ask` router.
//
// internal/graph emits nodes (package/file/function/method/type/
// struct/interface/field/import) and edges (contains/imports/
// has_method/has_field/embeds/implements/calls — the last Go-only).
// The intents and what they get:
//
//	symbol_lookup     — neighbors of the matched symbol (sibling
//	                    methods, fields, embedded types) so the agent
//	                    sees the whole "shape" of a type without
//	                    reading the file.
//	editing_context   — same neighborhood, plus the enclosing type
//	                    so refactors know what else uses the type.
//	architecture      — package/type roll-up for packages surfaced
//	                    by the semantic lane.
//	package_topology  — import edges between packages in the
//	                    semantic neighborhood.
//	callers           — incoming calls edges into matched symbols
//	                    (Go-only; falls back to ripgrep usage list
//	                    for other languages via context.go).
//	callees           — outgoing calls edges from matched symbols.
//
// Loader strategy: a single in-memory view per request. With the
// current scale (~800 nodes for this repo) that's a few hundred KB;
// when it stops fitting we can move to targeted SQL queries.
package mcp

import (
	"context"
	"sort"
	"strings"

	"github.com/alehatsman/dex/internal/graph"
	"github.com/alehatsman/dex/internal/store"
)

// graphView holds an in-memory snapshot of graph_nodes/graph_edges
// indexed for the queries the router needs. All maps point into the
// same underlying node/edge slices so memory cost is one slice copy.
type graphView struct {
	nodesByID        map[string]graphNode
	nodesByName      map[string][]graphNode // bare name → matching nodes
	nodesByQualified map[string][]graphNode // qualified name → matching nodes
	nodesByPackage   map[string][]graphNode // package path → all nodes in pkg
	nodesByPath      map[string][]graphNode // file path → all nodes in file
	edgesBySrc       map[string][]graphEdge
	edgesByDst       map[string][]graphEdge
	edgesByKind      map[graph.EdgeKind][]graphEdge
}

type graphNode struct {
	ID            string
	Kind          graph.NodeKind
	Name          string
	QualifiedName string
	PackagePath   string
	FilePath      string
	StartLine     int
	EndLine       int
	// Centrality columns, populated from graph_nodes. Used by call-edge
	// tools to sort peers by importance and to compose the role hint
	// attached to each result.
	InDegree        int
	OutDegree       int
	CrossPkgCallers int
	PageRank        float64
}

type graphEdge struct {
	Kind      graph.EdgeKind
	SrcID     string
	DstID     string
	FilePath  string
	StartLine int
}

// loadGraphView pulls every node and edge from the store and indexes
// them. Returns nil (no error) when the project has no graph indexed
// — the caller should treat that as "graph not available."
func loadGraphView(ctx context.Context, st *store.Store) (*graphView, error) {
	nodes, edges, err := st.GraphStats(ctx)
	if err != nil {
		return nil, err
	}
	if nodes == 0 && edges == 0 {
		return nil, nil
	}

	nodeRows, err := st.GraphAllNodes(ctx)
	if err != nil {
		return nil, err
	}
	edgeRows, err := st.GraphAllEdges(ctx)
	if err != nil {
		return nil, err
	}

	v := &graphView{
		nodesByID:        make(map[string]graphNode, len(nodeRows)),
		nodesByName:      map[string][]graphNode{},
		nodesByQualified: map[string][]graphNode{},
		nodesByPackage:   map[string][]graphNode{},
		nodesByPath:      map[string][]graphNode{},
		edgesBySrc:       map[string][]graphEdge{},
		edgesByDst:       map[string][]graphEdge{},
		edgesByKind:      map[graph.EdgeKind][]graphEdge{},
	}
	for _, r := range nodeRows {
		n := graphNode{
			ID:              r.ID,
			Kind:            graph.NodeKind(r.Kind),
			Name:            r.Name,
			QualifiedName:   r.QualifiedName,
			PackagePath:     r.PackagePath,
			FilePath:        r.FilePath,
			StartLine:       r.StartLine,
			EndLine:         r.EndLine,
			InDegree:        r.InDegree,
			OutDegree:       r.OutDegree,
			CrossPkgCallers: r.CrossPkgCallers,
			PageRank:        r.PageRank,
		}
		v.nodesByID[n.ID] = n
		if n.Name != "" {
			v.nodesByName[n.Name] = append(v.nodesByName[n.Name], n)
		}
		if n.QualifiedName != "" && n.QualifiedName != n.Name {
			v.nodesByQualified[n.QualifiedName] = append(v.nodesByQualified[n.QualifiedName], n)
		}
		if n.PackagePath != "" {
			v.nodesByPackage[n.PackagePath] = append(v.nodesByPackage[n.PackagePath], n)
		}
		if n.FilePath != "" {
			v.nodesByPath[n.FilePath] = append(v.nodesByPath[n.FilePath], n)
		}
	}
	for _, r := range edgeRows {
		e := graphEdge{
			Kind:      graph.EdgeKind(r.Kind),
			SrcID:     r.SrcID,
			DstID:     r.DstID,
			FilePath:  r.FilePath,
			StartLine: r.StartLine,
		}
		v.edgesBySrc[e.SrcID] = append(v.edgesBySrc[e.SrcID], e)
		v.edgesByDst[e.DstID] = append(v.edgesByDst[e.DstID], e)
		v.edgesByKind[e.Kind] = append(v.edgesByKind[e.Kind], e)
	}
	return v, nil
}

// chunkPageRank resolves a chunk's PageRank via the in-memory graph
// view. Used by pickSuggestedReads as a tiebreaker for exploration
// intents (architecture / package_topology) so a high-centrality hub
// like Indexer.Run beats a marginally-higher-scored tuning doc when
// scores cluster.
//
// Resolution prefers the node whose declared line range covers
// startLine; falls back to the highest-PageRank node in the file when
// none matches (file-level summary chunks point at line 0-0 and we
// want the file's most-central symbol to represent them). Returns 0
// when no graph node exists for the path — non-Go files, top-level
// consts, no graph indexed — which makes the tiebreaker degrade
// silently to "no preference."
func chunkPageRank(view *graphView, path string, startLine int) float64 {
	if view == nil {
		return 0
	}
	nodes := view.nodesByPath[path]
	if len(nodes) == 0 {
		return 0
	}
	var bestCovering float64
	for _, n := range nodes {
		if startLine >= n.StartLine && startLine <= n.EndLine && n.PageRank > bestCovering {
			bestCovering = n.PageRank
		}
	}
	if bestCovering > 0 {
		return bestCovering
	}
	var bestAny float64
	for _, n := range nodes {
		if n.PageRank > bestAny {
			bestAny = n.PageRank
		}
	}
	return bestAny
}

// enrichGraph populates GraphResult based on the resolved intent.
// Mutates out.Graph in place. Returns whether anything was emitted —
// the caller uses this to keep avoid/next_action consistent.
//
// Node IDs and edge from/to are rewritten to a compact form
// (`<pkg-tail>.<qualified-name>`) so agents don't have to parse
// `<module>::<pkg>::<kind>::<qname>` for every reference. The full
// IDs remain available via the in-memory view for any future query
// that takes a graph ID as input.
// maxGraphNodes / maxGraphEdges bound the graph lane so a big package
// (e.g. mooncake's cmd/* with dozens of entries) can't blow the
// response budget by itself. Hit empirically: 30/50 keeps the lane
// useful for orientation without dominating the payload, and survives
// `no_inline: true` so callers can rely on bundle size being roughly
// proportional to k. Truncation is silent — once full, further nodes
// and edges are dropped.
const (
	maxGraphNodes = 30
	maxGraphEdges = 50
)

func enrichGraph(out *ContextOutput, intent string, view *graphView, semHits []SemHit, symbols []SymbolHit) bool {
	if view == nil {
		return false
	}
	e := &graphEnricher{
		view:     view,
		semHits:  semHits,
		symbols:  symbols,
		gr:       &GraphResult{Nodes: []GraphNode{}, Edges: []GraphEdge{}},
		seenNode: map[string]bool{},
		seenEdge: map[string]bool{},
	}
	e.runForIntent(intent)
	if len(e.gr.Nodes) == 0 && len(e.gr.Edges) == 0 {
		return false
	}
	out.Graph = e.gr
	return true
}

// graphEnricher carries the working state for one enrichGraph call.
// Hoisting the closures off enrichGraph into methods keeps the dispatch
// switch short and the helpers individually testable.
type graphEnricher struct {
	view     *graphView
	semHits  []SemHit
	symbols  []SymbolHit
	gr       *GraphResult
	seenNode map[string]bool
	seenEdge map[string]bool
}

func (e *graphEnricher) addNode(n graphNode) {
	// Import nodes are emitted per-file in layer 1, so the same
	// dependency (e.g. `fmt`) shows up as N distinct node IDs.
	// Dedup on QualifiedName for imports so the agent sees one
	// entry per dependency, not one per importing file.
	key := n.ID
	if n.Kind == graph.NodeImport && n.QualifiedName != "" {
		key = "import:" + n.QualifiedName
	}
	if e.seenNode[key] || len(e.gr.Nodes) >= maxGraphNodes {
		return
	}
	e.seenNode[key] = true
	// For imports the compactID already encodes the import path, so
	// QualifiedName is redundant — drop it on the wire.
	qname := n.QualifiedName
	if n.Kind == graph.NodeImport {
		qname = ""
	}
	e.gr.Nodes = append(e.gr.Nodes, GraphNode{
		ID:            compactID(n),
		QualifiedName: qname,
		Kind:          string(n.Kind),
	})
}

func (e *graphEnricher) addEdge(ge graphEdge) {
	from, to := ge.SrcID, ge.DstID
	if n, ok := e.view.nodesByID[ge.SrcID]; ok {
		from = compactID(n)
	}
	if n, ok := e.view.nodesByID[ge.DstID]; ok {
		to = compactID(n)
	}
	// Dedup on the compact (from,kind,to) triple — the raw IDs
	// can differ for per-file import nodes that collapse to the
	// same dependency on the wire (see addNode), so a raw-ID
	// dedup leaks duplicates like `src -> fmt` × N.
	key := from + "|" + string(ge.Kind) + "|" + to
	if e.seenEdge[key] || len(e.gr.Edges) >= maxGraphEdges {
		return
	}
	e.seenEdge[key] = true
	e.gr.Edges = append(e.gr.Edges, GraphEdge{
		From: from,
		To:   to,
		Kind: string(ge.Kind),
	})
}

// symbolNeighborhood surfaces each matched symbol's container (parent
// type for methods/fields) and its siblings (other methods/fields on
// the same type, embedded types).
func (e *graphEnricher) symbolNeighborhood() {
	for _, sym := range e.symbols {
		lookup := e.view.nodesByName[sym.QualifiedName]
		if len(lookup) == 0 {
			// Some MCP symbol hits use the bare method name even
			// when the graph stored a qualified form like (*T).M.
			lookup = e.view.nodesByQualified[sym.QualifiedName]
		}
		for _, n := range lookup {
			e.addNode(n)
			for _, parentEdge := range e.view.edgesByDst[n.ID] {
				if parentEdge.Kind != graph.EdgeHasMethod && parentEdge.Kind != graph.EdgeHasField {
					continue
				}
				parent, ok := e.view.nodesByID[parentEdge.SrcID]
				if !ok {
					continue
				}
				e.addNode(parent)
				e.addEdge(parentEdge)
				for _, sibling := range e.view.edgesBySrc[parent.ID] {
					if sibling.Kind != graph.EdgeHasMethod && sibling.Kind != graph.EdgeHasField && sibling.Kind != graph.EdgeEmbeds {
						continue
					}
					e.addEdge(sibling)
					if dst, ok := e.view.nodesByID[sibling.DstID]; ok {
						e.addNode(dst)
					}
				}
			}
		}
	}
}

// packageRollup adds package + top-level type/function nodes for every
// package in pkgs.
func (e *graphEnricher) packageRollup(pkgs map[string]struct{}) {
	for pkg := range pkgs {
		for _, n := range e.view.nodesByPackage[pkg] {
			switch n.Kind {
			case graph.NodePackage, graph.NodeType, graph.NodeStruct, graph.NodeInterface, graph.NodeFunction:
				e.addNode(n)
			}
		}
	}
}

// architectureAnchorPkgs caps how many top-PageRank packages seed the
// architecture rollup. Picked to fill maxGraphNodes/maxGraphEdges with
// a meaningful cross-section without burning the budget on one package.
const architectureAnchorPkgs = 8

// topPackagesByPageRank returns the K packages with the highest
// aggregate PageRank (sum across all nodes in the package). Used by
// architecture rollup to seed the graph with the project's central
// packages instead of depending on whatever semHits happened to surface
// — a docs-dominated semantic lane otherwise collapses the rollup to
// the single Go file that leaked in. Packages with zero aggregate
// PageRank are skipped: missing centrality data means the graph rerank
// pass hasn't run and seeding would be arbitrary.
func (v *graphView) topPackagesByPageRank(k int) map[string]struct{} {
	type pkgScore struct {
		pkg string
		pr  float64
	}
	scores := make([]pkgScore, 0, len(v.nodesByPackage))
	for pkg, nodes := range v.nodesByPackage {
		var sum float64
		for _, n := range nodes {
			sum += n.PageRank
		}
		if sum <= 0 {
			continue
		}
		scores = append(scores, pkgScore{pkg, sum})
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].pr > scores[j].pr })
	if len(scores) > k {
		scores = scores[:k]
	}
	out := make(map[string]struct{}, len(scores))
	for _, s := range scores {
		out[s.pkg] = struct{}{}
	}
	return out
}

// callsExpansion walks the calls-edges in or out of the matched
// symbols. Direction picks which side the symbol is on:
// `dst` = callers (edges arriving at the symbol), `src` = callees.
func (e *graphEnricher) callsExpansion(direction string) {
	for _, sym := range e.symbols {
		lookup := e.view.nodesByQualified[sym.QualifiedName]
		if len(lookup) == 0 {
			lookup = e.view.nodesByName[sym.QualifiedName]
		}
		for _, n := range lookup {
			e.addNode(n)
			edges := e.view.edgesBySrc[n.ID]
			if direction == "dst" {
				edges = e.view.edgesByDst[n.ID]
			}
			for _, ge := range edges {
				if ge.Kind != graph.EdgeCalls {
					continue
				}
				peerID := ge.SrcID
				if direction == "src" {
					peerID = ge.DstID
				}
				if peer, ok := e.view.nodesByID[peerID]; ok {
					e.addNode(peer)
					e.addEdge(ge)
				}
			}
		}
	}
}

// packageTopology surfaces imports between packages in the semantic
// neighborhood. Always seeds package nodes themselves so the topology
// has anchors even when no import edges resolve.
func (e *graphEnricher) packageTopology() {
	pkgs := packagesFromPaths(e.view, e.semHits)
	for pkg := range pkgs {
		for _, n := range e.view.nodesByPackage[pkg] {
			if n.Kind == graph.NodePackage {
				e.addNode(n)
			}
		}
	}
	for _, ge := range e.view.edgesByKind[graph.EdgeImports] {
		srcN, srcOK := e.view.nodesByID[ge.SrcID]
		if !srcOK {
			continue
		}
		if _, in := pkgs[srcN.PackagePath]; !in {
			continue
		}
		e.addNode(srcN)
		if dst, ok := e.view.nodesByID[ge.DstID]; ok {
			e.addNode(dst)
		}
		e.addEdge(ge)
	}
}

// runForIntent dispatches to the right expansion mix for the intent.
// Default branch (behavior_search / unrecognized) unions
// symbol-neighborhood + package rollup so the caller always sees
// structural context when a graph exists.
func (e *graphEnricher) runForIntent(intent string) {
	switch intent {
	case IntentSymbolLookup, IntentEditingContext:
		e.symbolNeighborhood()
	case IntentCallers:
		e.callsExpansion("dst")
		if len(e.gr.Nodes) == 0 {
			e.symbolNeighborhood()
		}
	case IntentCallees:
		e.callsExpansion("src")
		if len(e.gr.Nodes) == 0 {
			e.symbolNeighborhood()
		}
	case IntentArchitecture:
		// Anchor on the project's structurally central packages so the
		// rollup stays useful even when the semantic lane skews to docs
		// and surfaces only one Go file by accident. PageRank-derived
		// anchors first; semHit-derived packages augment so a question
		// that does point at a specific subsystem still pulls that
		// subsystem in.
		pkgs := e.view.topPackagesByPageRank(architectureAnchorPkgs)
		for pkg := range packagesFromPaths(e.view, e.semHits) {
			pkgs[pkg] = struct{}{}
		}
		e.packageRollup(pkgs)
	case IntentPackageTopology:
		e.packageTopology()
	default:
		e.symbolNeighborhood()
		e.packageRollup(packagesFromPaths(e.view, e.semHits))
	}
}

// packagesFromPaths collects the set of package paths that contain at
// least one of the file paths in semHits. Lets architecture /
// package_topology focus on the neighborhood the user is actually
// asking about, instead of dumping the whole graph.
func packagesFromPaths(view *graphView, semHits []SemHit) map[string]struct{} {
	pkgs := map[string]struct{}{}
	for _, h := range semHits {
		for _, n := range view.nodesByPath[h.Path] {
			if n.PackagePath != "" {
				pkgs[n.PackagePath] = struct{}{}
			}
		}
	}
	return pkgs
}

// compactID condenses internal/graph.NodeID's
// `<module>::<pkg>::<kind>::<qualified-name>` into a form an agent can
// scan at a glance. Kept stable within one response so edges and
// nodes refer to the same string. Examples:
//
//	mcp.(*Server).ContextRouter    — methods, functions, types, fields
//	internal/mcp                    — packages (qualified_name *is* the path)
//	github.com/foo/bar              — imports (qualified_name is the path)
func compactID(n graphNode) string {
	switch n.Kind {
	case graph.NodePackage:
		if n.PackagePath != "" {
			return pkgTail(n.PackagePath)
		}
		return n.QualifiedName
	case graph.NodeImport:
		return n.QualifiedName
	}
	tail := pkgTail(n.PackagePath)
	if n.QualifiedName != "" {
		if tail != "" {
			return tail + "." + n.QualifiedName
		}
		return n.QualifiedName
	}
	if tail != "" && n.Name != "" {
		return tail + "." + n.Name
	}
	return n.Name
}

// pkgTail returns the last path segment of pkg, e.g.
// "github.com/x/y/internal/mcp" → "mcp". Empty string in, empty out.
func pkgTail(pkg string) string {
	if i := strings.LastIndex(pkg, "/"); i >= 0 {
		return pkg[i+1:]
	}
	return pkg
}
