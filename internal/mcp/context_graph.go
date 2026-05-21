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
	gr := &GraphResult{Nodes: []GraphNode{}, Edges: []GraphEdge{}}
	seenNode := map[string]bool{}
	seenEdge := map[string]bool{}
	addNode := func(n graphNode) {
		// Import nodes are emitted per-file in layer 1, so the same
		// dependency (e.g. `fmt`) shows up as N distinct node IDs.
		// Dedup on QualifiedName for imports so the agent sees one
		// entry per dependency, not one per importing file.
		key := n.ID
		if n.Kind == graph.NodeImport && n.QualifiedName != "" {
			key = "import:" + n.QualifiedName
		}
		if seenNode[key] || len(gr.Nodes) >= maxGraphNodes {
			return
		}
		seenNode[key] = true
		// For imports the compactID already encodes the import path, so
		// QualifiedName is redundant — drop it on the wire.
		qname := n.QualifiedName
		if n.Kind == graph.NodeImport {
			qname = ""
		}
		gr.Nodes = append(gr.Nodes, GraphNode{
			ID:            compactID(n),
			QualifiedName: qname,
			Kind:          string(n.Kind),
		})
	}
	addEdge := func(e graphEdge) {
		from, to := e.SrcID, e.DstID
		if n, ok := view.nodesByID[e.SrcID]; ok {
			from = compactID(n)
		}
		if n, ok := view.nodesByID[e.DstID]; ok {
			to = compactID(n)
		}
		// Dedup on the compact (from,kind,to) triple — the raw IDs
		// can differ for per-file import nodes that collapse to the
		// same dependency on the wire (see addNode), so a raw-ID
		// dedup leaks duplicates like `src -> fmt` × N.
		key := from + "|" + string(e.Kind) + "|" + to
		if seenEdge[key] || len(gr.Edges) >= maxGraphEdges {
			return
		}
		seenEdge[key] = true
		gr.Edges = append(gr.Edges, GraphEdge{
			From: from,
			To:   to,
			Kind: string(e.Kind),
		})
	}

	// graphSymbolNeighborhood and graphPackageRollup are the two
	// reusable expansions; they were inlined in the symbol_lookup /
	// architecture cases and are now used by the default branch too
	// so every intent yields populated nodes/edges when a graph is
	// indexed.
	symbolNeighborhood := func() {
		for _, sym := range symbols {
			lookup := view.nodesByName[sym.QualifiedName]
			if len(lookup) == 0 {
				// Some MCP symbol hits use the bare method name even
				// when the graph stored a qualified form like (*T).M.
				lookup = view.nodesByQualified[sym.QualifiedName]
			}
			for _, n := range lookup {
				addNode(n)
				for _, parentEdge := range view.edgesByDst[n.ID] {
					if parentEdge.Kind != graph.EdgeHasMethod && parentEdge.Kind != graph.EdgeHasField {
						continue
					}
					parent, ok := view.nodesByID[parentEdge.SrcID]
					if !ok {
						continue
					}
					addNode(parent)
					addEdge(parentEdge)
					for _, sibling := range view.edgesBySrc[parent.ID] {
						if sibling.Kind != graph.EdgeHasMethod && sibling.Kind != graph.EdgeHasField && sibling.Kind != graph.EdgeEmbeds {
							continue
						}
						addEdge(sibling)
						if dst, ok := view.nodesByID[sibling.DstID]; ok {
							addNode(dst)
						}
					}
				}
			}
		}
	}
	packageRollup := func() {
		pkgs := packagesFromPaths(view, semHits)
		for pkg := range pkgs {
			for _, n := range view.nodesByPackage[pkg] {
				switch n.Kind {
				case graph.NodePackage, graph.NodeType, graph.NodeStruct, graph.NodeInterface, graph.NodeFunction:
					addNode(n)
				}
			}
		}
	}

	// callsExpansion walks the calls-edges in or out of the matched
	// symbols. Direction picks which side the symbol is on:
	// `dst` = callers (edges arriving at the symbol), `src` = callees.
	callsExpansion := func(direction string) {
		for _, sym := range symbols {
			lookup := view.nodesByQualified[sym.QualifiedName]
			if len(lookup) == 0 {
				lookup = view.nodesByName[sym.QualifiedName]
			}
			for _, n := range lookup {
				addNode(n)
				var edges []graphEdge
				if direction == "dst" {
					edges = view.edgesByDst[n.ID]
				} else {
					edges = view.edgesBySrc[n.ID]
				}
				for _, e := range edges {
					if e.Kind != graph.EdgeCalls {
						continue
					}
					peerID := e.SrcID
					if direction == "src" {
						peerID = e.DstID
					}
					if peer, ok := view.nodesByID[peerID]; ok {
						addNode(peer)
						addEdge(e)
					}
				}
			}
		}
	}

	switch intent {
	case IntentSymbolLookup, IntentEditingContext:
		// For each matched symbol, surface its container (parent type
		// for methods/fields) and its siblings (other methods/fields
		// on the same type, embedded types).
		symbolNeighborhood()

	case IntentCallers:
		// Incoming calls edges. Falls back to neighborhood when no
		// calls edges resolve (e.g. a non-Go file matched the symbol).
		callsExpansion("dst")
		if len(gr.Nodes) == 0 {
			symbolNeighborhood()
		}

	case IntentCallees:
		callsExpansion("src")
		if len(gr.Nodes) == 0 {
			symbolNeighborhood()
		}

	case IntentArchitecture:
		// Package + top-level type/function rollup for packages
		// surfaced by the semantic lane.
		packageRollup()

	case IntentPackageTopology:
		// Imports between packages in the semantic neighborhood.
		pkgs := packagesFromPaths(view, semHits)
		// Always include the package nodes themselves so the topology
		// has anchors.
		for pkg := range pkgs {
			for _, n := range view.nodesByPackage[pkg] {
				if n.Kind == graph.NodePackage {
					addNode(n)
				}
			}
		}
		for _, e := range view.edgesByKind[graph.EdgeImports] {
			srcN, srcOK := view.nodesByID[e.SrcID]
			if !srcOK {
				continue
			}
			if _, in := pkgs[srcN.PackagePath]; !in {
				continue
			}
			addNode(srcN)
			if dst, ok := view.nodesByID[e.DstID]; ok {
				addNode(dst)
			}
			addEdge(e)
		}

	default:
		// behavior_search / unrecognized — fall back to the union of
		// symbol-neighborhood + package rollup so the caller always
		// sees structural context when a graph exists. Cheap: the
		// view is already in memory.
		symbolNeighborhood()
		packageRollup()
	}

	if len(gr.Nodes) == 0 && len(gr.Edges) == 0 {
		return false
	}
	out.Graph = gr
	return true
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
