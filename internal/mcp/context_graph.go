// Package mcp provides graph integration for mcsearch_context.
//
// Layer 1 of internal/graph supplies nodes (package/file/function/
// method/type/struct/interface/field/import) and structural edges
// (contains/imports/has_method/has_field/embeds). It does NOT yet
// emit `calls` edges — so the callers/callees intents still degrade
// to a semantic + symbol fallback. The intents that benefit today:
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
//
// Loader strategy: a single in-memory view per request. With the
// current scale (~800 nodes for this repo) that's a few hundred KB;
// when it stops fitting we can move to targeted SQL queries.
package mcp

import (
	"context"
	"strings"

	"github.com/alehatsman/mcsearch/internal/graph"
	"github.com/alehatsman/mcsearch/internal/store"
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
}

type graphEdge struct {
	Kind     graph.EdgeKind
	SrcID    string
	DstID    string
	FilePath string
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
			ID:            r.ID,
			Kind:          graph.NodeKind(r.Kind),
			Name:          r.Name,
			QualifiedName: r.QualifiedName,
			PackagePath:   r.PackagePath,
			FilePath:      r.FilePath,
			StartLine:     r.StartLine,
			EndLine:       r.EndLine,
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
			Kind:     graph.EdgeKind(r.Kind),
			SrcID:    r.SrcID,
			DstID:    r.DstID,
			FilePath: r.FilePath,
		}
		v.edgesBySrc[e.SrcID] = append(v.edgesBySrc[e.SrcID], e)
		v.edgesByDst[e.DstID] = append(v.edgesByDst[e.DstID], e)
		v.edgesByKind[e.Kind] = append(v.edgesByKind[e.Kind], e)
	}
	return v, nil
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
func enrichGraph(out *ContextOutput, intent string, view *graphView, semHits []SemHit, symbols []SymbolHit) bool {
	if view == nil {
		return false
	}
	gr := &GraphResult{Nodes: []GraphNode{}, Edges: []GraphEdge{}}
	seenNode := map[string]bool{}
	seenEdge := map[string]bool{}
	addNode := func(n graphNode) {
		if seenNode[n.ID] {
			return
		}
		seenNode[n.ID] = true
		gr.Nodes = append(gr.Nodes, GraphNode{
			ID:            compactID(n),
			QualifiedName: n.QualifiedName,
			Kind:          string(n.Kind),
		})
	}
	addEdge := func(e graphEdge) {
		key := e.SrcID + "|" + string(e.Kind) + "|" + e.DstID
		if seenEdge[key] {
			return
		}
		seenEdge[key] = true
		from, to := e.SrcID, e.DstID
		if n, ok := view.nodesByID[e.SrcID]; ok {
			from = compactID(n)
		}
		if n, ok := view.nodesByID[e.DstID]; ok {
			to = compactID(n)
		}
		gr.Edges = append(gr.Edges, GraphEdge{
			From: from,
			To:   to,
			Kind: string(e.Kind),
		})
	}

	switch intent {
	case IntentSymbolLookup, IntentEditingContext:
		// For each matched symbol, surface its container (parent type
		// for methods/fields) and its siblings (other methods/fields
		// on the same type, embedded types).
		for _, sym := range symbols {
			lookup := view.nodesByName[sym.QualifiedName]
			if len(lookup) == 0 {
				// Some MCP symbol hits use the bare method name even
				// when the graph stored a qualified form like (*T).M.
				lookup = view.nodesByQualified[sym.QualifiedName]
			}
			for _, n := range lookup {
				addNode(n)
				// Siblings via has_method / has_field on the same parent.
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

	case IntentArchitecture:
		// Package + top-level type/function rollup for packages
		// surfaced by the semantic lane.
		pkgs := packagesFromPaths(view, semHits)
		for pkg := range pkgs {
			for _, n := range view.nodesByPackage[pkg] {
				switch n.Kind {
				case graph.NodePackage, graph.NodeType, graph.NodeStruct, graph.NodeInterface, graph.NodeFunction:
					addNode(n)
				}
			}
		}

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

// graphSupportsIntent returns true when the graph layer can
// meaningfully contribute to this intent. Used by buildAvoid to
// suppress the "graph deferred" warning once the lane is wired.
func graphSupportsIntent(intent string) bool {
	switch intent {
	case IntentSymbolLookup, IntentEditingContext, IntentArchitecture, IntentPackageTopology:
		return true
	}
	return false
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
