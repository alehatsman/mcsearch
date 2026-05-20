package graph

// Centrality computation over the static call graph.
//
// Three statistics, all derived from the `calls` edges:
//
//   in_degree         — number of distinct callers (a -> b counts once
//                       even if a calls b multiple times).
//   out_degree        — number of distinct callees.
//   cross_pkg_callers — number of distinct caller PACKAGES other than
//                       the callee's own. Filters out high-fanin glue
//                       (loggers, errors) whose callers are all in the
//                       same package as themselves — and surfaces true
//                       domain hubs that real cross-package code uses.
//   pagerank          — classic damped random walk on the call graph,
//                       restart prob 0.15, 20 iterations. Quietly
//                       up-weights deeply-connected hubs even when
//                       their raw in_degree is unremarkable.
//
// All four end up on graph_nodes via store.GraphSetCentrality and feed
// the MCP tools' result ranking. Nodes that the call graph doesn't
// touch (types, fields, packages, ...) keep zero on every column —
// callers should treat zero as "unknown" rather than "low," since the
// computation skips entire node classes by design.

// pageRankIterations is the fixed iteration count. With damping 0.85
// and a sub-10k-node graph the score converges to within ~1e-4 well
// before 20 iterations; this is the standard "good enough" budget.
const pageRankIterations = 20

// pageRankDamping is the canonical 0.85 from Brin & Page. Damping
// represents the probability that a random surfer follows an outgoing
// edge instead of teleporting to a uniformly random node.
const pageRankDamping = 0.85

// CentralityResult is one node's computed statistics. Returned by
// ComputeCentrality keyed by node ID. Nodes that are never referenced
// by any `calls` edge are absent from the map (the caller can treat
// missing == all-zero).
type CentralityResult struct {
	InDegree        int
	OutDegree       int
	CrossPkgCallers int
	PageRank        float64
}

// ComputeCentrality walks the `calls` edges and returns per-node
// statistics keyed by node ID.
//
// Only edges with Kind == EdgeCalls are considered. Both endpoints
// must resolve in the supplied nodes slice; dangling edges (a calls
// edge pointing at a node we don't have) are skipped. Self-loops
// (a calls a) are ignored for degree counting but kept for PageRank
// — they're rare and PageRank handles them benignly.
func ComputeCentrality(nodes []Node, edges []Edge) map[string]CentralityResult {
	nodeByID := make(map[string]Node, len(nodes))
	for _, n := range nodes {
		nodeByID[n.ID] = n
	}

	// Distinct (src, dst) and (src, dst-pkg) sets for degree counting.
	// Multiple call-sites from the same source to the same target only
	// count once; otherwise a function called 5 times from a loop gets
	// the same caller credited 5 times.
	type edgeKey struct{ src, dst string }
	type pkgKey struct {
		callee string
		srcPkg string
	}
	distinctEdge := make(map[edgeKey]struct{}, len(edges))
	distinctSrcPkg := make(map[pkgKey]struct{}, len(edges))

	// Adjacency for PageRank — outgoing per source, with distinct
	// destinations only (same dedup as in_degree counting).
	outAdj := make(map[string]map[string]struct{}, len(nodes))

	for _, e := range edges {
		if e.Kind != EdgeCalls {
			continue
		}
		srcN, srcOK := nodeByID[e.SrcID]
		dstN, dstOK := nodeByID[e.DstID]
		if !srcOK || !dstOK {
			continue
		}
		distinctEdge[edgeKey{src: e.SrcID, dst: e.DstID}] = struct{}{}
		if srcN.PackagePath != "" && srcN.PackagePath != dstN.PackagePath {
			distinctSrcPkg[pkgKey{callee: e.DstID, srcPkg: srcN.PackagePath}] = struct{}{}
		}
		if outAdj[e.SrcID] == nil {
			outAdj[e.SrcID] = map[string]struct{}{}
		}
		outAdj[e.SrcID][e.DstID] = struct{}{}
	}

	out := make(map[string]CentralityResult, len(distinctEdge))
	for k := range distinctEdge {
		r := out[k.dst]
		r.InDegree++
		out[k.dst] = r
		r = out[k.src]
		r.OutDegree++
		out[k.src] = r
	}
	for k := range distinctSrcPkg {
		r := out[k.callee]
		r.CrossPkgCallers++
		out[k.callee] = r
	}

	// PageRank. Only run when the graph has at least one calls edge —
	// otherwise the uniform 1/N rank is meaningless noise on every node.
	if len(distinctEdge) > 0 {
		ranks := pageRank(nodes, outAdj)
		for id, r := range ranks {
			rec := out[id]
			rec.PageRank = r
			out[id] = rec
		}
	}
	return out
}

// pageRank runs the classic damped iteration. Nodes with no outgoing
// edges (dangling) redistribute their score uniformly to every node
// each round, which is the standard correction — without it the total
// score leaks out of the graph and rankings tilt toward sinks.
func pageRank(nodes []Node, outAdj map[string]map[string]struct{}) map[string]float64 {
	n := len(nodes)
	if n == 0 {
		return nil
	}
	ids := make([]string, 0, n)
	for _, node := range nodes {
		ids = append(ids, node.ID)
	}
	rank := make(map[string]float64, n)
	initial := 1.0 / float64(n)
	for _, id := range ids {
		rank[id] = initial
	}

	teleport := (1.0 - pageRankDamping) / float64(n)

	for iter := 0; iter < pageRankIterations; iter++ {
		// Dangling mass — score sitting on nodes with no outgoing
		// edges. Redistribute it uniformly so probability mass is
		// conserved across the iteration.
		var dangling float64
		for _, id := range ids {
			if _, ok := outAdj[id]; !ok {
				dangling += rank[id]
			}
		}
		danglingShare := pageRankDamping * dangling / float64(n)

		next := make(map[string]float64, n)
		for _, id := range ids {
			next[id] = teleport + danglingShare
		}
		for src, dsts := range outAdj {
			if len(dsts) == 0 {
				continue
			}
			contribution := pageRankDamping * rank[src] / float64(len(dsts))
			for dst := range dsts {
				next[dst] += contribution
			}
		}
		rank = next
	}
	return rank
}
