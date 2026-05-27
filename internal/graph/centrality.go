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
// Only edges with Kind == EdgeCalls are considered for direct counting.
// On top of those, virtual forwarding edges are synthesized from each
// interface method to every concrete method that implements it (via
// EdgeImplements + EdgeHasMethod). Without this synthesis every method
// reachable only through interface dispatch (e.g. concrete `Handler.Run`
// implementations called as `h.Run()`) would report in-degree 0 and
// look dead to a reader. Forwarding routes the interface method's
// incoming weight to its implementers — uniform split across N impls
// is the right semantic when the runtime type is unknown to the
// extractor.
//
// Both endpoints of an edge (real or virtual) must resolve in the
// supplied nodes slice; dangling edges (a calls edge pointing at a
// node we don't have) are skipped. Self-loops (a calls a) are ignored
// for degree counting but kept for PageRank — they're rare and PageRank
// handles them benignly.
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

	// Interface-dispatch forwarding. For each interface method I_M
	// implemented by concrete method T_M, add a virtual edge I_M → T_M
	// so that:
	//   - T_M gets in-degree ≥ 1 (no longer looks like dead code)
	//   - PageRank flows from a high-rank interface method to its
	//     implementers, divided uniformly across the N impls
	//
	// CrossPkgCallers is intentionally NOT incremented from virtual
	// edges: the real caller's package is no longer reachable through
	// the forwarding hop, so we'd be guessing. Direct callers already
	// credit M_I correctly.
	forwarding := buildImplementsForwarding(nodeByID, edges)
	for ifaceMID, concMIDs := range forwarding {
		if _, ok := nodeByID[ifaceMID]; !ok {
			continue
		}
		for _, concMID := range concMIDs {
			if _, ok := nodeByID[concMID]; !ok {
				continue
			}
			if concMID == ifaceMID {
				continue
			}
			distinctEdge[edgeKey{src: ifaceMID, dst: concMID}] = struct{}{}
			if outAdj[ifaceMID] == nil {
				outAdj[ifaceMID] = map[string]struct{}{}
			}
			outAdj[ifaceMID][concMID] = struct{}{}
		}
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

// buildImplementsForwarding returns interface-method ID → list of
// concrete-method IDs that should inherit the interface method's
// incoming weight. Built from two edge kinds:
//
//	EdgeImplements:  concrete-type → interface-type
//	EdgeHasMethod:   type → method (its receiver)
//
// For each (T implements I) pair, the interface's methods I.M are
// matched to the concrete's methods T.M by Node.Name. Same-named
// methods are joined; methods present on the interface but missing on
// the concrete (would mean T doesn't actually implement I) are skipped
// silently — go/types already verified the implements relationship.
//
// Generic methods are filtered out via the same "skip non-Named"
// path that extractImplements uses, so this stays in sync with which
// edges actually got emitted.
func buildImplementsForwarding(nodeByID map[string]Node, edges []Edge) map[string][]string {
	// typeID → method-name → method-node-ID. Built once from
	// EdgeHasMethod edges so the implements loop below stays linear in
	// the implements-edge count.
	methodsByType := make(map[string]map[string]string)
	for _, e := range edges {
		if e.Kind != EdgeHasMethod {
			continue
		}
		dst, ok := nodeByID[e.DstID]
		if !ok {
			continue
		}
		if methodsByType[e.SrcID] == nil {
			methodsByType[e.SrcID] = make(map[string]string)
		}
		methodsByType[e.SrcID][dst.Name] = e.DstID
	}

	out := make(map[string][]string)
	for _, e := range edges {
		if e.Kind != EdgeImplements {
			continue
		}
		concMethods := methodsByType[e.SrcID]
		ifaceMethods := methodsByType[e.DstID]
		if len(concMethods) == 0 || len(ifaceMethods) == 0 {
			continue
		}
		for name, ifaceMID := range ifaceMethods {
			concMID, ok := concMethods[name]
			if !ok {
				continue
			}
			out[ifaceMID] = append(out[ifaceMID], concMID)
		}
	}
	return out
}
