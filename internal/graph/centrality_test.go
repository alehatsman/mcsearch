package graph

import (
	"context"
	"math"
	"testing"
)

// TestComputeCentrality builds a tiny call graph by hand and verifies
// the degree / cross-pkg / PageRank outputs. Keeps the math test
// decoupled from go/types extraction.
func TestComputeCentrality(t *testing.T) {
	// Three nodes across two packages:
	//   a (pkg P) calls b (pkg P)
	//   a (pkg P) calls c (pkg Q)
	//   d (pkg Q) calls c (pkg Q)
	//   e (pkg Q) calls c (pkg Q)
	//
	// Expectations:
	//   c.InDegree         = 3 (callers a, d, e)
	//   c.CrossPkgCallers  = 1 (pkg P, not Q)
	//   b.InDegree         = 1, b.CrossPkgCallers = 0 (same-pkg only)
	//   a.OutDegree        = 2 (b, c)
	nodes := []Node{
		{ID: "a", Kind: NodeFunction, PackagePath: "P"},
		{ID: "b", Kind: NodeFunction, PackagePath: "P"},
		{ID: "c", Kind: NodeFunction, PackagePath: "Q"},
		{ID: "d", Kind: NodeFunction, PackagePath: "Q"},
		{ID: "e", Kind: NodeFunction, PackagePath: "Q"},
	}
	edges := []Edge{
		{Kind: EdgeCalls, SrcID: "a", DstID: "b"},
		{Kind: EdgeCalls, SrcID: "a", DstID: "c"},
		{Kind: EdgeCalls, SrcID: "d", DstID: "c"},
		{Kind: EdgeCalls, SrcID: "e", DstID: "c"},
		// Repeated call-site between the same pair — must NOT inflate
		// in_degree (distinct (src,dst) only).
		{Kind: EdgeCalls, SrcID: "a", DstID: "c", StartLine: 99},
		// Non-calls edge — must be ignored entirely.
		{Kind: EdgeHasMethod, SrcID: "a", DstID: "b"},
	}

	got := ComputeCentrality(nodes, edges)

	if got["c"].InDegree != 3 {
		t.Errorf("c.InDegree = %d, want 3", got["c"].InDegree)
	}
	if got["c"].CrossPkgCallers != 1 {
		t.Errorf("c.CrossPkgCallers = %d, want 1 (only pkg P)", got["c"].CrossPkgCallers)
	}
	if got["b"].InDegree != 1 {
		t.Errorf("b.InDegree = %d, want 1", got["b"].InDegree)
	}
	if got["b"].CrossPkgCallers != 0 {
		t.Errorf("b.CrossPkgCallers = %d, want 0 (a, b both in P)", got["b"].CrossPkgCallers)
	}
	if got["a"].OutDegree != 2 {
		t.Errorf("a.OutDegree = %d, want 2", got["a"].OutDegree)
	}
	// PageRank should sum to ~1.0 (probability distribution).
	var total float64
	for _, n := range nodes {
		total += got[n.ID].PageRank
	}
	if math.Abs(total-1.0) > 1e-6 {
		t.Errorf("pagerank sum = %v, want ~1.0", total)
	}
	// c has the most callers from the most packages, so it must have
	// the highest PageRank among reachable nodes.
	for _, id := range []string{"a", "b", "d", "e"} {
		if got["c"].PageRank <= got[id].PageRank {
			t.Errorf("pagerank: c (%v) should be > %s (%v)", got["c"].PageRank, id, got[id].PageRank)
		}
	}
}

// TestComputeCentralityEmpty exercises the zero-edge case — no calls
// edges means nothing to rank, and the function should return an empty
// map rather than NaN-filled noise.
func TestComputeCentralityEmpty(t *testing.T) {
	nodes := []Node{{ID: "a", Kind: NodeFunction, PackagePath: "P"}}
	got := ComputeCentrality(nodes, nil)
	if len(got) != 0 {
		t.Errorf("got %d entries for empty edges, want 0", len(got))
	}
}

// TestComputeCentralitySkipsDanglingEdges confirms edges whose endpoints
// aren't in the node set are dropped silently (cross-module calls into
// std lib, etc.) instead of inventing ghost nodes.
func TestComputeCentralitySkipsDanglingEdges(t *testing.T) {
	nodes := []Node{{ID: "a", Kind: NodeFunction, PackagePath: "P"}}
	edges := []Edge{
		{Kind: EdgeCalls, SrcID: "a", DstID: "missing"},
		{Kind: EdgeCalls, SrcID: "ghost", DstID: "a"},
	}
	got := ComputeCentrality(nodes, edges)
	if len(got) != 0 {
		t.Errorf("got %d entries with all-dangling edges, want 0", len(got))
	}
}

// TestRunPersistsCentrality exercises the full Indexer.Run path against
// the `simple` fixture so the centrality columns actually land in the
// store. Asserts that at least one node ends up with non-zero in_degree,
// which catches both the computation and the persistence wiring.
func TestRunPersistsCentrality(t *testing.T) {
	root := copyFixture(t, "simple")
	st := openTestStore(t)
	p := resolveTestProject(t, root)

	g := New(p, NewStoreAdapter(st), Options{})
	if _, err := g.Run(context.Background()); err != nil {
		t.Fatalf("Run: %v", err)
	}

	nodes, err := st.GraphAllNodes(context.Background())
	if err != nil {
		t.Fatalf("GraphAllNodes: %v", err)
	}
	var nonZero int
	for _, n := range nodes {
		if n.InDegree > 0 || n.OutDegree > 0 || n.PageRank > 0 {
			nonZero++
		}
	}
	if nonZero == 0 {
		t.Fatalf("no nodes carry centrality after Run — expected calls-edges in the simple fixture")
	}
}
