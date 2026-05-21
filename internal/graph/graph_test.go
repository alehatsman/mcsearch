package graph

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/alehatsman/dex/internal/proj"
	"github.com/alehatsman/dex/internal/store"
)

// copyFixture copies testdata/<name> into a fresh temp dir and returns
// the temp path. Tests that mutate files (prune, idempotency) need a
// writable copy so they don't bleed into other test runs.
func copyFixture(t *testing.T, name string) string {
	t.Helper()
	src := filepath.Join("testdata", name)
	dst := t.TempDir()
	if err := filepath.WalkDir(src, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, path)
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(target, data, 0o644)
	}); err != nil {
		t.Fatalf("copy fixture: %v", err)
	}
	return dst
}

func openTestStore(t *testing.T) *store.Store {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "test.db")
	st, err := store.Open(context.Background(), dbPath)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	return st
}

// resolveTestProject points a *proj.Project at root without going
// through proj.Resolve (which would EvalSymlinks). Tests that mutate
// files in t.TempDir() work the same either way; this skips the
// allow-list check.
func resolveTestProject(t *testing.T, root string) *proj.Project {
	t.Helper()
	cacheDir := t.TempDir()
	dbPath := filepath.Join(cacheDir, "index.db")
	return &proj.Project{Root: root, ID: "test", CacheDir: cacheDir, DBPath: dbPath}
}

// findNode returns the node with the given qualifiedName and kind, or
// nil if not present. Tests assert presence, not absence-of-others, so
// the graph can grow without breaking these.
func findNode(nodes []Node, kind NodeKind, qn string) *Node {
	for i := range nodes {
		if nodes[i].Kind == kind && nodes[i].QualifiedName == qn {
			return &nodes[i]
		}
	}
	return nil
}

func findEdge(edges []Edge, kind EdgeKind, srcID, dstID string) *Edge {
	for i := range edges {
		if edges[i].Kind == kind && edges[i].SrcID == srcID && edges[i].DstID == dstID {
			return &edges[i]
		}
	}
	return nil
}

func TestExtractGoFixture(t *testing.T) {
	root := copyFixture(t, "simple")
	res, err := ExtractGo(context.Background(), root)
	if err != nil {
		t.Fatalf("ExtractGo: %v (warnings=%v)", err, res)
	}
	if res.Packages < 2 {
		t.Errorf("packages: got %d, want >=2 (main, store)", res.Packages)
	}

	// Package nodes.
	pkgStore := findNode(res.Nodes, NodePackage, "example.com/simple/store")
	if pkgStore == nil {
		t.Fatalf("missing package node example.com/simple/store; nodes=%v", nodeQNs(res.Nodes))
	}
	pkgMain := findNode(res.Nodes, NodePackage, "example.com/simple")
	if pkgMain == nil {
		t.Fatalf("missing package node example.com/simple")
	}

	// Method nodes — (*Store).Get, (*Store).Set, (*Inner).lookup, (*Inner).write.
	for _, qn := range []string{"(*Store).Get", "(*Store).Set", "(*Inner).lookup", "(*Inner).write"} {
		if findNode(res.Nodes, NodeMethod, qn) == nil {
			t.Errorf("missing method node %s; methods=%v", qn, nodesOfKind(res.Nodes, NodeMethod))
		}
	}

	// Function node main.
	if findNode(res.Nodes, NodeFunction, "main") == nil {
		t.Errorf("missing function node main")
	}

	// Struct node Store.
	if findNode(res.Nodes, NodeStruct, "Store") == nil {
		t.Errorf("missing struct node Store")
	}
	// Struct node DB.
	if findNode(res.Nodes, NodeStruct, "DB") == nil {
		t.Errorf("missing struct node DB")
	}
	// Interface nodes.
	if findNode(res.Nodes, NodeInterface, "Reader") == nil {
		t.Errorf("missing interface node Reader")
	}
	if findNode(res.Nodes, NodeInterface, "ReadCloser") == nil {
		t.Errorf("missing interface node ReadCloser")
	}

	// File node store.go (relative path inside fixture).
	if findNode(res.Nodes, NodeFile, "example.com/simple/store/store.go") == nil {
		t.Errorf("missing file node example.com/simple/store/store.go; files=%v",
			nodesOfKind(res.Nodes, NodeFile))
	}

	// Import edge: example.com/simple → example.com/simple/store.
	mainID := pkgMain.ID
	importID := NodeID(modulePathOf(res), "example.com/simple", NodeImport, "example.com/simple/store")
	if findEdge(res.Edges, EdgeImports, mainID, importID) == nil {
		t.Errorf("missing imports edge example.com/simple → example.com/simple/store")
	}

	// has_method: Store → (*Store).Get.
	typeStoreID := NodeID(modulePathOf(res), "example.com/simple/store", NodeType, "Store")
	getMethodID := NodeID(modulePathOf(res), "example.com/simple/store", NodeMethod, "(*Store).Get")
	if findEdge(res.Edges, EdgeHasMethod, typeStoreID, getMethodID) == nil {
		t.Errorf("missing has_method edge Store → (*Store).Get")
	}

	// has_field: Store → Store.db.
	dbFieldID := NodeID(modulePathOf(res), "example.com/simple/store", NodeField, "Store.db")
	if findEdge(res.Edges, EdgeHasField, typeStoreID, dbFieldID) == nil {
		t.Errorf("missing has_field edge Store → Store.db")
	}

	// embeds: DB → Inner.
	dbTypeID := NodeID(modulePathOf(res), "example.com/simple/store", NodeType, "DB")
	innerTypeID := NodeID(modulePathOf(res), "example.com/simple/store", NodeType, "Inner")
	if findEdge(res.Edges, EdgeEmbeds, dbTypeID, innerTypeID) == nil {
		t.Errorf("missing embeds edge DB → Inner")
	}

	// embeds on interface: ReadCloser → Reader.
	rcID := NodeID(modulePathOf(res), "example.com/simple/store", NodeType, "ReadCloser")
	readerID := NodeID(modulePathOf(res), "example.com/simple/store", NodeType, "Reader")
	if findEdge(res.Edges, EdgeEmbeds, rcID, readerID) == nil {
		t.Errorf("missing embeds edge ReadCloser → Reader")
	}

	// implements: *Impl satisfies Reader — edge is from the named type Impl.
	implID := NodeID(modulePathOf(res), "example.com/simple/store", NodeType, "Impl")
	if findEdge(res.Edges, EdgeImplements, implID, readerID) == nil {
		t.Errorf("missing implements edge Impl → Reader; edges=%v", edgeKinds(res.Edges, EdgeImplements))
	}
}

// modulePathOf re-derives the module path from a result so tests
// don't hardcode it. ExtractGo embeds it into every ID via NodeID.
func modulePathOf(res *ExtractResult) string {
	for _, n := range res.Nodes {
		if n.Kind == NodePackage {
			// ID = "<module>::<pkg>::package::<pkg>"; split on "::".
			parts := strings.SplitN(n.ID, "::", 2)
			return parts[0]
		}
	}
	return ""
}

func nodeQNs(nodes []Node) []string {
	out := make([]string, 0, len(nodes))
	for _, n := range nodes {
		out = append(out, string(n.Kind)+":"+n.QualifiedName)
	}
	return out
}

func nodesOfKind(nodes []Node, kind NodeKind) []string {
	var out []string
	for _, n := range nodes {
		if n.Kind == kind {
			out = append(out, n.QualifiedName)
		}
	}
	return out
}

func edgeKinds(edges []Edge, kind EdgeKind) []string {
	var out []string
	for _, e := range edges {
		if e.Kind == kind {
			out = append(out, e.SrcID+" → "+e.DstID)
		}
	}
	return out
}

func TestIndexerIdempotent(t *testing.T) {
	root := copyFixture(t, "simple")
	st := openTestStore(t)
	p := resolveTestProject(t, root)
	gx := New(p, NewStoreAdapter(st), Options{})

	stats1, err := gx.Run(context.Background())
	if err != nil {
		t.Fatalf("run 1: %v", err)
	}
	if stats1.NodesUpserted == 0 {
		t.Fatal("run 1 upserted zero nodes")
	}

	// Second pass — same source, no changes.
	stats2, err := gx.Run(context.Background())
	if err != nil {
		t.Fatalf("run 2: %v", err)
	}
	if stats2.NodesPruned != 0 || stats2.EdgesPruned != 0 {
		t.Errorf("run 2 pruned rows on unchanged source: nodes=%d edges=%d",
			stats2.NodesPruned, stats2.EdgesPruned)
	}
	// Final counts in the DB should equal stats2's upserts.
	nodes, edges, err := st.GraphStats(context.Background())
	if err != nil {
		t.Fatalf("graph stats: %v", err)
	}
	if nodes != stats2.NodesUpserted {
		t.Errorf("db nodes=%d, upserted=%d", nodes, stats2.NodesUpserted)
	}
	if edges != stats2.EdgesUpserted {
		t.Errorf("db edges=%d, upserted=%d", edges, stats2.EdgesUpserted)
	}
}

func TestPruneRemovedFile(t *testing.T) {
	root := copyFixture(t, "simple")
	st := openTestStore(t)
	p := resolveTestProject(t, root)
	gx := New(p, NewStoreAdapter(st), Options{})

	if _, err := gx.Run(context.Background()); err != nil {
		t.Fatalf("first run: %v", err)
	}
	nodesBefore, _, _ := st.GraphStats(context.Background())

	// Delete types.go; re-run.
	if err := os.Remove(filepath.Join(root, "store", "types.go")); err != nil {
		t.Fatalf("remove types.go: %v", err)
	}
	// Give the second run a strictly later cutoff than the first.
	time.Sleep(2 * time.Millisecond)
	stats, err := gx.Run(context.Background())
	if err != nil {
		t.Fatalf("second run: %v", err)
	}
	if stats.NodesPruned == 0 {
		t.Errorf("expected some nodes pruned after deleting types.go, got 0")
	}
	nodesAfter, _, _ := st.GraphStats(context.Background())
	if nodesAfter >= nodesBefore {
		t.Errorf("node count did not drop after deletion: before=%d after=%d",
			nodesBefore, nodesAfter)
	}
}

func TestChunkLinkage(t *testing.T) {
	root := copyFixture(t, "simple")
	st := openTestStore(t)

	// Seed a chunk row that covers the (*Store).Get method's line range
	// in the fixture. The exact lines depend on the fixture; we
	// over-cover by giving the chunk a generous range.
	ctx := context.Background()
	storeGoPath := filepath.Join("store", "store.go")
	if err := st.UpsertMany(ctx, []store.PendingChunk{{
		Path:       storeGoPath,
		Kind:       "method_declaration",
		Name:       "Get",
		StartLine:  1,
		EndLine:    100,
		ContentSHA: "deadbeef",
		Content:    "stub",
		Vec:        []float32{1, 0, 0, 0},
	}}, time.Now()); err != nil {
		t.Fatalf("seed chunk: %v", err)
	}

	p := resolveTestProject(t, root)
	gx := New(p, NewStoreAdapter(st), Options{})
	stats, err := gx.Run(ctx)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if stats.LinkedToChunks == 0 {
		t.Fatal("expected at least one node linked to a chunk")
	}
	// Verify the link is persisted: query DB nodes and find one whose
	// FilePath matches the seeded chunk's path and whose chunk_id != 0.
	nodes, err := st.GraphAllNodes(ctx)
	if err != nil {
		t.Fatalf("read nodes: %v", err)
	}
	var hit bool
	for _, n := range nodes {
		if n.FilePath == storeGoPath && n.ChunkID > 0 {
			hit = true
			break
		}
	}
	if !hit {
		t.Errorf("no node in %s ended up with chunk_id set", storeGoPath)
	}
}

func TestSchemaMigration(t *testing.T) {
	// A fresh store should already have graph_nodes / graph_edges
	// after Open() (which runs migrate()).
	st := openTestStore(t)
	nodes, edges, err := st.GraphStats(context.Background())
	if err != nil {
		t.Fatalf("GraphStats on fresh store: %v", err)
	}
	if nodes != 0 || edges != 0 {
		t.Errorf("fresh store: nodes=%d edges=%d, want 0/0", nodes, edges)
	}
}

func TestExportJSONLRoundTrip(t *testing.T) {
	root := copyFixture(t, "simple")
	st := openTestStore(t)
	p := resolveTestProject(t, root)
	gx := New(p, NewStoreAdapter(st), Options{})
	if _, err := gx.Run(context.Background()); err != nil {
		t.Fatalf("run: %v", err)
	}

	out := t.TempDir()
	if err := ExportJSONL(context.Background(), NewStoreAdapter(st), out); err != nil {
		t.Fatalf("ExportJSONL: %v", err)
	}

	// Parse nodes.jsonl line by line and assert at least one is valid.
	nodesFile, err := os.Open(filepath.Join(out, "nodes.jsonl"))
	if err != nil {
		t.Fatalf("open nodes.jsonl: %v", err)
	}
	defer nodesFile.Close()

	var count int
	r := bufio.NewReader(nodesFile)
	for {
		line, err := r.ReadBytes('\n')
		if len(line) > 0 {
			var n nodeJSON
			if uerr := json.Unmarshal(bytesNoNewline(line), &n); uerr != nil {
				t.Fatalf("parse node line: %v line=%q", uerr, line)
			}
			if n.ID == "" || n.Kind == "" {
				t.Errorf("node line missing id or kind: %+v", n)
			}
			count++
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("read nodes.jsonl: %v", err)
		}
	}
	if count == 0 {
		t.Fatal("exported nodes.jsonl had zero lines")
	}

	// Sanity check edges file existence and non-emptiness.
	edgesInfo, err := os.Stat(filepath.Join(out, "edges.jsonl"))
	if err != nil {
		t.Fatalf("stat edges.jsonl: %v", err)
	}
	if edgesInfo.Size() == 0 {
		t.Errorf("edges.jsonl is empty")
	}
}

func bytesNoNewline(b []byte) []byte {
	if len(b) > 0 && b[len(b)-1] == '\n' {
		return b[:len(b)-1]
	}
	return b
}

// TestExtractGoCalls asserts the calls-edge extractor lights up the
// expected edges on the existing `simple` fixture. The fixture
// already exercises: cross-package function call (main → NewStore),
// cross-package method call (main → (*Store).Get), and same-package
// method calls on an embedded receiver ((*Store).Get → (*Inner).lookup,
// (*Store).Set → (*Inner).write). Calls into the stdlib (fmt.Println)
// must NOT produce an edge — the inTree filter is the line of defence
// against dangling dst IDs.
func TestExtractGoCalls(t *testing.T) {
	root := copyFixture(t, "simple")
	res, err := ExtractGo(context.Background(), root)
	if err != nil {
		t.Fatalf("ExtractGo: %v", err)
	}
	mod := modulePathOf(res)
	if mod == "" {
		t.Fatalf("could not infer module path")
	}

	storePkg := "example.com/simple/store"
	mainPkg := "example.com/simple"

	mainID := NodeID(mod, mainPkg, NodeFunction, "main")
	newStoreID := NodeID(mod, storePkg, NodeFunction, "NewStore")
	storeGetID := NodeID(mod, storePkg, NodeMethod, "(*Store).Get")
	storeSetID := NodeID(mod, storePkg, NodeMethod, "(*Store).Set")
	innerLookupID := NodeID(mod, storePkg, NodeMethod, "(*Inner).lookup")
	innerWriteID := NodeID(mod, storePkg, NodeMethod, "(*Inner).write")

	cases := []struct {
		name        string
		src, dst    string
		wantExists  bool
		failHelpMsg string
	}{
		{"main → NewStore", mainID, newStoreID, true, "cross-package function call should emit an edge"},
		{"main → (*Store).Get", mainID, storeGetID, true, "cross-package method call should emit an edge"},
		{"(*Store).Get → (*Inner).lookup", storeGetID, innerLookupID, true, "same-package method call on embedded receiver should emit an edge"},
		{"(*Store).Set → (*Inner).write", storeSetID, innerWriteID, true, "same-package method call on embedded receiver should emit an edge"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if findEdge(res.Edges, EdgeCalls, c.src, c.dst) == nil {
				t.Errorf("missing calls edge %s; calls=%v\nhelp: %s",
					c.name, edgeKinds(res.Edges, EdgeCalls), c.failHelpMsg)
			}
		})
	}

	// Negative case: nothing in the project calls fmt.Println, but
	// even if main did, the inTree filter must skip stdlib targets.
	// Confirm by checking no calls edge has a dst with "fmt" anywhere.
	for _, e := range res.Edges {
		if e.Kind != EdgeCalls {
			continue
		}
		if strings.Contains(e.DstID, "::fmt::") {
			t.Errorf("calls edge leaks stdlib target: %s → %s", e.SrcID, e.DstID)
		}
	}

	// Idempotency: re-extract and confirm the same edge IDs collide
	// (set count stays stable). Mirrors how the persister upserts.
	res2, err := ExtractGo(context.Background(), root)
	if err != nil {
		t.Fatalf("ExtractGo (second pass): %v", err)
	}
	n1, n2 := countCalls(res.Edges), countCalls(res2.Edges)
	if n1 != n2 {
		t.Errorf("calls edge count changed across runs: %d → %d", n1, n2)
	}
}

func countCalls(edges []Edge) int {
	n := 0
	for _, e := range edges {
		if e.Kind == EdgeCalls {
			n++
		}
	}
	return n
}
