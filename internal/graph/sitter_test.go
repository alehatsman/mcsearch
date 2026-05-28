package graph

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync/atomic"
	"testing"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/python"
)

// stubExtractor lets the framework tests exercise the dispatch loop
// without depending on a real per-language extractor. It records every
// file it sees and emits one synthetic node + edge per file so callers
// can assert the framework's stamping/aggregation behaviour.
type stubExtractor struct {
	name       string
	extensions []string
	// dropExt, if non-empty, signals that this extractor wants to
	// ignore files with this extension once observed — exposes the
	// per-instance state isolation guarantee.
	visited       []string // RelPath of every ProcessFile call
	initCalls     int32
	finalizeCalls int32
	failInit      bool
	failProcess   bool
	failFinalize  bool
}

func (s *stubExtractor) Name() string               { return s.name }
func (s *stubExtractor) Language() *sitter.Language { return python.GetLanguage() }
func (s *stubExtractor) Extensions() []string       { return s.extensions }
func (s *stubExtractor) Init(_ context.Context, _ string) error {
	atomic.AddInt32(&s.initCalls, 1)
	if s.failInit {
		return errors.New("boom-init")
	}
	return nil
}
func (s *stubExtractor) ProcessFile(_ context.Context, in FileInput) error {
	s.visited = append(s.visited, in.RelPath)
	if s.failProcess {
		return errors.New("boom-process: " + in.RelPath)
	}
	return nil
}
func (s *stubExtractor) Finalize(_ context.Context) ([]Node, []Edge, []string, error) {
	atomic.AddInt32(&s.finalizeCalls, 1)
	if s.failFinalize {
		return nil, nil, nil, errors.New("boom-finalize")
	}
	// Emit one synthetic node per visited file + one self-edge so the
	// framework's provenance stamping is observable.
	nodes := make([]Node, 0, len(s.visited))
	edges := make([]Edge, 0, len(s.visited))
	for _, rel := range s.visited {
		id := NodeID("", s.name, NodeFile, rel)
		nodes = append(nodes, Node{
			ID:            id,
			Kind:          NodeFile,
			Name:          filepath.Base(rel),
			QualifiedName: rel,
			PackagePath:   s.name,
			FilePath:      rel,
		})
		edges = append(edges, Edge{
			ID:    EdgeID(id, EdgeContains, id, rel, 0),
			Kind:  EdgeContains,
			SrcID: id,
			DstID: id,
		})
	}
	return nodes, edges, nil, nil
}

// writeTree builds a small project tree on disk for the framework tests.
// keys are slash-form relpaths from root; values are file contents.
func writeTree(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for rel, content := range files {
		abs := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
		if err := os.WriteFile(abs, []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", rel, err)
		}
	}
	return root
}

// TestExtractSitterEmptyRegistry confirms the default (no extractors
// registered) is a clean no-op. This is the load-bearing guarantee for
// landing the framework before any language plug-in — existing graph
// tests must not behave differently.
func TestExtractSitterEmptyRegistry(t *testing.T) {
	root := writeTree(t, map[string]string{
		"a.py": "x = 1\n",
		"b.go": "package main\nfunc main() {}\n",
	})

	reg := NewRegistry()
	res, err := ExtractSitterWith(context.Background(), root, reg)
	if err != nil {
		t.Fatalf("ExtractSitterWith: %v", err)
	}
	if len(res.Nodes) != 0 || len(res.Edges) != 0 || len(res.Warnings) != 0 {
		t.Errorf("empty registry should produce empty result; got nodes=%d edges=%d warnings=%v",
			len(res.Nodes), len(res.Edges), res.Warnings)
	}
}

// TestExtractSitterDefaultEntry covers the wrapper that uses
// DefaultRegistry — the actual entry point Indexer.Run calls. Asserts
// the path stays alive (any registered extractor that claims .py
// dispatches); doesn't couple the framework test to the specific
// shape the live Python extractor emits.
func TestExtractSitterDefaultEntry(t *testing.T) {
	root := writeTree(t, map[string]string{"a.py": "x = 1\n"})
	res, err := ExtractSitter(context.Background(), root)
	if err != nil {
		t.Fatalf("ExtractSitter: %v", err)
	}
	var sawFile bool
	for _, n := range res.Nodes {
		if n.Kind == NodeFile && n.FilePath == "a.py" {
			sawFile = true
			break
		}
	}
	if !sawFile {
		t.Errorf("expected DefaultRegistry to dispatch a.py and emit a file node; got nodes=%v",
			nodeQNs(res.Nodes))
	}
}

// TestExtractSitterDispatch wires a stub extractor against a tree of
// .py files and asserts (a) every matching file is dispatched, (b)
// non-matching files are skipped, (c) emitted edges carry the
// framework's provenance stamps.
func TestExtractSitterDispatch(t *testing.T) {
	root := writeTree(t, map[string]string{
		"a.py":      "x = 1\n",
		"sub/b.py":  "y = 2\n",
		"sub/c.go":  "package sub\nfunc Hi() {}\n",
		"sub/d.txt": "not code",
		"sub/e.PY":  "Z = 3\n", // uppercase extension exercises ToLower
	})

	stub := &stubExtractor{name: "pystub", extensions: []string{".py"}}
	reg := NewRegistry()
	reg.Register(func() Extractor { return stub })

	res, err := ExtractSitterWith(context.Background(), root, reg)
	if err != nil {
		t.Fatalf("ExtractSitterWith: %v", err)
	}

	wantVisited := []string{"a.py", "sub/b.py", "sub/e.PY"}
	got := append([]string(nil), stub.visited...)
	sort.Strings(got)
	sort.Strings(wantVisited)
	if strings.Join(got, ",") != strings.Join(wantVisited, ",") {
		t.Errorf("visited files mismatch:\n got: %v\nwant: %v", got, wantVisited)
	}
	if got1 := atomic.LoadInt32(&stub.initCalls); got1 != 1 {
		t.Errorf("Init calls: got %d, want 1 (Init runs lazily on first match)", got1)
	}
	if got1 := atomic.LoadInt32(&stub.finalizeCalls); got1 != 1 {
		t.Errorf("Finalize calls: got %d, want 1", got1)
	}
	if len(res.Nodes) != 3 || len(res.Edges) != 3 {
		t.Errorf("expected 3 nodes/3 edges (one per visited file); got nodes=%d edges=%d",
			len(res.Nodes), len(res.Edges))
	}
	// Every edge must carry both provenance tags.
	for _, e := range res.Edges {
		if got1, _ := e.Metadata["provenance"].(string); got1 != "sitter" {
			t.Errorf("edge missing provenance=sitter; metadata=%v", e.Metadata)
		}
		if got1, _ := e.Metadata["sitter_lang"].(string); got1 != "pystub" {
			t.Errorf("edge missing sitter_lang=pystub; metadata=%v", e.Metadata)
		}
	}
}

// TestExtractSitterHonorsIgnore confirms files inside an ignored
// directory (.dexignore-style or .gitignore-style) are skipped. We use
// node_modules — covered by DefaultPatterns in internal/ignore — so we
// don't have to write our own .dexignore.
func TestExtractSitterHonorsIgnore(t *testing.T) {
	root := writeTree(t, map[string]string{
		"keep.py":             "x = 1\n",
		"node_modules/dep.py": "y = 2\n",
		"vendor/sub/old.py":   "z = 3\n",
	})

	stub := &stubExtractor{name: "pystub", extensions: []string{".py"}}
	reg := NewRegistry()
	reg.Register(func() Extractor { return stub })

	if _, err := ExtractSitterWith(context.Background(), root, reg); err != nil {
		t.Fatalf("ExtractSitterWith: %v", err)
	}
	for _, p := range stub.visited {
		if strings.HasPrefix(p, "node_modules/") || strings.HasPrefix(p, "vendor/") {
			t.Errorf("ignored dir %q was visited; ignore filter not applied", p)
		}
	}
	if len(stub.visited) != 1 || stub.visited[0] != "keep.py" {
		t.Errorf("expected only keep.py to be visited, got %v", stub.visited)
	}
}

// TestExtractSitterMultipleExtractors confirms two extractors claiming
// different extensions don't see each other's files and each gets a
// fresh instance per Run.
func TestExtractSitterMultipleExtractors(t *testing.T) {
	root := writeTree(t, map[string]string{
		"a.py": "x = 1\n",
		"b.rs": "fn main() {}\n",
	})

	pyVisited, rsVisited := []string{}, []string{}
	reg := NewRegistry()
	reg.Register(func() Extractor {
		s := &stubExtractor{name: "pystub", extensions: []string{".py"}}
		t.Cleanup(func() { pyVisited = append(pyVisited, s.visited...) })
		return s
	})
	reg.Register(func() Extractor {
		s := &stubExtractor{name: "rsstub", extensions: []string{".rs"}}
		t.Cleanup(func() { rsVisited = append(rsVisited, s.visited...) })
		return s
	})

	res, err := ExtractSitterWith(context.Background(), root, reg)
	if err != nil {
		t.Fatalf("ExtractSitterWith: %v", err)
	}
	// Each extractor emitted one node for its sole file.
	if len(res.Nodes) != 2 {
		t.Errorf("expected 2 nodes (one per extractor's file); got %d", len(res.Nodes))
	}
	// The sitter_lang tag distinguishes the two even though the schema is identical.
	tags := map[string]int{}
	for _, e := range res.Edges {
		if lang, ok := e.Metadata["sitter_lang"].(string); ok {
			tags[lang]++
		}
	}
	if tags["pystub"] != 1 || tags["rsstub"] != 1 {
		t.Errorf("expected one edge per language; got %v", tags)
	}
}

// TestExtractSitterPerRunIsolation ensures factories produce fresh
// instances per Run — a second ExtractSitterWith call must not see
// leaked state from the first. This is what makes concurrent Runs
// (CLI + daemon) safe.
func TestExtractSitterPerRunIsolation(t *testing.T) {
	root := writeTree(t, map[string]string{"a.py": "x = 1\n"})

	built := 0
	reg := NewRegistry()
	reg.Register(func() Extractor {
		built++
		return &stubExtractor{name: "pystub", extensions: []string{".py"}}
	})

	// Register's probe call already created one instance; reset the
	// counter so we measure only Run-time creations.
	built = 0
	for i := 0; i < 3; i++ {
		if _, err := ExtractSitterWith(context.Background(), root, reg); err != nil {
			t.Fatalf("run %d: %v", i, err)
		}
	}
	if built != 3 {
		t.Errorf("expected 3 fresh extractor instances (one per Run); got %d", built)
	}
}

// TestExtractSitterInitErrorDegrades verifies that an Init failure
// degrades to "no extraction for this language" rather than aborting
// the whole sitter pass. The warning surfaces but the pass succeeds.
func TestExtractSitterInitErrorDegrades(t *testing.T) {
	root := writeTree(t, map[string]string{
		"a.py": "x = 1\n",
		"b.py": "y = 2\n",
	})

	stub := &stubExtractor{name: "pystub", extensions: []string{".py"}, failInit: true}
	reg := NewRegistry()
	reg.Register(func() Extractor { return stub })

	res, err := ExtractSitterWith(context.Background(), root, reg)
	if err != nil {
		t.Fatalf("ExtractSitterWith: %v", err)
	}
	if len(stub.visited) != 0 {
		t.Errorf("ProcessFile should not run after Init failure; visited=%v", stub.visited)
	}
	if len(res.Nodes) != 0 || len(res.Edges) != 0 {
		t.Errorf("no graph data should survive Init failure; got nodes=%d edges=%d",
			len(res.Nodes), len(res.Edges))
	}
	if len(res.Warnings) == 0 {
		t.Error("expected an init warning")
	}
}

// TestExtractSitterProcessErrorIsWarning confirms a per-file
// ProcessFile error doesn't abort the walk and surfaces as a warning
// instead. Other files still get processed; Finalize still runs.
func TestExtractSitterProcessErrorIsWarning(t *testing.T) {
	root := writeTree(t, map[string]string{
		"a.py": "x = 1\n",
		"b.py": "y = 2\n",
	})

	stub := &stubExtractor{name: "pystub", extensions: []string{".py"}, failProcess: true}
	reg := NewRegistry()
	reg.Register(func() Extractor { return stub })

	res, err := ExtractSitterWith(context.Background(), root, reg)
	if err != nil {
		t.Fatalf("ExtractSitterWith: %v", err)
	}
	if len(stub.visited) != 2 {
		t.Errorf("walker should visit both files even when ProcessFile errors; visited=%v", stub.visited)
	}
	if got1 := atomic.LoadInt32(&stub.finalizeCalls); got1 != 1 {
		t.Errorf("Finalize must still run after ProcessFile errors; calls=%d", got1)
	}
	if len(res.Warnings) < 2 {
		t.Errorf("expected at least 2 process warnings; got %v", res.Warnings)
	}
}

// TestExtractSitterPreservesExtractorMetadata confirms the framework
// only fills in provenance / sitter_lang when the extractor hasn't
// already set them — extractors that want to override (e.g. a
// "provenance: types" upgrade lane) keep their tag.
func TestExtractSitterPreservesExtractorMetadata(t *testing.T) {
	root := writeTree(t, map[string]string{"a.py": "x = 1\n"})

	stub := &preStampedStub{}
	reg := NewRegistry()
	reg.Register(func() Extractor { return stub })

	res, err := ExtractSitterWith(context.Background(), root, reg)
	if err != nil {
		t.Fatalf("ExtractSitterWith: %v", err)
	}
	if len(res.Edges) != 1 {
		t.Fatalf("expected 1 edge; got %d", len(res.Edges))
	}
	if got1, _ := res.Edges[0].Metadata["provenance"].(string); got1 != "types" {
		t.Errorf("framework overwrote extractor-set provenance; got %q want types", got1)
	}
}

type preStampedStub struct{}

func (preStampedStub) Name() string                                     { return "pre" }
func (preStampedStub) Language() *sitter.Language                       { return python.GetLanguage() }
func (preStampedStub) Extensions() []string                             { return []string{".py"} }
func (preStampedStub) Init(_ context.Context, _ string) error           { return nil }
func (preStampedStub) ProcessFile(_ context.Context, _ FileInput) error { return nil }
func (preStampedStub) Finalize(_ context.Context) ([]Node, []Edge, []string, error) {
	id := NodeID("", "pre", NodeFile, "a.py")
	return nil, []Edge{{
		ID:    EdgeID(id, EdgeContains, id, "a.py", 0),
		Kind:  EdgeContains,
		SrcID: id,
		DstID: id,
		// Pre-set provenance — framework must not overwrite.
		Metadata: map[string]any{"provenance": "types"},
	}}, nil, nil
}
