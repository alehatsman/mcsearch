package graph

import (
	"context"
	"testing"
)

// TestPythonExtractorFixture exercises the Python extractor against
// testdata/python_simple. The fixture deliberately covers every
// resolution lane: same-package bare call, self.X method dispatch,
// cross-package call via `from X import Y`, cross-package chain via
// `import X.Y` then `X.Y.f()`, and class instantiation as a call.
//
// Uses ExtractSitterWith with a private registry holding only the
// Python factory, so the test is isolated from any other extractor a
// future commit might add to DefaultRegistry.
func TestPythonExtractorFixture(t *testing.T) {
	root := copyFixture(t, "python_simple")
	reg := NewRegistry()
	reg.Register(newPythonExtractor)

	res, err := ExtractSitterWith(context.Background(), root, reg)
	if err != nil {
		t.Fatalf("ExtractSitterWith: %v", err)
	}

	// ---- Files ----
	wantFiles := []string{
		"app/main.py",
		"app/handler.py",
		"app/__init__.py",
		"utils/text.py",
		"utils/__init__.py",
	}
	for _, rel := range wantFiles {
		if findNode(res.Nodes, NodeFile, rel) == nil {
			t.Errorf("missing file node %q; files=%v", rel, nodesOfKind(res.Nodes, NodeFile))
		}
	}

	// ---- Packages — load-bearing for graph_deps + guide rendering ----
	wantPkgs := []string{"app", "app.main", "app.handler", "utils", "utils.text"}
	for _, p := range wantPkgs {
		if findNode(res.Nodes, NodePackage, p) == nil {
			t.Errorf("missing package node %q; packages=%v", p, nodesOfKind(res.Nodes, NodePackage))
		}
	}
	// Package → file contains: app.main → app/main.py.
	pkgMainID := NodeID("", "app.main", NodePackage, "app.main")
	mainFileNodeID := NodeID("", "app.main", NodeFile, "app/main.py")
	if findEdge(res.Edges, EdgeContains, pkgMainID, mainFileNodeID) == nil {
		t.Errorf("missing contains edge app.main package → app/main.py")
	}
	// Package-level imports edge for graph_deps: app.main pkg → utils.text import.
	utilsTextImpFromMain := NodeID("", "app.main", NodeImport, "utils.text")
	if findEdge(res.Edges, EdgeImports, pkgMainID, utilsTextImpFromMain) == nil {
		t.Errorf("missing package-level imports edge app.main → utils.text (needed by graph_deps)")
	}

	// ---- Functions (top-level, with their package paths) ----
	mainPkg := "app.main"
	handlerPkg := "app.handler"
	textPkg := "utils.text"

	for _, f := range []struct{ pkg, name string }{
		{mainPkg, "main"},
		{mainPkg, "helper"},
		{handlerPkg, "make_handler"},
		{textPkg, "upper"},
		{textPkg, "lower"},
	} {
		id := NodeID("", f.pkg, NodeFunction, f.name)
		if findByID(res.Nodes, id) == nil {
			t.Errorf("missing function %s::%s; functions=%v",
				f.pkg, f.name, nodesOfKindWithPkg(res.Nodes, NodeFunction))
		}
	}

	// ---- Class + its methods ----
	handlerClsID := NodeID("", handlerPkg, NodeClass, "Handler")
	if findByID(res.Nodes, handlerClsID) == nil {
		t.Fatalf("missing class node Handler in app.handler")
	}
	for _, m := range []string{"__init__", "greet", "format"} {
		mid := NodeID("", handlerPkg, NodeMethod, "Handler."+m)
		if findByID(res.Nodes, mid) == nil {
			t.Errorf("missing method Handler.%s; methods=%v", m,
				nodesOfKindWithPkg(res.Nodes, NodeMethod))
		}
	}

	// ---- Edges ----

	// has_method: Handler → Handler.greet
	greetID := NodeID("", handlerPkg, NodeMethod, "Handler.greet")
	if findEdge(res.Edges, EdgeHasMethod, handlerClsID, greetID) == nil {
		t.Errorf("missing has_method Handler → Handler.greet")
	}

	// imports edge: app/main.py → utils.text  (via `import utils.text`)
	mainFileID := NodeID("", mainPkg, NodeFile, "app/main.py")
	utilsTextImpID := NodeID("", mainPkg, NodeImport, "utils.text")
	if findEdge(res.Edges, EdgeImports, mainFileID, utilsTextImpID) == nil {
		t.Errorf("missing imports edge app/main.py → utils.text")
	}

	// imports edge: app/main.py → app.handler (via `from app.handler import …`)
	appHandlerImpID := NodeID("", mainPkg, NodeImport, "app.handler")
	if findEdge(res.Edges, EdgeImports, mainFileID, appHandlerImpID) == nil {
		t.Errorf("missing imports edge app/main.py → app.handler")
	}

	// ---- Calls ----
	mainFnID := NodeID("", mainPkg, NodeFunction, "main")
	helperFnID := NodeID("", mainPkg, NodeFunction, "helper")
	makeHandlerID := NodeID("", handlerPkg, NodeFunction, "make_handler")
	upperFnID := NodeID("", textPkg, NodeFunction, "upper")
	formatID := NodeID("", handlerPkg, NodeMethod, "Handler.format")

	calls := []struct {
		name     string
		src, dst string
		why      string
	}{
		{"main → helper (bare same-package)", mainFnID, helperFnID, "same-package bare call should resolve"},
		{"main → make_handler (from-import)", mainFnID, makeHandlerID, "from app.handler import make_handler → bare call should resolve cross-package"},
		{"main → Handler (class as callable, from-import)", mainFnID, handlerClsID, "constructor calls via from-import resolve to the class node"},
		{"main → utils.text.upper (import chain)", mainFnID, upperFnID, "`import utils.text` + `utils.text.upper()` walks the chain"},
		{"Handler.greet → Handler.format (self.X)", greetID, formatID, "self.X inside a method resolves to that method's class"},
	}
	for _, c := range calls {
		t.Run(c.name, func(t *testing.T) {
			if findEdge(res.Edges, EdgeCalls, c.src, c.dst) == nil {
				t.Errorf("missing calls edge %s\n  why: %s\n  all calls=%v",
					c.name, c.why, edgeKinds(res.Edges, EdgeCalls))
			}
		})
	}

	// ---- Provenance stamping ----
	for _, e := range res.Edges {
		if got, _ := e.Metadata["provenance"].(string); got != "sitter" {
			t.Errorf("edge missing provenance=sitter; %+v", e)
		}
		if got, _ := e.Metadata["sitter_lang"].(string); got != "python" {
			t.Errorf("edge missing sitter_lang=python; %+v", e)
		}
	}
}

// TestPythonRelativeImport drives only the relative-import lane so a
// regression there doesn't get buried in the broader fixture test.
func TestPythonRelativeImport(t *testing.T) {
	root := copyFixture(t, "python_simple")
	reg := NewRegistry()
	reg.Register(newPythonExtractor)

	res, err := ExtractSitterWith(context.Background(), root, reg)
	if err != nil {
		t.Fatalf("ExtractSitterWith: %v", err)
	}

	// utils/__init__.py imports `from .text import upper` — relative
	// resolves to package "utils", appended ".text" → module "utils.text".
	utilsInitID := NodeID("", "utils", NodeFile, "utils/__init__.py")
	impID := NodeID("", "utils", NodeImport, "utils.text")
	if findEdge(res.Edges, EdgeImports, utilsInitID, impID) == nil {
		t.Errorf("missing relative imports edge utils/__init__.py → utils.text; imports=%v",
			edgeKinds(res.Edges, EdgeImports))
	}
}

// findByID locates a node by ID. Tests use it when they care about
// existence at a specific (pkg, kind, qn) coordinate rather than just
// matching by QN.
func findByID(nodes []Node, id string) *Node {
	for i := range nodes {
		if nodes[i].ID == id {
			return &nodes[i]
		}
	}
	return nil
}

// nodesOfKindWithPkg is a debug helper — like nodesOfKind but
// includes the package path so the test failure is actually
// debuggable when QNs collide across packages.
func nodesOfKindWithPkg(nodes []Node, kind NodeKind) []string {
	var out []string
	for _, n := range nodes {
		if n.Kind == kind {
			out = append(out, n.PackagePath+"::"+n.QualifiedName)
		}
	}
	return out
}
