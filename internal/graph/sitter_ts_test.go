package graph

import (
	"context"
	"testing"
)

// TestTSExtractorFixture exercises the TypeScript extractor against
// testdata/ts_simple. Covers every resolution lane the first cut
// supports: same-file bare call, this.X method dispatch, namespace
// import (`* as text`), named import (`{ Handler }`), const-arrow
// callable (`noop`), `new ClassName()`, and the implicit-index
// specifier resolution (`./utils` → `src/utils/index`).
func TestTSExtractorFixture(t *testing.T) {
	root := copyFixture(t, "ts_simple")
	reg := NewRegistry()
	reg.Register(newTSExtractor)

	res, err := ExtractSitterWith(context.Background(), root, reg)
	if err != nil {
		t.Fatalf("ExtractSitterWith: %v", err)
	}

	// ---- Files ----
	wantFiles := []string{
		"src/main.ts",
		"src/text.ts",
		"src/handler.ts",
		"src/utils/index.ts",
	}
	for _, rel := range wantFiles {
		if findNode(res.Nodes, NodeFile, rel) == nil {
			t.Errorf("missing file node %q; files=%v", rel, nodesOfKind(res.Nodes, NodeFile))
		}
	}

	// ---- Packages (TS package path = file path minus extension) ----
	mainPkg := "src/main"
	textPkg := "src/text"
	handlerPkg := "src/handler"
	utilsPkg := "src/utils/index"

	for _, p := range []string{mainPkg, textPkg, handlerPkg, utilsPkg} {
		if findNode(res.Nodes, NodePackage, p) == nil {
			t.Errorf("missing package node %q; packages=%v", p, nodesOfKind(res.Nodes, NodePackage))
		}
	}

	// ---- Functions ----
	for _, f := range []struct{ pkg, name string }{
		{mainPkg, "main"},
		{mainPkg, "helper"},
		{handlerPkg, "makeHandler"},
		{textPkg, "upper"},
		{textPkg, "lower"},
		{utilsPkg, "id"},
		{utilsPkg, "noop"}, // arrow-const must be promoted to NodeFunction
	} {
		id := NodeID("", f.pkg, NodeFunction, f.name)
		if findByID(res.Nodes, id) == nil {
			t.Errorf("missing function %s::%s; functions=%v",
				f.pkg, f.name, nodesOfKindWithPkg(res.Nodes, NodeFunction))
		}
	}

	// ---- Class + methods ----
	handlerClsID := NodeID("", handlerPkg, NodeClass, "Handler")
	if findByID(res.Nodes, handlerClsID) == nil {
		t.Fatalf("missing class Handler in %s", handlerPkg)
	}
	for _, m := range []string{"constructor", "greet", "format"} {
		mid := NodeID("", handlerPkg, NodeMethod, "Handler."+m)
		if findByID(res.Nodes, mid) == nil {
			t.Errorf("missing method Handler.%s; methods=%v", m, nodesOfKindWithPkg(res.Nodes, NodeMethod))
		}
	}

	// ---- Interface ----
	if findByID(res.Nodes, NodeID("", mainPkg, NodeInterface, "MainConfig")) == nil {
		t.Errorf("missing interface MainConfig; interfaces=%v", nodesOfKindWithPkg(res.Nodes, NodeInterface))
	}

	// ---- has_method edge ----
	greetID := NodeID("", handlerPkg, NodeMethod, "Handler.greet")
	if findEdge(res.Edges, EdgeHasMethod, handlerClsID, greetID) == nil {
		t.Errorf("missing has_method Handler → Handler.greet")
	}

	// ---- Package-level imports edge (graph_deps relies on this) ----
	mainPkgID := NodeID("", mainPkg, NodePackage, mainPkg)
	// Import edges are emitted twice: once at file level (with
	// resolved specifier), once at package level. The dst node ID
	// uses the raw specifier so cross-package graph deps surface the
	// import path the user actually wrote.
	for _, spec := range []string{"./handler", "./text", "./utils"} {
		impID := NodeID("", mainPkg, NodeImport, spec)
		if findEdge(res.Edges, EdgeImports, mainPkgID, impID) == nil {
			t.Errorf("missing package-level imports edge %s → %s", mainPkg, spec)
		}
	}

	// ---- Calls ----
	mainFnID := NodeID("", mainPkg, NodeFunction, "main")
	helperFnID := NodeID("", mainPkg, NodeFunction, "helper")
	makeHandlerID := NodeID("", handlerPkg, NodeFunction, "makeHandler")
	upperFnID := NodeID("", textPkg, NodeFunction, "upper")
	noopID := NodeID("", utilsPkg, NodeFunction, "noop")
	formatID := NodeID("", handlerPkg, NodeMethod, "Handler.format")

	calls := []struct {
		name     string
		src, dst string
		why      string
	}{
		{"main → helper (same-file bare)", mainFnID, helperFnID, "same-file bare call"},
		{"main → makeHandler (named import)", mainFnID, makeHandlerID, "import { makeHandler } resolves the bare call cross-file"},
		{"main → Handler (new + named import)", mainFnID, handlerClsID, "new ClassName() resolves to the class"},
		{"main → text.upper (namespace import)", mainFnID, upperFnID, "import * as text → text.upper() walks the namespace"},
		{"main → noop (named import to arrow-const)", mainFnID, noopID, "arrow-const noop must be reachable as a callee via named import"},
		{"makeHandler → Handler (new same-file)", makeHandlerID, handlerClsID, "new Handler() inside makeHandler resolves to the local class"},
		{"Handler.greet → Handler.format (this.X)", greetID, formatID, "this.X inside a method resolves to same class"},
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
		if got, _ := e.Metadata["sitter_lang"].(string); got != "typescript" {
			t.Errorf("edge missing sitter_lang=typescript; %+v", e)
		}
	}
}
