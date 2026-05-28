package graph

import (
	"context"
	"testing"
)

// TestRustExtractorFixture exercises the Rust extractor against
// testdata/rust_simple. Covers same-file bare, self.X / Self::,
// `use path::Item` bare resolution, `Type::method()` scoped, plus
// trait declaration + implements edge from impl-for.
func TestRustExtractorFixture(t *testing.T) {
	root := copyFixture(t, "rust_simple")
	reg := NewRegistry()
	reg.Register(newRustExtractor)

	res, err := ExtractSitterWith(context.Background(), root, reg)
	if err != nil {
		t.Fatalf("ExtractSitterWith: %v", err)
	}

	// ---- Files ----
	wantFiles := []string{
		"src/main.rs",
		"src/text.rs",
		"src/handler.rs",
	}
	for _, rel := range wantFiles {
		if findNode(res.Nodes, NodeFile, rel) == nil {
			t.Errorf("missing file node %q; files=%v", rel, nodesOfKind(res.Nodes, NodeFile))
		}
	}

	// ---- Packages ----
	mainPkg := "main"
	textPkg := "text"
	handlerPkg := "handler"
	for _, p := range []string{mainPkg, textPkg, handlerPkg} {
		if findNode(res.Nodes, NodePackage, p) == nil {
			t.Errorf("missing package node %q; packages=%v", p, nodesOfKind(res.Nodes, NodePackage))
		}
	}

	// ---- Functions ----
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

	// ---- Struct (NodeClass) + methods (from inherent impl block) ----
	handlerClsID := NodeID("", handlerPkg, NodeClass, "Handler")
	if findByID(res.Nodes, handlerClsID) == nil {
		t.Fatalf("missing struct Handler in %s", handlerPkg)
	}
	for _, m := range []string{"new", "format", "greet"} {
		mid := NodeID("", handlerPkg, NodeMethod, "Handler."+m)
		if findByID(res.Nodes, mid) == nil {
			t.Errorf("missing method Handler.%s; methods=%v", m, nodesOfKindWithPkg(res.Nodes, NodeMethod))
		}
	}

	// ---- Trait + implements edge ----
	traitID := NodeID("", handlerPkg, NodeInterface, "Greeter")
	if findByID(res.Nodes, traitID) == nil {
		t.Errorf("missing trait Greeter; interfaces=%v", nodesOfKindWithPkg(res.Nodes, NodeInterface))
	}
	if findEdge(res.Edges, EdgeImplements, handlerClsID, traitID) == nil {
		t.Errorf("missing implements edge Handler → Greeter")
	}

	// ---- Package-level imports (graph_deps lane) ----
	mainPkgID := NodeID("", mainPkg, NodePackage, mainPkg)
	for _, target := range []string{"handler::Handler", "handler::make_handler", "text::upper", "handler", "text"} {
		impID := NodeID("", mainPkg, NodeImport, target)
		if findEdge(res.Edges, EdgeImports, mainPkgID, impID) == nil {
			t.Errorf("missing package-level imports edge %s → %s", mainPkg, target)
		}
	}

	// ---- Calls ----
	mainFnID := NodeID("", mainPkg, NodeFunction, "main")
	helperFnID := NodeID("", mainPkg, NodeFunction, "helper")
	makeHandlerID := NodeID("", handlerPkg, NodeFunction, "make_handler")
	upperFnID := NodeID("", textPkg, NodeFunction, "upper")
	formatID := NodeID("", handlerPkg, NodeMethod, "Handler.format")
	newID := NodeID("", handlerPkg, NodeMethod, "Handler.new")
	greetID := NodeID("", handlerPkg, NodeMethod, "Handler.greet")

	calls := []struct {
		name     string
		src, dst string
		why      string
	}{
		{"main → helper (same-file bare)", mainFnID, helperFnID, "same-file bare call"},
		{"main → make_handler (use-bare)", mainFnID, makeHandlerID, "`use ::make_handler` brings free fn into scope; bare call resolves"},
		{"main → upper (use-bare)", mainFnID, upperFnID, "`use crate::text::upper` resolves bare upper(...)"},
		{"main → Handler.new (Type::method)", mainFnID, newID, "Handler::new(...) — scoped call resolves to inherent-impl method via import table"},
		{"Handler.greet → Handler.format (self.X)", greetID, formatID, "self.format(...) in trait impl resolves to inherent-impl method on Handler"},
		{"make_handler → Handler.new (same-file Type::method)", makeHandlerID, newID, "same-file Handler::new — symbol lookup without import"},
	}
	for _, c := range calls {
		t.Run(c.name, func(t *testing.T) {
			if findEdge(res.Edges, EdgeCalls, c.src, c.dst) == nil {
				t.Errorf("missing calls edge %s\n  why: %s\n  all calls=%v",
					c.name, c.why, edgeKinds(res.Edges, EdgeCalls))
			}
		})
	}

	// ---- Provenance ----
	for _, e := range res.Edges {
		if got, _ := e.Metadata["provenance"].(string); got != "sitter" {
			t.Errorf("edge missing provenance=sitter; %+v", e)
		}
		if got, _ := e.Metadata["sitter_lang"].(string); got != "rust" {
			t.Errorf("edge missing sitter_lang=rust; %+v", e)
		}
	}
}
