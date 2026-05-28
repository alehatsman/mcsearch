package graph

// Rust tree-sitter extractor. Emits package / file / function / class
// (struct/enum) / interface (trait) / method / import nodes and
// contains / has_method / imports / calls edges for every .rs file
// under projectRoot.
//
// Package-path encoding: Rust module path in `::` form, derived from
// the file path with a few normalizations:
//   src/lib.rs              → "lib"
//   src/main.rs             → "main"
//   src/foo.rs              → "foo"
//   src/foo/mod.rs          → "foo"   (mod.rs collapses to dir name)
//   src/foo/bar.rs          → "foo::bar"
//   tests/integration.rs    → "tests::integration"
//   not-under-src/x.rs      → "not-under-src::x"
//
// This is enough for Cargo-shaped projects, which cover the realistic
// majority. Manual `#[path = "..."]` overrides, build.rs codegen, and
// multi-crate workspaces collapse imperfectly — first-cut accepts the
// loss in exchange for not parsing Cargo.toml.
//
// Resolution lanes:
//   - same-file bare call           — symbols table lookup
//   - `Self::method()` / `self.x()` — receiver type's method
//   - `Type::method()`              — method on a same-file type, OR
//                                     a `use path::To::Type` import
//   - `mod::func()`                 — `use path::mod` aliases the path;
//                                     unaliased modules reach by full path
//   - `use foo::Bar` then `Bar()`   — class-as-callable + new resolves
//                                     to the imported type
//   - `Trait` / glob imports / `super::` paths — out of scope first cut.
//
// `use` paths starting with `crate::` are normalized to project-relative
// module paths (the leading `crate::` is stripped). `self::` and
// `super::` paths are left raw — the resolver simply won't find them
// in the symbol table, which is the right failure mode.

import (
	"context"
	"path/filepath"
	"strings"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/rust"
)

func init() { Register(newRustExtractor) }

func newRustExtractor() Extractor {
	return &rustExtractor{
		nodeIDs:     map[string]struct{}{},
		symbols:     map[string]map[string]string{},
		fileImports: map[string]*rustImportTable{},
	}
}

type rustExtractor struct {
	projectRoot string

	nodes []Node
	edges []Edge

	nodeIDs map[string]struct{}
	// symbols: packagePath → name → nodeID. Methods are keyed as
	// `ReceiverType.method` so a Type::method call site is one lookup.
	symbols map[string]map[string]string

	// fileImports keyed by file relpath (slash form).
	fileImports map[string]*rustImportTable

	pendingCalls []rustPendingCall

	warnings []string
}

// rustImportTable: each local name (the rightmost segment of a `use`
// path, or an alias) maps to (modulePath, originalName). The
// modulePath is in `::` form, matching the packagePath we synthesize
// from file paths.
type rustImportTable struct {
	fromImports map[string]rustImport
}

type rustImport struct {
	modulePath string // crate-relative path in `::` form
	name       string // original name on the other side
}

type rustPendingCall struct {
	callerID   string
	callerPkg  string
	callerRecv string // receiver type when caller is a method; "" otherwise
	calleeExpr rustCallee
	filePath   string
	line       int
}

// rustCallee variants — like pyCallee but with two extra shapes:
//
//	"bare"      → identifier (foo)
//	"scoped"    → scoped_identifier path (A::B::c)
//	"self"      → self.X / Self::X
//	"method"    → x.method() with x not `self`
//	"skip"      → can't resolve syntactically
type rustCallee struct {
	kind  string
	parts []string
}

// ---- Extractor interface ---------------------------------------------------

func (e *rustExtractor) Name() string               { return "rust" }
func (e *rustExtractor) Language() *sitter.Language { return rust.GetLanguage() }
func (e *rustExtractor) Extensions() []string       { return []string{".rs"} }

func (e *rustExtractor) Init(_ context.Context, root string) error {
	e.projectRoot = root
	return nil
}

func (e *rustExtractor) ProcessFile(_ context.Context, in FileInput) error {
	pkg := rustPackagePath(in.RelPath)

	pkgID := NodeID("", pkg, NodePackage, pkg)
	e.addNode(Node{
		ID:            pkgID,
		Kind:          NodePackage,
		Name:          rustLeafName(pkg),
		QualifiedName: pkg,
		PackagePath:   pkg,
		Metadata:      map[string]any{"language": "rust"},
	})

	fileID := NodeID("", pkg, NodeFile, in.RelPath)
	e.addNode(Node{
		ID:            fileID,
		Kind:          NodeFile,
		Name:          filepath.Base(in.RelPath),
		QualifiedName: in.RelPath,
		PackagePath:   pkg,
		FilePath:      in.RelPath,
		StartLine:     1,
		EndLine:       lineOfPoint(in.Root.EndPoint().Row),
		Metadata:      map[string]any{"language": "rust"},
	})
	e.edges = append(e.edges, Edge{
		ID:        EdgeID(pkgID, EdgeContains, fileID, in.RelPath, 1),
		Kind:      EdgeContains,
		SrcID:     pkgID,
		DstID:     fileID,
		FilePath:  in.RelPath,
		StartLine: 1,
		EndLine:   1,
	})

	imports := &rustImportTable{fromImports: map[string]rustImport{}}
	e.fileImports[in.RelPath] = imports

	for i := 0; i < int(in.Root.NamedChildCount()); i++ {
		e.processTopLevel(in.Root.NamedChild(i), in.Source, in.RelPath, pkg, fileID, imports)
	}
	return nil
}

func (e *rustExtractor) Finalize(_ context.Context) ([]Node, []Edge, []string, error) {
	for _, c := range e.pendingCalls {
		dst := e.resolveCall(c)
		if dst == "" {
			continue
		}
		e.edges = append(e.edges, Edge{
			ID:        EdgeID(c.callerID, EdgeCalls, dst, c.filePath, c.line),
			Kind:      EdgeCalls,
			SrcID:     c.callerID,
			DstID:     dst,
			FilePath:  c.filePath,
			StartLine: c.line,
			EndLine:   c.line,
		})
	}
	return e.nodes, e.edges, e.warnings, nil
}

// ---- top-level processing --------------------------------------------------

func (e *rustExtractor) processTopLevel(
	n *sitter.Node, src []byte,
	filePath, pkg, fileID string, imports *rustImportTable,
) {
	if n == nil {
		return
	}
	switch n.Type() {
	case "function_item":
		e.addFunction(n, src, filePath, pkg, fileID, "")
	case "struct_item", "enum_item":
		e.addType(n, src, filePath, pkg, fileID, NodeClass)
	case "trait_item":
		e.addTrait(n, src, filePath, pkg, fileID)
	case "impl_item":
		e.addImpl(n, src, filePath, pkg, fileID)
	case "use_declaration":
		e.parseUseDecl(n, src, filePath, pkg, fileID, imports)
	case "mod_item":
		// Inline submodule `mod foo { ... }` — first-cut skips. The
		// declaration form `mod foo;` does emit an import edge so
		// graph_deps still sees the dependency.
		if n.ChildByFieldName("body") == nil {
			e.emitModDecl(n, src, filePath, pkg, fileID)
		}
	}
}

// addFunction registers a top-level fn or a method inside an impl.
// receiverType is the type name when this fn lives inside an impl
// block; "" for free functions.
func (e *rustExtractor) addFunction(
	n *sitter.Node, src []byte,
	filePath, pkg, fileID, receiverType string,
) {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		return
	}
	name := nodeText(nameNode, src)
	if name == "" {
		return
	}
	kind := NodeFunction
	qn := name
	if receiverType != "" {
		kind = NodeMethod
		qn = receiverType + "." + name
	}
	id := NodeID("", pkg, kind, qn)
	startLine := lineOfPoint(n.StartPoint().Row)
	endLine := lineOfPoint(n.EndPoint().Row)
	meta := map[string]any{"language": "rust"}
	if receiverType != "" {
		meta["receiver"] = receiverType
	}
	if e.addNode(Node{
		ID:            id,
		Kind:          kind,
		Name:          name,
		QualifiedName: qn,
		PackagePath:   pkg,
		FilePath:      filePath,
		StartLine:     startLine,
		EndLine:       endLine,
		Metadata:      meta,
	}) {
		e.symbols[pkg] = ensureMap(e.symbols[pkg])
		e.symbols[pkg][qn] = id
		if receiverType == "" {
			e.symbols[pkg][name] = id
		}
	}
	e.edges = append(e.edges, Edge{
		ID:        EdgeID(fileID, EdgeContains, id, filePath, startLine),
		Kind:      EdgeContains,
		SrcID:     fileID,
		DstID:     id,
		FilePath:  filePath,
		StartLine: startLine,
		EndLine:   endLine,
	})
	if receiverType != "" {
		recvID := NodeID("", pkg, NodeClass, receiverType)
		e.edges = append(e.edges, Edge{
			ID:        EdgeID(recvID, EdgeHasMethod, id, filePath, startLine),
			Kind:      EdgeHasMethod,
			SrcID:     recvID,
			DstID:     id,
			FilePath:  filePath,
			StartLine: startLine,
			EndLine:   endLine,
		})
	}
	if body := n.ChildByFieldName("body"); body != nil {
		e.collectCalls(body, src, id, pkg, receiverType, filePath)
	}
}

// addType handles struct_item and enum_item — both surface as
// NodeClass since Go's struct/interface distinction doesn't map
// cleanly. Field/variant extraction is out of scope first cut.
func (e *rustExtractor) addType(
	n *sitter.Node, src []byte,
	filePath, pkg, fileID string, kind NodeKind,
) {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		return
	}
	name := nodeText(nameNode, src)
	if name == "" {
		return
	}
	id := NodeID("", pkg, kind, name)
	startLine := lineOfPoint(n.StartPoint().Row)
	endLine := lineOfPoint(n.EndPoint().Row)
	meta := map[string]any{"language": "rust"}
	if n.Type() == "enum_item" {
		meta["form"] = "enum"
	}
	if e.addNode(Node{
		ID:            id,
		Kind:          kind,
		Name:          name,
		QualifiedName: name,
		PackagePath:   pkg,
		FilePath:      filePath,
		StartLine:     startLine,
		EndLine:       endLine,
		Metadata:      meta,
	}) {
		e.symbols[pkg] = ensureMap(e.symbols[pkg])
		e.symbols[pkg][name] = id
	}
	e.edges = append(e.edges, Edge{
		ID:        EdgeID(fileID, EdgeContains, id, filePath, startLine),
		Kind:      EdgeContains,
		SrcID:     fileID,
		DstID:     id,
		FilePath:  filePath,
		StartLine: startLine,
		EndLine:   endLine,
	})
}

// addTrait: NodeInterface so consumers can distinguish from
// struct/enum. Trait methods are walked just like an impl's methods
// but tagged on_interface in metadata.
func (e *rustExtractor) addTrait(
	n *sitter.Node, src []byte,
	filePath, pkg, fileID string,
) {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		return
	}
	name := nodeText(nameNode, src)
	if name == "" {
		return
	}
	id := NodeID("", pkg, NodeInterface, name)
	startLine := lineOfPoint(n.StartPoint().Row)
	endLine := lineOfPoint(n.EndPoint().Row)
	if e.addNode(Node{
		ID:            id,
		Kind:          NodeInterface,
		Name:          name,
		QualifiedName: name,
		PackagePath:   pkg,
		FilePath:      filePath,
		StartLine:     startLine,
		EndLine:       endLine,
		Metadata:      map[string]any{"language": "rust"},
	}) {
		e.symbols[pkg] = ensureMap(e.symbols[pkg])
		e.symbols[pkg][name] = id
	}
	e.edges = append(e.edges, Edge{
		ID:        EdgeID(fileID, EdgeContains, id, filePath, startLine),
		Kind:      EdgeContains,
		SrcID:     fileID,
		DstID:     id,
		FilePath:  filePath,
		StartLine: startLine,
		EndLine:   endLine,
	})

	body := n.ChildByFieldName("body")
	if body == nil {
		return
	}
	for i := 0; i < int(body.NamedChildCount()); i++ {
		child := body.NamedChild(i)
		if child == nil || child.Type() != "function_item" {
			continue
		}
		// Methods inside a trait are signatures; some have bodies
		// (default impls), some don't. addFunction handles both:
		// it'll walk the body when present and skip when absent.
		e.addFunction(child, src, filePath, pkg, fileID, name)
	}
}

// addImpl walks an impl_item — extract receiver type name, then emit
// each function_item child as a method on that type. The trait form
// `impl Trait for Type` also emits an implements edge.
func (e *rustExtractor) addImpl(
	n *sitter.Node, src []byte,
	filePath, pkg, fileID string,
) {
	typeNode := n.ChildByFieldName("type")
	if typeNode == nil {
		return
	}
	receiverType := rustTypeName(typeNode, src)
	if receiverType == "" {
		return
	}

	// `impl Trait for Type` carries a trait field — emit implements
	// edge so trait → impl is queryable.
	if traitNode := n.ChildByFieldName("trait"); traitNode != nil {
		traitName := rustTypeName(traitNode, src)
		if traitName != "" {
			recvID := NodeID("", pkg, NodeClass, receiverType)
			traitID := NodeID("", pkg, NodeInterface, traitName)
			e.edges = append(e.edges, Edge{
				ID:    EdgeID(recvID, EdgeImplements, traitID, "", 0),
				Kind:  EdgeImplements,
				SrcID: recvID,
				DstID: traitID,
			})
		}
	}

	body := n.ChildByFieldName("body")
	if body == nil {
		return
	}
	for i := 0; i < int(body.NamedChildCount()); i++ {
		child := body.NamedChild(i)
		if child == nil || child.Type() != "function_item" {
			continue
		}
		e.addFunction(child, src, filePath, pkg, fileID, receiverType)
	}
}

// rustTypeName extracts a usable receiver-type name from an impl's
// `type` or `trait` field. Handles: simple identifier, scoped
// identifier (last component wins), generic type (drop the params).
func rustTypeName(n *sitter.Node, src []byte) string {
	switch n.Type() {
	case "type_identifier":
		return nodeText(n, src)
	case "scoped_type_identifier":
		// path::Last — use the last component as the local name.
		last := lastNamedChild(n)
		if last != nil {
			return nodeText(last, src)
		}
	case "generic_type":
		// Type<T> — child by field "type" gives the base.
		if base := n.ChildByFieldName("type"); base != nil {
			return rustTypeName(base, src)
		}
	}
	return ""
}

// ---- mod + use parsing -----------------------------------------------------

// emitModDecl emits an import node + edge for a `mod foo;` declaration
// (no body). graph_deps then surfaces the dependency on the sibling
// file or submodule directory.
func (e *rustExtractor) emitModDecl(
	n *sitter.Node, src []byte,
	filePath, currentPkg, fileID string,
) {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		return
	}
	name := nodeText(nameNode, src)
	if name == "" {
		return
	}
	// `mod foo;` in pkg "p" pulls in "p::foo". For the crate root
	// (pkg "lib" or "main"), it pulls in just "foo".
	target := name
	if currentPkg != "" && currentPkg != "lib" && currentPkg != "main" {
		target = currentPkg + "::" + name
	}
	e.emitImportEdges(target, filePath, currentPkg, fileID, lineOfPoint(n.StartPoint().Row))
}

// parseUseDecl handles all `use` shapes:
//
//	use foo::bar::Baz;
//	use foo::bar::{Baz, Qux};
//	use foo::bar::Baz as B;
//	use foo::bar::*;       (glob — emit import edge, no binding)
//	use crate::foo::Bar;   (strip "crate::" prefix)
//
// Items resolved to (packagePath, originalName, alias) get registered
// in the file's import table. Glob and `self::` / `super::` paths
// emit only the import edge.
func (e *rustExtractor) parseUseDecl(
	n *sitter.Node, src []byte,
	filePath, currentPkg, fileID string,
	imports *rustImportTable,
) {
	arg := n.ChildByFieldName("argument")
	if arg == nil {
		return
	}
	startLine := lineOfPoint(n.StartPoint().Row)
	e.expandUseTree(arg, src, "", filePath, currentPkg, fileID, startLine, imports)
}

// expandUseTree walks the tree of use_declaration arguments —
// possibly nested (`use a::{b::{c, d}, e}`). prefix accumulates the
// dotted path collected so far in `::` form (no trailing `::`).
func (e *rustExtractor) expandUseTree(
	n *sitter.Node, src []byte,
	prefix, filePath, currentPkg, fileID string,
	startLine int,
	imports *rustImportTable,
) {
	switch n.Type() {
	case "scoped_use_list":
		// path::{...} — the path is the new prefix; iterate the list.
		pathNode := n.ChildByFieldName("path")
		listNode := n.ChildByFieldName("list")
		head := nodeText(pathNode, src)
		newPrefix := joinPrefix(prefix, head)
		if listNode != nil {
			for i := 0; i < int(listNode.NamedChildCount()); i++ {
				e.expandUseTree(listNode.NamedChild(i), src, newPrefix, filePath, currentPkg, fileID, startLine, imports)
			}
		}
	case "use_list":
		for i := 0; i < int(n.NamedChildCount()); i++ {
			e.expandUseTree(n.NamedChild(i), src, prefix, filePath, currentPkg, fileID, startLine, imports)
		}
	case "use_as_clause":
		pathNode := n.ChildByFieldName("path")
		aliasNode := n.ChildByFieldName("alias")
		if pathNode == nil || aliasNode == nil {
			return
		}
		full := joinPrefix(prefix, nodeText(pathNode, src))
		alias := nodeText(aliasNode, src)
		modPath, name := splitLastSegment(full)
		modPath = normalizeUsePath(modPath)
		if alias != "" && name != "" {
			imports.fromImports[alias] = rustImport{modulePath: modPath, name: name}
		}
		e.emitImportEdges(full, filePath, currentPkg, fileID, startLine)
	case "use_wildcard":
		// path::*  — emit only the import edge; glob bindings aren't
		// resolved to specific symbols.
		pathNode := lastNamedChild(n)
		full := joinPrefix(prefix, nodeText(pathNode, src))
		e.emitImportEdges(full+"::*", filePath, currentPkg, fileID, startLine)
	case "scoped_identifier":
		// Terminal: foo::bar::Baz. Last segment is the binding's
		// local name and original name; everything before is the
		// modulePath. Combine with the accumulated prefix.
		full := joinPrefix(prefix, nodeText(n, src))
		modPath, name := splitLastSegment(full)
		modPath = normalizeUsePath(modPath)
		if name != "" {
			imports.fromImports[name] = rustImport{modulePath: modPath, name: name}
		}
		e.emitImportEdges(full, filePath, currentPkg, fileID, startLine)
	case "identifier":
		// `use foo;` — single-name use; treat foo as a binding to a
		// crate-root module foo (no prefix). modulePath stays empty;
		// the binding maps a module name to itself.
		name := nodeText(n, src)
		full := joinPrefix(prefix, name)
		if name != "" {
			modPath, leaf := splitLastSegment(full)
			modPath = normalizeUsePath(modPath)
			imports.fromImports[name] = rustImport{modulePath: modPath, name: leaf}
		}
		e.emitImportEdges(full, filePath, currentPkg, fileID, startLine)
	}
}

// emitImportEdges adds the NodeImport + file-level and package-level
// imports edges for one resolved use-path. Multiple use-decls
// targeting the same path collide on NodeID, which is the desired
// dedupe. usePath is normalized (leading `crate::` stripped) so the
// emitted target matches packagePaths a consumer would synthesize
// from file layout.
func (e *rustExtractor) emitImportEdges(
	usePath, filePath, currentPkg, fileID string,
	startLine int,
) {
	usePath = normalizeUsePath(usePath)
	if usePath == "" {
		return
	}
	impID := NodeID("", currentPkg, NodeImport, usePath)
	e.addNode(Node{
		ID:            impID,
		Kind:          NodeImport,
		Name:          usePath,
		QualifiedName: usePath,
		PackagePath:   currentPkg,
		Metadata:      map[string]any{"language": "rust"},
	})
	e.edges = append(e.edges, Edge{
		ID:        EdgeID(fileID, EdgeImports, impID, filePath, startLine),
		Kind:      EdgeImports,
		SrcID:     fileID,
		DstID:     impID,
		FilePath:  filePath,
		StartLine: startLine,
		EndLine:   startLine,
	})
	pkgID := NodeID("", currentPkg, NodePackage, currentPkg)
	e.edges = append(e.edges, Edge{
		ID:    EdgeID(pkgID, EdgeImports, impID, "", 0),
		Kind:  EdgeImports,
		SrcID: pkgID,
		DstID: impID,
	})
}

// ---- call collection + resolution ------------------------------------------

func (e *rustExtractor) collectCalls(
	body *sitter.Node, src []byte,
	callerID, callerPkg, callerRecv, filePath string,
) {
	var walk func(*sitter.Node)
	walk = func(n *sitter.Node) {
		if n == nil {
			return
		}
		switch n.Type() {
		// Don't attribute calls inside nested items to the outer fn.
		case "function_item", "struct_item", "enum_item",
			"trait_item", "impl_item", "closure_expression":
			return
		case "call_expression":
			fn := n.ChildByFieldName("function")
			if fn != nil {
				expr := classifyRustCallee(fn, src)
				if expr.kind != "skip" {
					e.pendingCalls = append(e.pendingCalls, rustPendingCall{
						callerID:   callerID,
						callerPkg:  callerPkg,
						callerRecv: callerRecv,
						calleeExpr: expr,
						filePath:   filePath,
						line:       lineOfPoint(n.StartPoint().Row),
					})
				}
			}
		}
		for i := 0; i < int(n.NamedChildCount()); i++ {
			walk(n.NamedChild(i))
		}
	}
	walk(body)
}

// classifyRustCallee inspects a call_expression's `function` field
// and classifies it. method-style x.method() parses with function =
// field_expression; that's the "method" lane.
func classifyRustCallee(n *sitter.Node, src []byte) rustCallee {
	switch n.Type() {
	case "identifier":
		return rustCallee{kind: "bare", parts: []string{nodeText(n, src)}}
	case "scoped_identifier":
		parts := flattenRustScoped(n, src)
		if len(parts) < 2 {
			return rustCallee{kind: "skip"}
		}
		if parts[0] == "Self" || parts[0] == "self" {
			return rustCallee{kind: "self", parts: parts}
		}
		return rustCallee{kind: "scoped", parts: parts}
	case "field_expression":
		obj := n.ChildByFieldName("value")
		fieldNode := n.ChildByFieldName("field")
		if obj == nil || fieldNode == nil {
			return rustCallee{kind: "skip"}
		}
		field := nodeText(fieldNode, src)
		if obj.Type() == "self" {
			return rustCallee{kind: "self", parts: []string{"self", field}}
		}
		if obj.Type() == "identifier" {
			return rustCallee{kind: "method", parts: []string{nodeText(obj, src), field}}
		}
		return rustCallee{kind: "skip"}
	case "generic_function":
		// fn::<T>(args) — the actual function is in the `function` field.
		if inner := n.ChildByFieldName("function"); inner != nil {
			return classifyRustCallee(inner, src)
		}
	}
	return rustCallee{kind: "skip"}
}

// flattenRustScoped turns scoped_identifier A::B::c into ["A","B","c"].
func flattenRustScoped(n *sitter.Node, src []byte) []string {
	var out []string
	cur := n
	for cur != nil && cur.Type() == "scoped_identifier" {
		nameNode := cur.ChildByFieldName("name")
		pathNode := cur.ChildByFieldName("path")
		if nameNode == nil {
			return nil
		}
		out = append([]string{nodeText(nameNode, src)}, out...)
		cur = pathNode
	}
	if cur != nil {
		switch cur.Type() {
		case "identifier", "type_identifier", "scoped_type_identifier",
			"self", "metavariable", "super", "crate":
			out = append([]string{nodeText(cur, src)}, out...)
		default:
			return nil
		}
	}
	return out
}

func (e *rustExtractor) resolveCall(c rustPendingCall) string {
	switch c.calleeExpr.kind {
	case "bare":
		name := c.calleeExpr.parts[0]
		// Same-file fn.
		if id := e.symbolIn(c.callerPkg, name); id != "" {
			return id
		}
		// `use path::Item;` — Item is a free fn, struct, or enum we
		// can resolve via the import table.
		imports := e.fileImports[c.filePath]
		if imports == nil {
			return ""
		}
		if fi, ok := imports.fromImports[name]; ok {
			return e.symbolIn(fi.modulePath, fi.name)
		}
		return ""

	case "self":
		// self.X / Self::X — resolve to a method on the impl's receiver.
		if c.callerRecv == "" || len(c.calleeExpr.parts) < 2 {
			return ""
		}
		methodName := c.calleeExpr.parts[1]
		return e.symbolIn(c.callerPkg, c.callerRecv+"."+methodName)

	case "scoped":
		// A::B::c — last segment is the callee; everything before is
		// the path. If the head matches a binding in the import table,
		// resolve through it; otherwise treat the whole prefix as a
		// modulePath.
		parts := c.calleeExpr.parts
		last := parts[len(parts)-1]
		head := parts[0]
		mid := parts[1 : len(parts)-1]

		imports := e.fileImports[c.filePath]
		if imports != nil {
			if fi, ok := imports.fromImports[head]; ok {
				// `use crate::foo::Bar;` then `Bar::method()` →
				// resolve to Bar.method in package fi.modulePath.
				if len(mid) == 0 {
					return e.symbolIn(fi.modulePath, fi.name+"."+last)
				}
			}
		}
		// Same-file `Type::method()` — when the head matches a type
		// defined in the caller's own package, resolve to that type's
		// method directly. This catches inherent-impl method calls
		// within the file that declared the type.
		if len(mid) == 0 {
			if id := e.symbolIn(c.callerPkg, head+"."+last); id != "" {
				return id
			}
		}
		// `crate::foo::bar` / `foo::bar::baz` — treat everything-but-
		// last as a packagePath and last as a name.
		modPath := strings.Join(parts[:len(parts)-1], "::")
		modPath = normalizeUsePath(modPath)
		if id := e.symbolIn(modPath, last); id != "" {
			return id
		}
		return ""

	case "method":
		// x.method() — without type info we can't resolve the receiver.
		// Skip; this is the LSP-as-consumer upgrade lane.
		return ""
	}
	return ""
}

func (e *rustExtractor) symbolIn(pkg, name string) string {
	bucket := e.symbols[pkg]
	if bucket == nil {
		return ""
	}
	return bucket[name]
}

// ---- helpers ---------------------------------------------------------------

func (e *rustExtractor) addNode(n Node) bool {
	if _, ok := e.nodeIDs[n.ID]; ok {
		return false
	}
	e.nodeIDs[n.ID] = struct{}{}
	e.nodes = append(e.nodes, n)
	return true
}

// rustPackagePath maps a project-root-relative .rs path to a
// crate-relative module path in `::` form. See the package comment
// for the rules; this function is the single source of truth.
func rustPackagePath(relPath string) string {
	p := filepath.ToSlash(relPath)
	p = strings.TrimSuffix(p, ".rs")
	p = strings.TrimPrefix(p, "src/")
	if p == "lib" || p == "main" {
		return p
	}
	if strings.HasSuffix(p, "/mod") {
		p = strings.TrimSuffix(p, "/mod")
	}
	return strings.ReplaceAll(p, "/", "::")
}

// rustLeafName returns the rightmost `::` segment for display.
func rustLeafName(pkg string) string {
	if i := strings.LastIndex(pkg, "::"); i >= 0 {
		return pkg[i+2:]
	}
	return pkg
}

// normalizeUsePath strips a leading `crate::` so the resulting string
// can be directly compared with synthesized packagePaths.
func normalizeUsePath(p string) string {
	return strings.TrimPrefix(p, "crate::")
}

// splitLastSegment splits a `::`-delimited path on its final
// separator. Returns (prefix, last). For input "foo::bar::Baz" it
// yields ("foo::bar", "Baz"). For "Baz" alone it returns ("", "Baz").
func splitLastSegment(p string) (string, string) {
	if i := strings.LastIndex(p, "::"); i >= 0 {
		return p[:i], p[i+2:]
	}
	return "", p
}

// joinPrefix appends segment to prefix using `::`. Either empty side
// is handled gracefully.
func joinPrefix(prefix, segment string) string {
	switch {
	case prefix == "":
		return segment
	case segment == "":
		return prefix
	default:
		return prefix + "::" + segment
	}
}
