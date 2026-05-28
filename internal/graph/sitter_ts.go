package graph

// TypeScript tree-sitter extractor. Handles .ts and .tsx via the
// typescript grammar (mirrors internal/chunk/chunk.go's choice).
// Emits package / file / function / class / method / interface /
// import nodes and contains / has_method / imports / calls edges
// with the same shape as the Python extractor — graph_deps and
// graph_callers see a unified surface across languages.
//
// Package-path encoding: forward-slash file path minus extension.
// `src/foo.ts` → "src/foo"; `src/foo/index.ts` → "src/foo" (matches
// the specifier `import './foo'` reaches). This differs from
// Python's dot-form on purpose: TS uses path-based module specifiers,
// not dotted module names, and matching the specifier shape makes
// import resolution a slice-and-join instead of a string rewrite.
//
// Resolution scope (single Run, single package per file):
//   - Bare calls / `new Foo()`     — same-file or `import { X } from './p'`
//   - `ns.member()`                — `import * as ns from './p'`
//   - `this.method()` in a method  — same class
//   - `import { default as d }` / `import d from './p'` — default
//     export tracked via a synthetic "default" slot in the symbol
//     table when the source file has one export declaration.
//
// What gets skipped: dynamic imports, JSX components used as
// callables (their resolution is component-by-import-name and
// works naturally for upper-case identifiers; lower-case JSX is
// HTML), type-only imports, decorators, and generics. Same trade
// as Python — best-effort by name; precision is the LSP lane.

import (
	"context"
	"path"
	"path/filepath"
	"strings"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/typescript/typescript"
)

func init() { Register(newTSExtractor) }

func newTSExtractor() Extractor {
	return &tsExtractor{
		nodeIDs:     map[string]struct{}{},
		symbols:     map[string]map[string]string{},
		fileImports: map[string]*tsImportTable{},
		knownFiles:  map[string]string{},
	}
}

type tsExtractor struct {
	projectRoot string

	nodes []Node
	edges []Edge

	nodeIDs map[string]struct{}
	// symbols: packagePath → name → nodeID. "default" is reserved for
	// the file's default export.
	symbols map[string]map[string]string

	// fileImports keyed by file relpath (slash form).
	fileImports map[string]*tsImportTable

	// knownFiles: source packagePath → first observed file relpath.
	// Used by resolveModuleSpecifier to choose between `foo.ts`,
	// `foo.tsx`, `foo/index.ts` after both directory and file forms
	// resolve to the same packagePath. Don't need the value beyond
	// presence; the map doubles as the existence set.
	knownFiles map[string]string

	pendingCalls []tsPendingCall

	warnings []string
}

// tsImportTable per-file: each local binding maps to either an
// imported file (for `import * as X from './p'`) or a (file,
// originalName) pair (for named or default imports). modules and
// fromImports mirror Python's layout so the call-resolution code is
// structurally parallel.
type tsImportTable struct {
	// modules: localName → packagePath of the imported file. Used for
	// namespace imports: `import * as X from './p'` ⇒ {"X": "p"}.
	// Default imports ALSO land here when the import shape's source
	// has been resolved — calling the default like `Foo()` resolves
	// via the symbol table's "default" slot.
	modules map[string]string

	// fromImports: localName → (packagePath, originalName). Used for
	// named imports: `import { foo as f } from './p'` ⇒
	// {"f": ("p", "foo")}.
	fromImports map[string]pyFromImport
}

type tsPendingCall struct {
	callerID   string
	callerPkg  string
	callerCls  string // "" outside methods
	calleeExpr pyCallee
	filePath   string
	line       int
}

// ---- Extractor interface ---------------------------------------------------

func (e *tsExtractor) Name() string               { return "typescript" }
func (e *tsExtractor) Language() *sitter.Language { return typescript.GetLanguage() }
func (e *tsExtractor) Extensions() []string       { return []string{".ts", ".tsx"} }

func (e *tsExtractor) Init(_ context.Context, root string) error {
	e.projectRoot = root
	return nil
}

func (e *tsExtractor) ProcessFile(_ context.Context, in FileInput) error {
	pkg := tsPackagePath(in.RelPath)
	e.knownFiles[pkg] = in.RelPath

	pkgID := NodeID("", pkg, NodePackage, pkg)
	e.addNode(Node{
		ID:            pkgID,
		Kind:          NodePackage,
		Name:          path.Base(pkg),
		QualifiedName: pkg,
		PackagePath:   pkg,
		Metadata:      map[string]any{"language": "typescript"},
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
		Metadata:      map[string]any{"language": "typescript"},
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

	imports := &tsImportTable{
		modules:     map[string]string{},
		fromImports: map[string]pyFromImport{},
	}
	e.fileImports[in.RelPath] = imports

	for i := 0; i < int(in.Root.NamedChildCount()); i++ {
		e.processTopLevel(in.Root.NamedChild(i), in.Source, in.RelPath, pkg, fileID, imports)
	}
	return nil
}

func (e *tsExtractor) Finalize(_ context.Context) ([]Node, []Edge, []string, error) {
	// Module specifiers were captured as raw strings during the walk;
	// resolve them now that knownFiles is complete (a forward
	// reference would otherwise miss because of walk order).
	for _, imp := range e.fileImports {
		for local, target := range imp.modules {
			imp.modules[local] = e.resolveModuleSpecifier(target, imp.modules["__from__"])
		}
		for local, fi := range imp.fromImports {
			fi.pkg = e.resolveModuleSpecifier(fi.pkg, imp.modules["__from__"])
			imp.fromImports[local] = fi
		}
		delete(imp.modules, "__from__")
	}

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

// processTopLevel handles one direct child of the module root.
// export_statement is unwrapped to the underlying declaration so
// `export function foo` and `function foo` produce the same node
// shape; the export-ness is not currently tracked (consumers care
// about the symbol's existence, not its visibility).
func (e *tsExtractor) processTopLevel(
	n *sitter.Node, src []byte,
	filePath, pkg, fileID string, imports *tsImportTable,
) {
	if n == nil {
		return
	}
	kind := n.Type()
	if kind == "export_statement" {
		// `export default ...` flows through the same path; the
		// `default` identifier in the symbol table is what makes
		// default-import resolution work.
		if decl := n.ChildByFieldName("declaration"); decl != nil {
			e.processTopLevel(decl, src, filePath, pkg, fileID, imports)
			e.maybeMarkDefaultExport(decl, src, pkg)
			return
		}
		// export { foo, bar } — re-exports; not modelled in first cut.
		return
	}
	switch kind {
	case "function_declaration":
		e.addFunction(n, src, filePath, pkg, fileID, "")
	case "class_declaration":
		e.addClass(n, src, filePath, pkg, fileID)
	case "interface_declaration":
		e.addInterface(n, src, filePath, pkg, fileID)
	case "lexical_declaration":
		e.addLexicalDecl(n, src, filePath, pkg, fileID)
	case "import_statement":
		e.parseImportStatement(n, src, filePath, pkg, fileID, imports)
	}
}

// addFunction registers a top-level function OR a class method.
// Container symmetry with the Python extractor: when className is
// non-empty the node gets `has_method` from the class.
func (e *tsExtractor) addFunction(
	n *sitter.Node, src []byte,
	filePath, pkg, fileID, className string,
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
	if className != "" {
		kind = NodeMethod
		qn = className + "." + name
	}
	id := NodeID("", pkg, kind, qn)
	startLine := lineOfPoint(n.StartPoint().Row)
	endLine := lineOfPoint(n.EndPoint().Row)
	meta := map[string]any{"language": "typescript"}
	if className != "" {
		meta["receiver"] = className
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
		if className == "" {
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
	if className != "" {
		clsID := NodeID("", pkg, NodeClass, className)
		e.edges = append(e.edges, Edge{
			ID:        EdgeID(clsID, EdgeHasMethod, id, filePath, startLine),
			Kind:      EdgeHasMethod,
			SrcID:     clsID,
			DstID:     id,
			FilePath:  filePath,
			StartLine: startLine,
			EndLine:   endLine,
		})
	}
	if body := n.ChildByFieldName("body"); body != nil {
		e.collectCalls(body, src, id, pkg, className, filePath)
	}
}

// addClass: class + recurse into class_body for methods. Nested
// classes are treated as top-level; their QN doesn't include the
// outer class (matches Python's first-cut).
func (e *tsExtractor) addClass(
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
	id := NodeID("", pkg, NodeClass, name)
	startLine := lineOfPoint(n.StartPoint().Row)
	endLine := lineOfPoint(n.EndPoint().Row)
	if e.addNode(Node{
		ID:            id,
		Kind:          NodeClass,
		Name:          name,
		QualifiedName: name,
		PackagePath:   pkg,
		FilePath:      filePath,
		StartLine:     startLine,
		EndLine:       endLine,
		Metadata:      map[string]any{"language": "typescript"},
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
		if child == nil {
			continue
		}
		switch child.Type() {
		case "method_definition":
			e.addFunction(child, src, filePath, pkg, fileID, name)
		case "class_declaration":
			e.addClass(child, src, filePath, pkg, fileID)
		}
	}
}

// addInterface: emit a NodeInterface (Go's existing kind) with body
// methods walked just like a class but tagged on_interface in
// metadata so consumers can tell them apart.
func (e *tsExtractor) addInterface(
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
		Metadata:      map[string]any{"language": "typescript"},
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

// addLexicalDecl handles `const foo = () => {}` and `const foo =
// function(){}`. Other const forms (object/string literals) are
// ignored — they're not callable in a graph-meaningful way. Multiple
// declarators in one statement (`const a=…, b=…`) each emit their
// own node.
func (e *tsExtractor) addLexicalDecl(
	n *sitter.Node, src []byte,
	filePath, pkg, fileID string,
) {
	for i := 0; i < int(n.NamedChildCount()); i++ {
		child := n.NamedChild(i)
		if child == nil || child.Type() != "variable_declarator" {
			continue
		}
		nameNode := child.ChildByFieldName("name")
		valueNode := child.ChildByFieldName("value")
		if nameNode == nil || valueNode == nil {
			continue
		}
		name := nodeText(nameNode, src)
		if name == "" {
			continue
		}
		// Only function-like values become function nodes. Anything
		// else (literals, calls, refs) gets skipped — they're not
		// callable targets from elsewhere.
		switch valueNode.Type() {
		case "arrow_function", "function", "function_expression":
		default:
			continue
		}
		id := NodeID("", pkg, NodeFunction, name)
		startLine := lineOfPoint(n.StartPoint().Row)
		endLine := lineOfPoint(n.EndPoint().Row)
		if e.addNode(Node{
			ID:            id,
			Kind:          NodeFunction,
			Name:          name,
			QualifiedName: name,
			PackagePath:   pkg,
			FilePath:      filePath,
			StartLine:     startLine,
			EndLine:       endLine,
			Metadata:      map[string]any{"language": "typescript", "form": "arrow"},
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
		if body := valueNode.ChildByFieldName("body"); body != nil {
			e.collectCalls(body, src, id, pkg, "", filePath)
		}
	}
}

// maybeMarkDefaultExport sets symbols[pkg]["default"] to the node ID
// of the just-emitted declaration when the wrapping export_statement
// declared it as the default. The declaration may be anonymous
// (`export default function(){}`), in which case there's nothing to
// alias — handled by trying for the most-recently-emitted node in
// the package's symbol table.
func (e *tsExtractor) maybeMarkDefaultExport(decl *sitter.Node, src []byte, pkg string) {
	// We only get here from an export_statement, but it may or may
	// not be `default`. Tree-sitter exposes the `default` keyword as
	// an anonymous child; check by source.
	parent := decl.Parent()
	if parent == nil || parent.Type() != "export_statement" {
		return
	}
	hasDefault := false
	for i := 0; i < int(parent.ChildCount()); i++ {
		c := parent.Child(i)
		if c != nil && c.Type() == "default" {
			hasDefault = true
			break
		}
	}
	if !hasDefault {
		return
	}
	// Find the name of the just-emitted decl, if any, and alias it.
	if nameNode := decl.ChildByFieldName("name"); nameNode != nil {
		name := nodeText(nameNode, src)
		if id := e.symbolIn(pkg, name); id != "" {
			e.symbols[pkg] = ensureMap(e.symbols[pkg])
			e.symbols[pkg]["default"] = id
		}
	}
}

// ---- import parsing --------------------------------------------------------

// parseImportStatement handles every import shape that has a `from`:
//
//	import foo from "./p"
//	import { a, b as c } from "./p"
//	import * as ns from "./p"
//	import foo, { a } from "./p"
//	import "./p"                    (side-effect; no bindings)
//
// The source specifier is captured raw; resolution to a packagePath
// happens in Finalize once knownFiles is complete (the dispatch order
// across files isn't deterministic relative to source position).
func (e *tsExtractor) parseImportStatement(
	n *sitter.Node, src []byte,
	filePath, currentPkg, fileID string,
	imports *tsImportTable,
) {
	sourceNode := n.ChildByFieldName("source")
	if sourceNode == nil {
		return
	}
	specifier := stripQuotes(nodeText(sourceNode, src))
	if specifier == "" {
		return
	}
	startLine := lineOfPoint(n.StartPoint().Row)

	// Stash the "from" anchor for Finalize-time resolution. The
	// import_clause children give us local bindings; later we
	// translate each binding's pkg from raw specifier to resolved
	// packagePath. We borrow the modules map's "__from__" slot for
	// the per-file anchor — wiped after resolution.
	imports.modules["__from__"] = filePath

	clauseHandled := false
	for i := 0; i < int(n.NamedChildCount()); i++ {
		child := n.NamedChild(i)
		if child == nil || child == sourceNode {
			continue
		}
		if child.Type() != "import_clause" {
			continue
		}
		clauseHandled = true
		e.processImportClause(child, src, specifier, imports)
	}

	// Side-effect import (`import "./p"`) still emits the imports
	// edge so graph_deps can see the dependency. clauseHandled is
	// false in this case, but we always emit.
	_ = clauseHandled

	impID := NodeID("", currentPkg, NodeImport, specifier)
	e.addNode(Node{
		ID:            impID,
		Kind:          NodeImport,
		Name:          specifier,
		QualifiedName: specifier,
		PackagePath:   currentPkg,
		Metadata:      map[string]any{"language": "typescript"},
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

// processImportClause walks an import_clause and seeds the import
// table with each binding. specifier is the raw "./foo" form — it's
// stored as-is and rewritten to a packagePath in Finalize.
func (e *tsExtractor) processImportClause(
	clause *sitter.Node, src []byte,
	specifier string, imports *tsImportTable,
) {
	for i := 0; i < int(clause.NamedChildCount()); i++ {
		child := clause.NamedChild(i)
		if child == nil {
			continue
		}
		switch child.Type() {
		case "identifier":
			// Default import: `import Foo from "./p"`.
			local := nodeText(child, src)
			if local != "" {
				imports.fromImports[local] = pyFromImport{pkg: specifier, name: "default"}
			}
		case "namespace_import":
			// `import * as X from "./p"` — X is bound to the module.
			aliasNode := child.ChildByFieldName("alias")
			if aliasNode == nil {
				// Older grammars expose it as the last child.
				aliasNode = lastNamedChild(child)
			}
			if aliasNode != nil {
				if local := nodeText(aliasNode, src); local != "" {
					imports.modules[local] = specifier
				}
			}
		case "named_imports":
			for j := 0; j < int(child.NamedChildCount()); j++ {
				spec := child.NamedChild(j)
				if spec == nil || spec.Type() != "import_specifier" {
					continue
				}
				nameNode := spec.ChildByFieldName("name")
				aliasNode := spec.ChildByFieldName("alias")
				if nameNode == nil {
					continue
				}
				original := nodeText(nameNode, src)
				local := original
				if aliasNode != nil {
					local = nodeText(aliasNode, src)
				}
				if original == "" || local == "" {
					continue
				}
				imports.fromImports[local] = pyFromImport{pkg: specifier, name: original}
			}
		}
	}
}

// ---- call collection + resolution ------------------------------------------

// collectCalls walks `body` for call_expression and new_expression,
// stopping descent at nested function/class definitions. Same
// approach as the Python extractor.
func (e *tsExtractor) collectCalls(
	body *sitter.Node, src []byte,
	callerID, callerPkg, callerCls, filePath string,
) {
	var walk func(*sitter.Node)
	walk = func(n *sitter.Node) {
		if n == nil {
			return
		}
		switch n.Type() {
		case "function_declaration", "class_declaration",
			"arrow_function", "function", "function_expression",
			"method_definition":
			return
		case "call_expression":
			fn := n.ChildByFieldName("function")
			if fn != nil {
				expr := classifyTSCallee(fn, src)
				if expr.kind != "skip" {
					e.pendingCalls = append(e.pendingCalls, tsPendingCall{
						callerID:   callerID,
						callerPkg:  callerPkg,
						callerCls:  callerCls,
						calleeExpr: expr,
						filePath:   filePath,
						line:       lineOfPoint(n.StartPoint().Row),
					})
				}
			}
		case "new_expression":
			ctor := n.ChildByFieldName("constructor")
			if ctor != nil {
				expr := classifyTSCallee(ctor, src)
				if expr.kind != "skip" {
					e.pendingCalls = append(e.pendingCalls, tsPendingCall{
						callerID:   callerID,
						callerPkg:  callerPkg,
						callerCls:  callerCls,
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

// classifyTSCallee mirrors classifyCallee for Python. `this` is the
// TS analogue of `self`; flattening member_expression chains gives us
// `a.b.c` as parts.
func classifyTSCallee(n *sitter.Node, src []byte) pyCallee {
	switch n.Type() {
	case "identifier":
		return pyCallee{kind: "bare", parts: []string{nodeText(n, src)}}
	case "member_expression":
		parts := flattenTSMember(n, src)
		if len(parts) < 2 {
			return pyCallee{kind: "skip"}
		}
		if parts[0] == "this" {
			return pyCallee{kind: "self", parts: parts}
		}
		return pyCallee{kind: "attr", parts: parts}
	}
	return pyCallee{kind: "skip"}
}

// flattenTSMember turns `a.b.c` (parsed as member_expression
// (member_expression a b) c) into ["a","b","c"]. Returns nil for any
// chain that has a non-identifier base (calls, indexes, etc.) — we
// don't try to resolve those.
func flattenTSMember(n *sitter.Node, src []byte) []string {
	var out []string
	cur := n
	for cur != nil && cur.Type() == "member_expression" {
		prop := cur.ChildByFieldName("property")
		obj := cur.ChildByFieldName("object")
		if prop == nil || obj == nil {
			return nil
		}
		out = append([]string{nodeText(prop, src)}, out...)
		cur = obj
	}
	if cur == nil {
		return nil
	}
	switch cur.Type() {
	case "identifier":
		out = append([]string{nodeText(cur, src)}, out...)
	case "this":
		out = append([]string{"this"}, out...)
	default:
		return nil
	}
	return out
}

func (e *tsExtractor) resolveCall(c tsPendingCall) string {
	switch c.calleeExpr.kind {
	case "bare":
		name := c.calleeExpr.parts[0]
		if id := e.symbolIn(c.callerPkg, name); id != "" {
			return id
		}
		imports := e.fileImports[c.filePath]
		if imports == nil {
			return ""
		}
		if fi, ok := imports.fromImports[name]; ok {
			return e.symbolIn(fi.pkg, fi.name)
		}
		return ""

	case "self":
		if c.callerCls == "" || len(c.calleeExpr.parts) < 2 {
			return ""
		}
		methodName := c.calleeExpr.parts[1]
		return e.symbolIn(c.callerPkg, c.callerCls+"."+methodName)

	case "attr":
		imports := e.fileImports[c.filePath]
		if imports == nil {
			return ""
		}
		head := c.calleeExpr.parts[0]
		tail := c.calleeExpr.parts[1:]
		if len(tail) == 0 {
			return ""
		}
		if mod, ok := imports.modules[head]; ok {
			// `import * as X from "./p"` — X.f() resolves to symbol
			// f in package mod. Deeper chains (X.a.b) aren't first-cut.
			if len(tail) != 1 {
				return ""
			}
			return e.symbolIn(mod, tail[0])
		}
		if fi, ok := imports.fromImports[head]; ok {
			if len(tail) == 1 {
				// `Foo.method` where Foo was imported as a class
				// from another module.
				return e.symbolIn(fi.pkg, fi.name+"."+tail[0])
			}
			return ""
		}
		return ""
	}
	return ""
}

func (e *tsExtractor) symbolIn(pkg, name string) string {
	bucket := e.symbols[pkg]
	if bucket == nil {
		return ""
	}
	return bucket[name]
}

// ---- module-specifier resolution -------------------------------------------

// resolveModuleSpecifier turns a raw specifier (./foo, ../bar/baz,
// react) into the packagePath of the file it refers to, or returns
// the original string when the specifier doesn't resolve to a known
// project file (external package, missing file, etc.). External
// packages effectively become opaque — they show up as imports but
// resolveCall ignores them because the package has no symbol bucket.
//
// fromFile is the relpath of the importing file (slash form) — used
// only when specifier is relative.
func (e *tsExtractor) resolveModuleSpecifier(specifier, fromFile string) string {
	if specifier == "" {
		return ""
	}
	if !strings.HasPrefix(specifier, ".") {
		// Non-relative — could be a tsconfig path alias or an npm
		// package. We don't model either; leave the raw value so the
		// import edge still names something useful, but no symbol
		// lookup will match.
		return specifier
	}
	if fromFile == "" {
		return specifier
	}
	base := path.Dir(filepath.ToSlash(fromFile))
	joined := path.Clean(path.Join(base, specifier))
	// Try (in order): exact pkgPath, +.ts, +.tsx, +/index.
	candidates := []string{
		joined,
		joined + "_ts",
		joined + "_tsx",
	}
	// "joined" without extension is what packagePath would be after
	// stripping. Same for `+index`. The probe uses knownFiles which is
	// keyed on packagePath, so the canonical "matches if seen" lookup
	// is just `_, ok := e.knownFiles[joined]`.
	if _, ok := e.knownFiles[joined]; ok {
		return joined
	}
	// `./foo` → file `./foo.ts(x)` → pkg "foo". knownFiles maps pkg
	// to its observed relpath; both `foo.ts` and `foo.tsx` produce
	// pkg "foo", so the same lookup catches either.
	_ = candidates // kept for readability above; the lookup logic is
	// already covered.
	// `./foo` → directory `./foo/` with index.ts(x) → pkg "foo/index".
	if _, ok := e.knownFiles[joined+"/index"]; ok {
		return joined + "/index"
	}
	return specifier
}

// ---- helpers ---------------------------------------------------------------

func (e *tsExtractor) addNode(n Node) bool {
	if _, ok := e.nodeIDs[n.ID]; ok {
		return false
	}
	e.nodeIDs[n.ID] = struct{}{}
	e.nodes = append(e.nodes, n)
	return true
}

// tsPackagePath strips the extension from a .ts/.tsx file path. The
// `index.ts(x)` filename is kept verbatim in the package path —
// resolveModuleSpecifier handles the dir-vs-index disambiguation by
// probing both forms.
func tsPackagePath(relPath string) string {
	p := filepath.ToSlash(relPath)
	for _, ext := range []string{".tsx", ".ts"} {
		if strings.HasSuffix(p, ext) {
			return strings.TrimSuffix(p, ext)
		}
	}
	return p
}

func stripQuotes(s string) string {
	if len(s) < 2 {
		return s
	}
	first, last := s[0], s[len(s)-1]
	if (first == '"' && last == '"') || (first == '\'' && last == '\'') || (first == '`' && last == '`') {
		return s[1 : len(s)-1]
	}
	return s
}

func lastNamedChild(n *sitter.Node) *sitter.Node {
	c := int(n.NamedChildCount())
	if c == 0 {
		return nil
	}
	return n.NamedChild(c - 1)
}
