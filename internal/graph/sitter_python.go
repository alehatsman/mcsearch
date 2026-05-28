package graph

// Python tree-sitter extractor. Emits file / function / class / method
// / import nodes and contains / has_method / imports / calls edges for
// every .py file under projectRoot.
//
// Package-path encoding: dot-form module path (e.g. "foo.bar" for
// foo/bar.py, "foo" for foo/__init__.py). Matches Python's idiomatic
// spelling and lets cross-file resolution work by lookup —
// `from foo.bar import baz` directly indexes packagePath="foo.bar",
// name="baz". Go's slash-form package paths still live in the same
// graph_nodes table; the `language` metadata field disambiguates.
//
// Call-resolution precision is best-effort by name (tree-sitter has
// no type info). Same-file calls resolve precisely; cross-file
// resolves through imports + symbol table; everything else (dynamic
// dispatch, monkey-patched modules, late-binding decorators) is
// skipped silently. Type-resolved precision lands later via the
// LSP-as-consumer upgrade (vision scope cut #4).

import (
	"context"
	"path/filepath"
	"strings"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/python"
)

func init() { Register(newPythonExtractor) }

// newPythonExtractor returns a fresh extractor instance. The framework
// builds one per Run so accumulator state is isolated.
func newPythonExtractor() Extractor {
	return &pythonExtractor{
		nodeIDs:     map[string]struct{}{},
		symbols:     map[string]map[string]string{},
		fileImports: map[string]*pyImportTable{},
	}
}

type pythonExtractor struct {
	projectRoot string

	// Emitted graph data, accumulated across ProcessFile calls and
	// returned from Finalize.
	nodes []Node
	edges []Edge
	// nodeIDs dedupes accidental double-emits (e.g. a class re-walked
	// from a decorated_definition wrapper) without an O(n) scan.
	nodeIDs map[string]struct{}

	// Per-package symbol table: packagePath → name → nodeID.
	// Populated in ProcessFile, consumed in Finalize for call
	// resolution. Multiple symbols per package live here (functions,
	// classes, methods — the latter as `ClassName.method`).
	symbols map[string]map[string]string

	// Per-file import context. fileImports[relPath] holds the import
	// shape parsed for the file containing the call; Finalize uses it
	// to map bare/attribute callees back to packagePaths.
	fileImports map[string]*pyImportTable

	// Deferred call sites. Resolved in Finalize once every file's
	// symbols are in the table; same-file resolution is also done
	// there so the logic is in one place.
	pendingCalls []pyPendingCall

	// Warnings surfaced through Finalize.
	warnings []string
}

// pyImportTable captures all `import X` and `from X import Y` forms
// in one file. The shape is keyed by the *local name* the file uses
// — that's what a call site spells — so resolution is one map lookup.
type pyImportTable struct {
	// modules: localName → fully-qualified module dotted path.
	// `import foo.bar`         → {"foo": "foo.bar"} (Python binds the
	//                            top-level name; `foo.bar.baz()` calls
	//                            still need the chain)
	// `import foo.bar as fb`   → {"fb":  "foo.bar"}
	modules map[string]string

	// fromImports: localName → (packagePath, originalName).
	// `from foo import bar`           → {"bar":  ("foo",     "bar")}
	// `from foo.baz import qux as q`  → {"q":    ("foo.baz", "qux")}
	fromImports map[string]pyFromImport
}

type pyFromImport struct {
	pkg  string // module the name was imported from, dot form
	name string // original name on the other side
}

// pyPendingCall is one deferred call site, captured during the
// per-file walk and resolved by Finalize after every file's
// declarations are in the symbol table.
type pyPendingCall struct {
	callerID   string
	callerPkg  string // packagePath of the file containing the caller
	callerCls  string // enclosing class name when caller is a method; "" otherwise
	calleeExpr pyCallee
	filePath   string
	line       int
}

// pyCallee captures the syntactic shape of a call's callee. Sitter
// returns a CallExpr's `function` field as one of: identifier, attribute,
// or something we don't try to resolve (call result, subscript, lambda).
//
// kind values:
//
//	"bare"  — single identifier;            parts = ["name"]
//	"attr"  — attribute chain a.b.c;        parts = ["a","b","c"]
//	"self"  — self.X variant of attr;       parts = ["self","X", ...]
//	"skip"  — unresolvable / dynamic
type pyCallee struct {
	kind  string
	parts []string
}

// ---- Extractor interface ---------------------------------------------------

func (e *pythonExtractor) Name() string               { return "python" }
func (e *pythonExtractor) Language() *sitter.Language { return python.GetLanguage() }
func (e *pythonExtractor) Extensions() []string       { return []string{".py"} }

func (e *pythonExtractor) Init(_ context.Context, root string) error {
	e.projectRoot = root
	return nil
}

func (e *pythonExtractor) ProcessFile(_ context.Context, in FileInput) error {
	pkg := pythonPackagePath(in.RelPath)

	// Package node, emitted lazily on first sight of a file in pkg.
	// addNode dedupes by ID, so subsequent files in the same package
	// hit the fast path. Package nodes mirror Go's layer so graph_deps
	// / graph_callers / guide rendering treat Python the same way.
	pkgID := NodeID("", pkg, NodePackage, pkg)
	e.addNode(Node{
		ID:            pkgID,
		Kind:          NodePackage,
		Name:          packageLeafName(pkg),
		QualifiedName: pkg,
		PackagePath:   pkg,
		Metadata:      map[string]any{"language": "python"},
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
		Metadata:      map[string]any{"language": "python"},
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

	imports := &pyImportTable{
		modules:     map[string]string{},
		fromImports: map[string]pyFromImport{},
	}
	e.fileImports[in.RelPath] = imports

	// Walk only the module's *named* top-level children. Comments and
	// whitespace come through as anonymous nodes that we don't want
	// to descend into.
	root := in.Root
	for i := 0; i < int(root.NamedChildCount()); i++ {
		child := root.NamedChild(i)
		e.processTopLevel(child, in.Source, in.RelPath, pkg, fileID, imports)
	}
	return nil
}

func (e *pythonExtractor) Finalize(_ context.Context) ([]Node, []Edge, []string, error) {
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

// processTopLevel handles one direct child of the module root. The
// child is either a definition (function/class/decorated), an import
// statement, or something we don't care about (assignment, if-block,
// docstring expression). Imports are recorded into table; defs add
// nodes/edges and recurse for nested defs + call sites inside their
// bodies.
func (e *pythonExtractor) processTopLevel(
	n *sitter.Node, src []byte,
	filePath, pkg, fileID string, imports *pyImportTable,
) {
	if n == nil {
		return
	}
	kind := n.Type()
	// decorated_definition wraps a function or class; unwrap once.
	if kind == "decorated_definition" {
		inner := n.ChildByFieldName("definition")
		if inner != nil {
			e.processTopLevel(inner, src, filePath, pkg, fileID, imports)
		}
		return
	}
	switch kind {
	case "function_definition":
		e.addFunction(n, src, filePath, pkg, fileID, "")
	case "class_definition":
		e.addClass(n, src, filePath, pkg, fileID)
	case "import_statement":
		e.parseImport(n, src, filePath, pkg, fileID, imports)
	case "import_from_statement":
		e.parseImportFrom(n, src, filePath, pkg, fileID, imports)
	}
}

// addFunction registers a top-level function OR a class method. When
// className is non-empty, the function is a method on that class and
// has_method + class containment edges are also emitted.
func (e *pythonExtractor) addFunction(
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
	meta := map[string]any{"language": "python"}
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
		// Top-level functions are also indexed by bare name so
		// same-file bare calls resolve. Methods are reachable only
		// via ClassName.method.
		if className == "" {
			e.symbols[pkg][name] = id
		}
	}

	// containment: file → function/method always; class → method via
	// has_method when applicable.
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

	// Collect call sites inside the body. The walker descends only
	// into expressions, not into nested function defs, so a `def f()`
	// inside `def g()` doesn't attribute its inner calls to g.
	body := n.ChildByFieldName("body")
	if body != nil {
		e.collectCalls(body, src, id, pkg, className, filePath)
	}
}

// addClass emits a class node + recurses into the body to pick up
// methods. Nested classes are processed recursively as top-level
// (their QN does not include the outer class — first cut).
func (e *pythonExtractor) addClass(
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
		Metadata:      map[string]any{"language": "python"},
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
		kind := child.Type()
		if kind == "decorated_definition" {
			inner := child.ChildByFieldName("definition")
			if inner != nil {
				child = inner
				kind = inner.Type()
			}
		}
		switch kind {
		case "function_definition":
			e.addFunction(child, src, filePath, pkg, fileID, name)
		case "class_definition":
			// Nested class — emit as a top-level class. Python allows
			// it; the QN-without-outer is a first-cut simplification.
			e.addClass(child, src, filePath, pkg, fileID)
		}
	}
}

// ---- import parsing --------------------------------------------------------

// parseImport handles `import foo`, `import foo.bar`, `import foo as f`,
// `import foo, bar`. Tree-sitter emits a sequence of dotted_name or
// aliased_import children directly under import_statement. Each
// imported module also becomes an `import` node + `imports` edge so
// the graph surface matches what Go's extractor emits.
func (e *pythonExtractor) parseImport(
	n *sitter.Node, src []byte,
	filePath, currentPkg, fileID string,
	imports *pyImportTable,
) {
	startLine := lineOfPoint(n.StartPoint().Row)
	for i := 0; i < int(n.NamedChildCount()); i++ {
		child := n.NamedChild(i)
		if child == nil {
			continue
		}
		var fullPath string
		switch child.Type() {
		case "dotted_name":
			fullPath = nodeText(child, src)
			if fullPath == "" {
				continue
			}
			// `import foo.bar` binds the top-level name `foo`; the
			// local name resolves to package `foo`, NOT `foo.bar`.
			// Call sites like `foo.bar.baz()` extend the chain from
			// here, so the resolver rebuilds the full path itself.
			local := topName(fullPath)
			imports.modules[local] = local
		case "aliased_import":
			nameNode := child.ChildByFieldName("name")
			aliasNode := child.ChildByFieldName("alias")
			if nameNode == nil || aliasNode == nil {
				continue
			}
			fullPath = nodeText(nameNode, src)
			alias := nodeText(aliasNode, src)
			if fullPath == "" || alias == "" {
				continue
			}
			imports.modules[alias] = fullPath
		default:
			continue
		}

		impID := NodeID("", currentPkg, NodeImport, fullPath)
		e.addNode(Node{
			ID:            impID,
			Kind:          NodeImport,
			Name:          fullPath,
			QualifiedName: fullPath,
			PackagePath:   currentPkg,
			Metadata:      map[string]any{"language": "python"},
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
		e.emitPkgImportEdge(currentPkg, impID)
	}
}

// parseImportFrom handles `from foo import bar`, `from . import x`,
// `from ..foo.bar import baz as b`. Relative imports (leading dots)
// resolve against the file's package.
func (e *pythonExtractor) parseImportFrom(
	n *sitter.Node, src []byte,
	filePath, currentPkg, fileID string,
	imports *pyImportTable,
) {
	moduleNode := n.ChildByFieldName("module_name")
	if moduleNode == nil {
		return
	}
	module := resolveImportModule(moduleNode, src, currentPkg)
	if module == "" {
		return
	}

	// Iterate name / aliased_import children (the module_name child
	// is skipped — we consumed it above).
	for i := 0; i < int(n.NamedChildCount()); i++ {
		child := n.NamedChild(i)
		if child == nil || child == moduleNode {
			continue
		}
		switch child.Type() {
		case "dotted_name":
			name := nodeText(child, src)
			if name == "" {
				continue
			}
			imports.fromImports[name] = pyFromImport{pkg: module, name: name}
		case "aliased_import":
			nameNode := child.ChildByFieldName("name")
			aliasNode := child.ChildByFieldName("alias")
			if nameNode == nil || aliasNode == nil {
				continue
			}
			name := nodeText(nameNode, src)
			alias := nodeText(aliasNode, src)
			if name == "" || alias == "" {
				continue
			}
			imports.fromImports[alias] = pyFromImport{pkg: module, name: name}
		}
	}

	// Emit an import node + file→import edge per imported module so
	// the graph surface mirrors what Go's extractor does (graph_deps
	// answers stay symmetric across languages).
	impID := NodeID("", currentPkg, NodeImport, module)
	e.addNode(Node{
		ID:            impID,
		Kind:          NodeImport,
		Name:          module,
		QualifiedName: module,
		PackagePath:   currentPkg,
		Metadata:      map[string]any{"language": "python"},
	})
	startLine := lineOfPoint(n.StartPoint().Row)
	e.edges = append(e.edges, Edge{
		ID:        EdgeID(fileID, EdgeImports, impID, filePath, startLine),
		Kind:      EdgeImports,
		SrcID:     fileID,
		DstID:     impID,
		FilePath:  filePath,
		StartLine: startLine,
		EndLine:   startLine,
	})
	e.emitPkgImportEdge(currentPkg, impID)
}

// emitPkgImportEdge mirrors Go's per-package imports edge (`pkg →
// import` with file/line = ""/0) so graph_deps' package-level query
// finds Python imports the same way it finds Go's. Multiple files in
// the same package importing the same module collide on EdgeID, which
// is exactly what we want — one edge per (pkg, import).
func (e *pythonExtractor) emitPkgImportEdge(pkg, impID string) {
	pkgID := NodeID("", pkg, NodePackage, pkg)
	e.edges = append(e.edges, Edge{
		ID:    EdgeID(pkgID, EdgeImports, impID, "", 0),
		Kind:  EdgeImports,
		SrcID: pkgID,
		DstID: impID,
	})
}

// resolveImportModule returns the dotted module path for a
// from-import's source. Handles plain dotted_name and the
// relative_import case (leading dots resolve against currentPkg).
func resolveImportModule(n *sitter.Node, src []byte, currentPkg string) string {
	switch n.Type() {
	case "dotted_name":
		return nodeText(n, src)
	case "relative_import":
		// relative_import children: import_prefix (the dots) +
		// optional dotted_name.
		dots := 0
		var tail string
		for i := 0; i < int(n.NamedChildCount()); i++ {
			c := n.NamedChild(i)
			if c == nil {
				continue
			}
			switch c.Type() {
			case "import_prefix":
				dots = strings.Count(nodeText(c, src), ".")
			case "dotted_name":
				tail = nodeText(c, src)
			}
		}
		// `.` resolves to the file's own package; `..` strips one
		// dot from the package path; etc.
		parts := strings.Split(currentPkg, ".")
		strip := dots - 1
		if strip < 0 {
			strip = 0
		}
		if strip > len(parts) {
			strip = len(parts)
		}
		base := strings.Join(parts[:len(parts)-strip], ".")
		if tail == "" {
			return base
		}
		if base == "" {
			return tail
		}
		return base + "." + tail
	}
	return ""
}

// ---- call collection + resolution ------------------------------------------

// collectCalls walks `body` and accumulates every `call` node into
// pendingCalls. The walker stops at nested function/class definitions
// (we don't attribute their inner calls to the enclosing function;
// they get processed as their own nodes).
func (e *pythonExtractor) collectCalls(
	body *sitter.Node, src []byte,
	callerID, callerPkg, callerCls, filePath string,
) {
	var walk func(*sitter.Node)
	walk = func(n *sitter.Node) {
		if n == nil {
			return
		}
		switch n.Type() {
		case "function_definition", "class_definition", "lambda":
			return
		case "call":
			fn := n.ChildByFieldName("function")
			if fn != nil {
				expr := classifyCallee(fn, src)
				if expr.kind != "skip" {
					e.pendingCalls = append(e.pendingCalls, pyPendingCall{
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

// classifyCallee turns a tree-sitter `function` field into a
// resolvable shape. Anything we can't normalize to identifier or
// attribute chain becomes "skip".
func classifyCallee(n *sitter.Node, src []byte) pyCallee {
	switch n.Type() {
	case "identifier":
		return pyCallee{kind: "bare", parts: []string{nodeText(n, src)}}
	case "attribute":
		parts := flattenAttribute(n, src)
		if len(parts) < 2 {
			return pyCallee{kind: "skip"}
		}
		if parts[0] == "self" {
			return pyCallee{kind: "self", parts: parts}
		}
		return pyCallee{kind: "attr", parts: parts}
	}
	return pyCallee{kind: "skip"}
}

// flattenAttribute turns `a.b.c` (parsed as attribute(attribute(a,b),c))
// into ["a","b","c"]. Returns nil if any intermediate isn't a plain
// identifier — we don't try to resolve `(x or y).z()`.
func flattenAttribute(n *sitter.Node, src []byte) []string {
	var out []string
	cur := n
	for cur != nil && cur.Type() == "attribute" {
		attr := cur.ChildByFieldName("attribute")
		obj := cur.ChildByFieldName("object")
		if attr == nil || obj == nil {
			return nil
		}
		out = append([]string{nodeText(attr, src)}, out...)
		cur = obj
	}
	if cur == nil || cur.Type() != "identifier" {
		return nil
	}
	out = append([]string{nodeText(cur, src)}, out...)
	return out
}

// resolveCall maps a pyPendingCall to a destination NodeID, or "" if
// the callee is unresolvable / out of project scope.
func (e *pythonExtractor) resolveCall(c pyPendingCall) string {
	switch c.calleeExpr.kind {
	case "bare":
		name := c.calleeExpr.parts[0]
		// 1) Same-file (same-package) symbol — top-level functions
		//    and classes both live in e.symbols[c.callerPkg].
		if id := e.symbolIn(c.callerPkg, name); id != "" {
			return id
		}
		// 2) Look through the file's `from X import name` table.
		imports := e.fileImports[c.filePath]
		if imports == nil {
			return ""
		}
		if fi, ok := imports.fromImports[name]; ok {
			return e.symbolIn(fi.pkg, fi.name)
		}
		return ""

	case "self":
		// self.method() inside a method on class C → resolve to C.method.
		if c.callerCls == "" || len(c.calleeExpr.parts) < 2 {
			return ""
		}
		methodName := c.calleeExpr.parts[1]
		return e.symbolIn(c.callerPkg, c.callerCls+"."+methodName)

	case "attr":
		// a.b.c() — try to walk: a is either an imported module
		// (modules[a]) or a from-imported class (fromImports[a]).
		// Resolve trailing name against the resulting package.
		imports := e.fileImports[c.filePath]
		if imports == nil {
			return ""
		}
		head := c.calleeExpr.parts[0]
		tail := c.calleeExpr.parts[1:]
		if len(tail) == 0 {
			return ""
		}
		// `import X.Y.Z` binds the local name X (or, for aliased
		// imports, the alias) to its referent module path. A call
		// site `X.Y.Z.f()` extends that path: the full dotted access
		// is bound + tail, and the trailing element is the callable.
		if mod, ok := imports.modules[head]; ok {
			full := mod
			if len(tail) > 0 {
				full = mod + "." + strings.Join(tail, ".")
			}
			lastDot := strings.LastIndex(full, ".")
			if lastDot < 0 {
				return ""
			}
			return e.symbolIn(full[:lastDot], full[lastDot+1:])
		}
		// `from foo import Bar` then `Bar.method()` — resolve to
		// the class's method node if foo defined that class.
		if fi, ok := imports.fromImports[head]; ok {
			if len(tail) == 1 {
				return e.symbolIn(fi.pkg, fi.name+"."+tail[0])
			}
			// Deeper chains (Bar.x.y) aren't first-cut.
			return ""
		}
		return ""
	}
	return ""
}

func (e *pythonExtractor) symbolIn(pkg, name string) string {
	bucket := e.symbols[pkg]
	if bucket == nil {
		return ""
	}
	return bucket[name]
}

// ---- helpers ---------------------------------------------------------------

// addNode appends n only if its ID hasn't been added yet. Returns true
// on first insertion so the caller can also seed the symbol table.
func (e *pythonExtractor) addNode(n Node) bool {
	if _, ok := e.nodeIDs[n.ID]; ok {
		return false
	}
	e.nodeIDs[n.ID] = struct{}{}
	e.nodes = append(e.nodes, n)
	return true
}

// pythonPackagePath converts a .py file's project-root-relative path
// to its Python module path in dot form. __init__ is folded into its
// containing directory's name (Python's package convention).
func pythonPackagePath(relPath string) string {
	p := filepath.ToSlash(relPath)
	p = strings.TrimSuffix(p, ".py")
	if p == "__init__" {
		return ""
	}
	if strings.HasSuffix(p, "/__init__") {
		p = strings.TrimSuffix(p, "/__init__")
	}
	return strings.ReplaceAll(p, "/", ".")
}

// packageLeafName returns the last component of a dot-form package
// path. Used as the human-readable `name` on NodePackage rows so
// renderers can show `bar` instead of `foo.bar` in tight UI columns.
func packageLeafName(pkg string) string {
	if i := strings.LastIndexByte(pkg, '.'); i >= 0 {
		return pkg[i+1:]
	}
	if pkg == "" {
		return "."
	}
	return pkg
}

// topName returns the leading component of a dotted name (`foo.bar.baz`
// → `foo`). Empty input yields empty output.
func topName(dotted string) string {
	if i := strings.IndexByte(dotted, '.'); i >= 0 {
		return dotted[:i]
	}
	return dotted
}

func nodeText(n *sitter.Node, src []byte) string {
	if n == nil {
		return ""
	}
	start := int(n.StartByte())
	end := int(n.EndByte())
	if start < 0 || end > len(src) || start > end {
		return ""
	}
	return string(src[start:end])
}

// lineOfPoint converts tree-sitter's 0-based row to a 1-based line.
func lineOfPoint(row uint32) int { return int(row) + 1 }

func ensureMap(m map[string]string) map[string]string {
	if m == nil {
		return map[string]string{}
	}
	return m
}
