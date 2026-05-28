package graph

// Java tree-sitter extractor. Emits package / file / class / interface
// / enum / method / import nodes and contains / has_method / imports /
// calls edges for every .java file under projectRoot.
//
// Package-path encoding: takes the `package` declaration's dotted
// name verbatim (e.g. "com.example.app"). Files without a package
// declaration fall back to the directory path in dot form. Java's
// compiler requires file location to match the declared package; we
// trust whichever is declared so the graph matches what `import`
// statements resolve to.
//
// Resolution lanes:
//   - bare call `foo()` inside class C  → C.foo (same-class method)
//   - `this.method()`                   → C.method
//   - `ClassName.staticMethod()`        → ClassName.staticMethod in
//                                         same-package OR imported pkg
//   - `new ClassName(...)`              → the ClassName class node
//   - method calls on typed receivers   → out of scope; needs type
//                                         info (LSP-as-consumer lane)
//
// What's deferred: nested/inner classes (only top-level + their direct
// methods), super.X dispatch, generic type-parameter receivers,
// annotations as call sites, `import static path.*` wildcards.

import (
	"context"
	"path/filepath"
	"strings"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/java"
)

func init() { Register(newJavaExtractor) }

func newJavaExtractor() Extractor {
	return &javaExtractor{
		nodeIDs:     map[string]struct{}{},
		symbols:     map[string]map[string]string{},
		fileImports: map[string]*javaImportTable{},
	}
}

type javaExtractor struct {
	projectRoot string

	nodes []Node
	edges []Edge

	nodeIDs map[string]struct{}
	symbols map[string]map[string]string

	fileImports map[string]*javaImportTable

	pendingCalls []javaPendingCall

	warnings []string
}

// javaImportTable: each local class name maps to (packagePath,
// className). For `import static path.Foo.bar`, we also seed the
// statics map with bare member-name → (Foo's pkg, "Foo.bar") so a
// bare call to `bar()` resolves cross-class.
type javaImportTable struct {
	classes map[string]javaClassImport // local class name → import
	statics map[string]javaStaticImport
}

type javaClassImport struct {
	pkg  string // package the class lives in (dot form)
	name string // the class's bare name
}

type javaStaticImport struct {
	pkg  string // declaring class's package
	cls  string // declaring class name
	memb string // imported member name
}

type javaPendingCall struct {
	callerID   string
	callerPkg  string
	callerCls  string // enclosing class name when caller is a method
	calleeExpr javaCallee
	filePath   string
	line       int
}

// javaCallee variants:
//
//	"bare"   — method_invocation with no object: foo()
//	"self"   — this.foo() or super.foo()
//	"attr"   — A.B.C.foo() receiver chain rooted in an identifier
//	"new"    — object_creation_expression: new Foo() or new pkg.Foo()
//	"skip"   — unresolvable / dynamic receiver
type javaCallee struct {
	kind  string
	parts []string
}

// ---- Extractor interface ---------------------------------------------------

func (e *javaExtractor) Name() string               { return "java" }
func (e *javaExtractor) Language() *sitter.Language { return java.GetLanguage() }
func (e *javaExtractor) Extensions() []string       { return []string{".java"} }

func (e *javaExtractor) Init(_ context.Context, root string) error {
	e.projectRoot = root
	return nil
}

func (e *javaExtractor) ProcessFile(_ context.Context, in FileInput) error {
	// First pass — find the package declaration. The declaration is
	// always at the top of the file (Java grammar requires it), so a
	// single-pass walk that visits package_declaration before any
	// other top-level item is fine.
	pkg := javaPackagePath(in.Root, in.Source, in.RelPath)

	pkgID := NodeID("", pkg, NodePackage, pkg)
	e.addNode(Node{
		ID:            pkgID,
		Kind:          NodePackage,
		Name:          javaLeafName(pkg),
		QualifiedName: pkg,
		PackagePath:   pkg,
		Metadata:      map[string]any{"language": "java"},
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
		Metadata:      map[string]any{"language": "java"},
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

	imports := &javaImportTable{
		classes: map[string]javaClassImport{},
		statics: map[string]javaStaticImport{},
	}
	e.fileImports[in.RelPath] = imports

	for i := 0; i < int(in.Root.NamedChildCount()); i++ {
		e.processTopLevel(in.Root.NamedChild(i), in.Source, in.RelPath, pkg, fileID, imports)
	}
	return nil
}

func (e *javaExtractor) Finalize(_ context.Context) ([]Node, []Edge, []string, error) {
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

func (e *javaExtractor) processTopLevel(
	n *sitter.Node, src []byte,
	filePath, pkg, fileID string, imports *javaImportTable,
) {
	if n == nil {
		return
	}
	switch n.Type() {
	case "package_declaration":
		// Already consumed by javaPackagePath; nothing else to do.
	case "import_declaration":
		e.parseImport(n, src, filePath, pkg, fileID, imports)
	case "class_declaration", "record_declaration":
		e.addClassLike(n, src, filePath, pkg, fileID, NodeClass)
	case "interface_declaration":
		e.addClassLike(n, src, filePath, pkg, fileID, NodeInterface)
	case "enum_declaration":
		e.addClassLike(n, src, filePath, pkg, fileID, NodeClass)
	}
}

// addClassLike emits a class/interface/enum node and walks its body
// for method_declaration + constructor_declaration. Nested types are
// not modelled in first cut — their declarations are skipped.
func (e *javaExtractor) addClassLike(
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
	if e.addNode(Node{
		ID:            id,
		Kind:          kind,
		Name:          name,
		QualifiedName: name,
		PackagePath:   pkg,
		FilePath:      filePath,
		StartLine:     startLine,
		EndLine:       endLine,
		Metadata:      map[string]any{"language": "java"},
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
		case "method_declaration":
			e.addMethod(child, src, filePath, pkg, fileID, name, false)
		case "constructor_declaration":
			e.addMethod(child, src, filePath, pkg, fileID, name, true)
		}
	}
}

// addMethod registers a method or constructor on the given class.
// Constructors get the synthetic name "<init>" to keep them
// distinguishable from regular methods named after the class.
func (e *javaExtractor) addMethod(
	n *sitter.Node, src []byte,
	filePath, pkg, fileID, className string, isCtor bool,
) {
	var methodName string
	if isCtor {
		methodName = "<init>"
	} else {
		nameNode := n.ChildByFieldName("name")
		if nameNode == nil {
			return
		}
		methodName = nodeText(nameNode, src)
		if methodName == "" {
			return
		}
	}
	qn := className + "." + methodName
	id := NodeID("", pkg, NodeMethod, qn)
	startLine := lineOfPoint(n.StartPoint().Row)
	endLine := lineOfPoint(n.EndPoint().Row)
	meta := map[string]any{"language": "java", "receiver": className}
	if isCtor {
		meta["constructor"] = true
	}
	if e.addNode(Node{
		ID:            id,
		Kind:          NodeMethod,
		Name:          methodName,
		QualifiedName: qn,
		PackagePath:   pkg,
		FilePath:      filePath,
		StartLine:     startLine,
		EndLine:       endLine,
		Metadata:      meta,
	}) {
		e.symbols[pkg] = ensureMap(e.symbols[pkg])
		e.symbols[pkg][qn] = id
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

	if body := n.ChildByFieldName("body"); body != nil {
		e.collectCalls(body, src, id, pkg, className, filePath)
	}
}

// ---- import parsing --------------------------------------------------------

// parseImport handles `import foo.bar.Baz;`, `import foo.bar.*;`, and
// `import static foo.bar.Baz.member;`.
func (e *javaExtractor) parseImport(
	n *sitter.Node, src []byte,
	filePath, currentPkg, fileID string,
	imports *javaImportTable,
) {
	// Detect "static" by scanning anonymous children — tree-sitter
	// java exposes it as the `static` keyword token, not a field.
	isStatic := false
	for i := 0; i < int(n.ChildCount()); i++ {
		if c := n.Child(i); c != nil && c.Type() == "static" {
			isStatic = true
			break
		}
	}
	// Detect wildcard via the `*` token. The path itself is in the
	// first scoped_identifier or identifier child.
	isWildcard := false
	var pathNode *sitter.Node
	for i := 0; i < int(n.NamedChildCount()); i++ {
		c := n.NamedChild(i)
		if c == nil {
			continue
		}
		switch c.Type() {
		case "scoped_identifier", "identifier":
			if pathNode == nil {
				pathNode = c
			}
		case "asterisk":
			isWildcard = true
		}
	}
	if pathNode == nil {
		return
	}
	full := nodeText(pathNode, src)
	if full == "" {
		return
	}

	startLine := lineOfPoint(n.StartPoint().Row)
	e.emitJavaImportEdge(full, filePath, currentPkg, fileID, startLine)

	if isWildcard {
		// `import foo.bar.*;` — can't pin individual class names. The
		// import edge gives graph_deps something; symbol resolution
		// will simply miss these (acceptable trade).
		return
	}
	if isStatic {
		// `import static foo.bar.Foo.bar;` — path is foo.bar.Foo.bar.
		// Member is the last component; the class is the segment
		// before it; the package is everything before that.
		mod, member := splitJavaPathTail(full)
		pkgOfCls, clsName := splitJavaPathTail(mod)
		if member != "" && clsName != "" {
			imports.statics[member] = javaStaticImport{
				pkg: pkgOfCls, cls: clsName, memb: member,
			}
		}
		return
	}
	// Regular `import foo.bar.Baz;` — split (pkg, className).
	mod, name := splitJavaPathTail(full)
	if name != "" {
		imports.classes[name] = javaClassImport{pkg: mod, name: name}
	}
}

// emitJavaImportEdge mirrors emitImportEdges in the other extractors.
func (e *javaExtractor) emitJavaImportEdge(
	usePath, filePath, currentPkg, fileID string,
	startLine int,
) {
	impID := NodeID("", currentPkg, NodeImport, usePath)
	e.addNode(Node{
		ID:            impID,
		Kind:          NodeImport,
		Name:          usePath,
		QualifiedName: usePath,
		PackagePath:   currentPkg,
		Metadata:      map[string]any{"language": "java"},
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

func (e *javaExtractor) collectCalls(
	body *sitter.Node, src []byte,
	callerID, callerPkg, callerCls, filePath string,
) {
	var walk func(*sitter.Node)
	walk = func(n *sitter.Node) {
		if n == nil {
			return
		}
		switch n.Type() {
		case "class_declaration", "interface_declaration",
			"enum_declaration", "method_declaration",
			"constructor_declaration", "lambda_expression":
			return
		case "method_invocation":
			expr := classifyJavaInvocation(n, src)
			if expr.kind != "skip" {
				e.pendingCalls = append(e.pendingCalls, javaPendingCall{
					callerID:   callerID,
					callerPkg:  callerPkg,
					callerCls:  callerCls,
					calleeExpr: expr,
					filePath:   filePath,
					line:       lineOfPoint(n.StartPoint().Row),
				})
			}
		case "object_creation_expression":
			expr := classifyJavaNewExpr(n, src)
			if expr.kind != "skip" {
				e.pendingCalls = append(e.pendingCalls, javaPendingCall{
					callerID:   callerID,
					callerPkg:  callerPkg,
					callerCls:  callerCls,
					calleeExpr: expr,
					filePath:   filePath,
					line:       lineOfPoint(n.StartPoint().Row),
				})
			}
		}
		for i := 0; i < int(n.NamedChildCount()); i++ {
			walk(n.NamedChild(i))
		}
	}
	walk(body)
}

func classifyJavaInvocation(n *sitter.Node, src []byte) javaCallee {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		return javaCallee{kind: "skip"}
	}
	name := nodeText(nameNode, src)
	obj := n.ChildByFieldName("object")
	if obj == nil {
		return javaCallee{kind: "bare", parts: []string{name}}
	}
	switch obj.Type() {
	case "this":
		return javaCallee{kind: "self", parts: []string{"this", name}}
	case "super":
		// super.X — bind to "super" so resolveCall can skip cleanly.
		return javaCallee{kind: "self", parts: []string{"super", name}}
	case "identifier":
		return javaCallee{kind: "attr", parts: []string{nodeText(obj, src), name}}
	case "field_access":
		parts := flattenJavaFieldAccess(obj, src)
		if parts == nil {
			return javaCallee{kind: "skip"}
		}
		return javaCallee{kind: "attr", parts: append(parts, name)}
	}
	return javaCallee{kind: "skip"}
}

// classifyJavaNewExpr turns `new Foo(...)` into a class lookup. The
// type may be scoped: `new pkg.Foo()` flattens to ["pkg","Foo"].
func classifyJavaNewExpr(n *sitter.Node, src []byte) javaCallee {
	typeNode := n.ChildByFieldName("type")
	if typeNode == nil {
		return javaCallee{kind: "skip"}
	}
	switch typeNode.Type() {
	case "type_identifier":
		return javaCallee{kind: "new", parts: []string{nodeText(typeNode, src)}}
	case "scoped_type_identifier":
		parts := flattenJavaScopedType(typeNode, src)
		if len(parts) == 0 {
			return javaCallee{kind: "skip"}
		}
		return javaCallee{kind: "new", parts: parts}
	case "generic_type":
		// new Foo<T>() — the actual type identifier is in the first
		// type_identifier child.
		for i := 0; i < int(typeNode.NamedChildCount()); i++ {
			c := typeNode.NamedChild(i)
			if c != nil && c.Type() == "type_identifier" {
				return javaCallee{kind: "new", parts: []string{nodeText(c, src)}}
			}
		}
	}
	return javaCallee{kind: "skip"}
}

// flattenJavaFieldAccess turns `a.b.c` (field_access(field_access(a,b),c))
// into ["a","b","c"]. Returns nil for any chain whose base isn't a
// plain identifier (e.g. method call result, cast, array indexing).
func flattenJavaFieldAccess(n *sitter.Node, src []byte) []string {
	var out []string
	cur := n
	for cur != nil && cur.Type() == "field_access" {
		fieldNode := cur.ChildByFieldName("field")
		obj := cur.ChildByFieldName("object")
		if fieldNode == nil || obj == nil {
			return nil
		}
		out = append([]string{nodeText(fieldNode, src)}, out...)
		cur = obj
	}
	if cur == nil || cur.Type() != "identifier" {
		return nil
	}
	out = append([]string{nodeText(cur, src)}, out...)
	return out
}

// flattenJavaScopedType handles `a.b.C` for scoped_type_identifier.
// Returns the dot-separated segments in source order.
func flattenJavaScopedType(n *sitter.Node, src []byte) []string {
	full := nodeText(n, src)
	if full == "" {
		return nil
	}
	return strings.Split(full, ".")
}

func (e *javaExtractor) resolveCall(c javaPendingCall) string {
	switch c.calleeExpr.kind {
	case "bare":
		name := c.calleeExpr.parts[0]
		// Same-class method dispatch — bare call inside a method
		// resolves to that class's own method first.
		if c.callerCls != "" {
			if id := e.symbolIn(c.callerPkg, c.callerCls+"."+name); id != "" {
				return id
			}
		}
		// Static-imported member: `import static path.Foo.bar` then
		// `bar()` resolves to Foo.bar in that other class.
		imports := e.fileImports[c.filePath]
		if imports != nil {
			if si, ok := imports.statics[name]; ok {
				return e.symbolIn(si.pkg, si.cls+"."+si.memb)
			}
		}
		return ""

	case "self":
		if c.calleeExpr.parts[0] == "super" {
			return "" // no parent-class tracking yet
		}
		if c.callerCls == "" || len(c.calleeExpr.parts) < 2 {
			return ""
		}
		methodName := c.calleeExpr.parts[1]
		return e.symbolIn(c.callerPkg, c.callerCls+"."+methodName)

	case "attr":
		parts := c.calleeExpr.parts
		if len(parts) < 2 {
			return ""
		}
		// Last segment is the callee; everything before is the
		// receiver chain. We resolve only when the head matches an
		// imported class (`ClassName.method()`) or a same-package
		// class.
		method := parts[len(parts)-1]
		recvChain := parts[:len(parts)-1]
		head := recvChain[0]
		imports := e.fileImports[c.filePath]
		if imports != nil {
			if ci, ok := imports.classes[head]; ok {
				if len(recvChain) == 1 {
					return e.symbolIn(ci.pkg, ci.name+"."+method)
				}
				return ""
			}
		}
		// Same-package class: `ClassName.method()` without an import.
		if len(recvChain) == 1 {
			if id := e.symbolIn(c.callerPkg, head+"."+method); id != "" {
				return id
			}
		}
		return ""

	case "new":
		parts := c.calleeExpr.parts
		// Simple case: `new Foo(...)`.
		if len(parts) == 1 {
			name := parts[0]
			// Same-package class first.
			if id := e.symbolIn(c.callerPkg, name); id != "" {
				return id
			}
			// Imported class.
			imports := e.fileImports[c.filePath]
			if imports != nil {
				if ci, ok := imports.classes[name]; ok {
					return e.symbolIn(ci.pkg, ci.name)
				}
			}
			return ""
		}
		// Scoped: `new pkg.Foo()` — last segment is the class, the
		// rest is the package.
		modPath := strings.Join(parts[:len(parts)-1], ".")
		return e.symbolIn(modPath, parts[len(parts)-1])
	}
	return ""
}

func (e *javaExtractor) symbolIn(pkg, name string) string {
	bucket := e.symbols[pkg]
	if bucket == nil {
		return ""
	}
	return bucket[name]
}

// ---- helpers ---------------------------------------------------------------

func (e *javaExtractor) addNode(n Node) bool {
	if _, ok := e.nodeIDs[n.ID]; ok {
		return false
	}
	e.nodeIDs[n.ID] = struct{}{}
	e.nodes = append(e.nodes, n)
	return true
}

// javaPackagePath finds the file's `package` declaration and returns
// its dotted name. Falls back to the directory path (in dot form)
// when no declaration is present (Java's "default package" case).
func javaPackagePath(root *sitter.Node, src []byte, relPath string) string {
	for i := 0; i < int(root.NamedChildCount()); i++ {
		c := root.NamedChild(i)
		if c == nil || c.Type() != "package_declaration" {
			continue
		}
		for j := 0; j < int(c.NamedChildCount()); j++ {
			cc := c.NamedChild(j)
			if cc == nil {
				continue
			}
			if cc.Type() == "scoped_identifier" || cc.Type() == "identifier" {
				return nodeText(cc, src)
			}
		}
	}
	// Fallback: derive from directory path. Strip the common
	// src/main/java prefix Maven/Gradle projects use so a file at
	// src/main/java/com/example/Foo.java reports "com.example".
	dir := filepath.ToSlash(filepath.Dir(relPath))
	for _, prefix := range []string{"src/main/java/", "src/test/java/", "src/"} {
		if strings.HasPrefix(dir+"/", prefix) {
			dir = strings.TrimPrefix(dir+"/", prefix)
			dir = strings.TrimSuffix(dir, "/")
			break
		}
	}
	if dir == "." || dir == "" {
		return ""
	}
	return strings.ReplaceAll(dir, "/", ".")
}

func javaLeafName(pkg string) string {
	if i := strings.LastIndexByte(pkg, '.'); i >= 0 {
		return pkg[i+1:]
	}
	if pkg == "" {
		return "<default>"
	}
	return pkg
}

// splitJavaPathTail splits a dotted path on the last `.`. Returns
// (prefix, lastSegment). For "foo.bar.Baz" yields ("foo.bar", "Baz").
func splitJavaPathTail(p string) (string, string) {
	if i := strings.LastIndexByte(p, '.'); i >= 0 {
		return p[:i], p[i+1:]
	}
	return "", p
}
