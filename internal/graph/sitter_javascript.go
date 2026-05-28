package graph

// JavaScript tree-sitter extractor. Handles .js and .jsx via the
// javascript grammar (mirrors internal/chunk/chunk.go's choice).
//
// Structurally a port of the TypeScript extractor (sitter_ts.go)
// minus the type-only declarations (interface, type alias) the JS
// grammar doesn't surface. Same package-path scheme (file path minus
// extension, slash form), same resolution lanes (bare, this.X,
// namespace member, named/default import, new ClassName), and same
// specifier-resolution probing (.js, .jsx, /index.js) — done via
// extension-stripped packagePath so the lookup is one map check.
//
// Cross-language calls (JS → TS or TS → JS) don't resolve: each
// extractor instance owns its own symbol table. Pure JS or pure TS
// projects are the common case; mixed projects fall back to file
// edges + import edges without callee linkage.

import (
	"context"
	"path"
	"path/filepath"
	"strings"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/javascript"
)

func init() { Register(newJSExtractor) }

func newJSExtractor() Extractor {
	return &jsExtractor{
		nodeIDs:     map[string]struct{}{},
		symbols:     map[string]map[string]string{},
		fileImports: map[string]*tsImportTable{},
		knownFiles:  map[string]string{},
	}
}

type jsExtractor struct {
	projectRoot string

	nodes []Node
	edges []Edge

	nodeIDs map[string]struct{}
	symbols map[string]map[string]string

	fileImports map[string]*tsImportTable
	knownFiles  map[string]string

	pendingCalls []tsPendingCall

	warnings []string
}

// ---- Extractor interface ---------------------------------------------------

func (e *jsExtractor) Name() string               { return "javascript" }
func (e *jsExtractor) Language() *sitter.Language { return javascript.GetLanguage() }
func (e *jsExtractor) Extensions() []string       { return []string{".js", ".jsx"} }

func (e *jsExtractor) Init(_ context.Context, root string) error {
	e.projectRoot = root
	return nil
}

func (e *jsExtractor) ProcessFile(_ context.Context, in FileInput) error {
	pkg := jsPackagePath(in.RelPath)
	e.knownFiles[pkg] = in.RelPath

	pkgID := NodeID("", pkg, NodePackage, pkg)
	e.addNode(Node{
		ID:            pkgID,
		Kind:          NodePackage,
		Name:          path.Base(pkg),
		QualifiedName: pkg,
		PackagePath:   pkg,
		Metadata:      map[string]any{"language": "javascript"},
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
		Metadata:      map[string]any{"language": "javascript"},
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

func (e *jsExtractor) Finalize(_ context.Context) ([]Node, []Edge, []string, error) {
	for _, imp := range e.fileImports {
		for local, target := range imp.modules {
			if local == "__from__" {
				continue
			}
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

func (e *jsExtractor) processTopLevel(
	n *sitter.Node, src []byte,
	filePath, pkg, fileID string, imports *tsImportTable,
) {
	if n == nil {
		return
	}
	kind := n.Type()
	if kind == "export_statement" {
		if decl := n.ChildByFieldName("declaration"); decl != nil {
			e.processTopLevel(decl, src, filePath, pkg, fileID, imports)
			e.maybeMarkDefaultExport(decl, src, pkg)
			return
		}
		return
	}
	switch kind {
	case "function_declaration":
		e.addFunction(n, src, filePath, pkg, fileID, "")
	case "class_declaration":
		e.addClass(n, src, filePath, pkg, fileID)
	case "lexical_declaration", "variable_declaration":
		e.addLexicalDecl(n, src, filePath, pkg, fileID)
	case "import_statement":
		e.parseImportStatement(n, src, filePath, pkg, fileID, imports)
	}
}

func (e *jsExtractor) addFunction(
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
	meta := map[string]any{"language": "javascript"}
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

func (e *jsExtractor) addClass(
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
		Metadata:      map[string]any{"language": "javascript"},
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

func (e *jsExtractor) addLexicalDecl(
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
			Metadata:      map[string]any{"language": "javascript", "form": "arrow"},
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

func (e *jsExtractor) maybeMarkDefaultExport(decl *sitter.Node, src []byte, pkg string) {
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
	if nameNode := decl.ChildByFieldName("name"); nameNode != nil {
		name := nodeText(nameNode, src)
		if id := e.symbolIn(pkg, name); id != "" {
			e.symbols[pkg] = ensureMap(e.symbols[pkg])
			e.symbols[pkg]["default"] = id
		}
	}
}

// ---- import parsing --------------------------------------------------------

func (e *jsExtractor) parseImportStatement(
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

	imports.modules["__from__"] = filePath

	for i := 0; i < int(n.NamedChildCount()); i++ {
		child := n.NamedChild(i)
		if child == nil || child == sourceNode {
			continue
		}
		if child.Type() != "import_clause" {
			continue
		}
		e.processImportClause(child, src, specifier, imports)
	}

	impID := NodeID("", currentPkg, NodeImport, specifier)
	e.addNode(Node{
		ID:            impID,
		Kind:          NodeImport,
		Name:          specifier,
		QualifiedName: specifier,
		PackagePath:   currentPkg,
		Metadata:      map[string]any{"language": "javascript"},
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

func (e *jsExtractor) processImportClause(
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
			local := nodeText(child, src)
			if local != "" {
				imports.fromImports[local] = pyFromImport{pkg: specifier, name: "default"}
			}
		case "namespace_import":
			aliasNode := child.ChildByFieldName("alias")
			if aliasNode == nil {
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

func (e *jsExtractor) collectCalls(
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

func (e *jsExtractor) resolveCall(c tsPendingCall) string {
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
			if len(tail) != 1 {
				return ""
			}
			return e.symbolIn(mod, tail[0])
		}
		if fi, ok := imports.fromImports[head]; ok {
			if len(tail) == 1 {
				return e.symbolIn(fi.pkg, fi.name+"."+tail[0])
			}
			return ""
		}
		return ""
	}
	return ""
}

func (e *jsExtractor) symbolIn(pkg, name string) string {
	bucket := e.symbols[pkg]
	if bucket == nil {
		return ""
	}
	return bucket[name]
}

// resolveModuleSpecifier — JS specifier resolution. Mirrors TS's
// resolver but probes the JS knownFiles map. The package path is
// extension-stripped (`.js`, `.jsx` both collapse to the same key),
// so the same lookup handles both file forms; the dir+`/index`
// fallback covers barrel modules.
func (e *jsExtractor) resolveModuleSpecifier(specifier, fromFile string) string {
	if specifier == "" {
		return ""
	}
	if !strings.HasPrefix(specifier, ".") {
		return specifier
	}
	if fromFile == "" {
		return specifier
	}
	base := path.Dir(filepath.ToSlash(fromFile))
	joined := path.Clean(path.Join(base, specifier))
	if _, ok := e.knownFiles[joined]; ok {
		return joined
	}
	if _, ok := e.knownFiles[joined+"/index"]; ok {
		return joined + "/index"
	}
	return specifier
}

// ---- helpers ---------------------------------------------------------------

func (e *jsExtractor) addNode(n Node) bool {
	if _, ok := e.nodeIDs[n.ID]; ok {
		return false
	}
	e.nodeIDs[n.ID] = struct{}{}
	e.nodes = append(e.nodes, n)
	return true
}

// jsPackagePath strips the trailing `.js` or `.jsx` extension. Files
// without those extensions are returned verbatim so the caller still
// gets a stable key.
func jsPackagePath(relPath string) string {
	p := filepath.ToSlash(relPath)
	for _, ext := range []string{".jsx", ".js"} {
		if strings.HasSuffix(p, ext) {
			return strings.TrimSuffix(p, ext)
		}
	}
	return p
}
