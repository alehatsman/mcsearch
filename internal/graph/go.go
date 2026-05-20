package graph

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/tools/go/packages"
)

// hasGoModule walks up from start looking for a go.mod (or go.work).
// packages.Load("./...") emits a confusing "directory prefix . does not
// contain main module" warning when neither exists; short-circuit to
// keep the warnings list useful on non-Go projects.
func hasGoModule(start string) bool {
	dir := start
	for {
		for _, name := range []string{"go.mod", "go.work"} {
			if _, err := os.Stat(filepath.Join(dir, name)); err == nil {
				return true
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return false
		}
		dir = parent
	}
}

// ExtractResult is the output of a single Go-graph extraction pass.
// Warnings collects per-package load errors so the CLI can surface
// them without aborting the index pass — a half-broken project should
// still yield a partial graph.
type ExtractResult struct {
	Packages int
	Nodes    []Node
	Edges    []Edge
	Warnings []string
}

// ExtractGo loads every Go package under projectRoot (./...) and emits
// the structural nodes/edges defined in this layer:
//
//	nodes:  package, file, function, method, type (struct|interface|other), field, import
//	edges:  contains  (package → file, file → func/method/type, struct → field, interface → method)
//	        imports   (package → import)
//	        has_method (type/interface → method)
//	        has_field  (struct → field)
//	        embeds     (struct → embedded type)
//
// The function never panics on broken packages: packages.Load returns
// per-package errors in pkg.Errors, which become Warnings. Only a
// top-level Load failure (config / build system) is returned as err.
func ExtractGo(ctx context.Context, projectRoot string) (*ExtractResult, error) {
	if !hasGoModule(projectRoot) {
		return &ExtractResult{}, nil
	}
	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedSyntax |
			packages.NeedTypes | packages.NeedTypesInfo | packages.NeedImports |
			packages.NeedDeps | packages.NeedModule,
		Dir:     projectRoot,
		Context: ctx,
	}
	pkgs, err := packages.Load(cfg, "./...")
	if err != nil {
		return nil, fmt.Errorf("packages.Load: %w", err)
	}

	res := &ExtractResult{}
	modulePath := inferModulePath(pkgs)

	// nodeSet dedupes nodes that get emitted from multiple sites
	// (e.g. a type referenced by several methods). Map preserves
	// insertion order via the index, then we flatten into a slice.
	nodeSet := newNodeSet()
	edgeSet := newEdgeSet()

	// inTree bounds calls-edge resolution to the project's own
	// packages. Without this we'd synthesize dst node IDs for std-lib
	// and external deps that have no corresponding graph_nodes row —
	// loadGraphView would fail to dereference them and the agent
	// would see dangling edges.
	inTree := make(map[string]bool, len(pkgs))
	for _, p := range pkgs {
		if p.PkgPath != "" {
			inTree[p.PkgPath] = true
		}
	}

	for _, p := range pkgs {
		for _, e := range p.Errors {
			res.Warnings = append(res.Warnings, fmt.Sprintf("%s: %s", p.PkgPath, e.Msg))
		}
		// Skip packages with no PkgPath (the synthetic command-line-arguments
		// pseudo-package, or a Load failure leaving the entry hollow).
		if p.PkgPath == "" {
			continue
		}
		res.Packages++
		extractPackage(p, projectRoot, modulePath, inTree, nodeSet, edgeSet)
	}

	res.Nodes = nodeSet.flatten()
	res.Edges = edgeSet.flatten()
	return res, nil
}

// inferModulePath picks the most likely module path for the project.
// pkgs[i].Module is nil for std-only loads or when GOFLAGS/build flags
// suppress module info, so we fall back to the first non-empty value
// or the empty string (callers tolerate that — node IDs stay stable
// within a project even without a module path).
func inferModulePath(pkgs []*packages.Package) string {
	for _, p := range pkgs {
		if p.Module != nil && p.Module.Path != "" {
			return p.Module.Path
		}
	}
	return ""
}

func extractPackage(p *packages.Package, projectRoot, modulePath string, inTree map[string]bool, nodes *nodeSet, edges *edgeSet) {
	pkgID := NodeID(modulePath, p.PkgPath, NodePackage, p.PkgPath)
	nodes.add(Node{
		ID:            pkgID,
		Kind:          NodePackage,
		Name:          p.Name,
		QualifiedName: p.PkgPath,
		PackagePath:   p.PkgPath,
	})

	// Imports. Iterate in a deterministic order so two runs produce the
	// same edge IDs (EdgeID uses start_line=0 for these, so duplicates
	// across imports would collide on PK — dedup via edgeSet covers it
	// regardless).
	importPaths := make([]string, 0, len(p.Imports))
	for ip := range p.Imports {
		importPaths = append(importPaths, ip)
	}
	sort.Strings(importPaths)
	for _, ip := range importPaths {
		impID := NodeID(modulePath, p.PkgPath, NodeImport, ip)
		nodes.add(Node{
			ID:            impID,
			Kind:          NodeImport,
			Name:          ip,
			QualifiedName: ip,
			PackagePath:   p.PkgPath,
		})
		edges.add(Edge{
			ID:    EdgeID(pkgID, EdgeImports, impID, "", 0),
			Kind:  EdgeImports,
			SrcID: pkgID,
			DstID: impID,
		})
	}

	if p.Fset == nil {
		return
	}
	fset := p.Fset

	for _, file := range p.Syntax {
		extractFile(p, file, fset, pkgID, projectRoot, modulePath, inTree, nodes, edges)
	}

	extractImplements(p, modulePath, nodes, edges)
}

func extractFile(
	p *packages.Package,
	file *ast.File,
	fset *token.FileSet,
	pkgID, projectRoot, modulePath string,
	inTree map[string]bool,
	nodes *nodeSet, edges *edgeSet,
) {
	pos := fset.Position(file.Pos())
	abs := pos.Filename
	if abs == "" {
		return
	}
	rel, err := filepath.Rel(projectRoot, abs)
	if err != nil || strings.HasPrefix(rel, "..") {
		// Outside the project tree (cgo-generated, GOROOT). Don't
		// pollute the graph with paths the caller can't navigate to.
		return
	}
	endPos := fset.Position(file.End())

	fileQN := p.PkgPath + "/" + filepath.Base(rel)
	fileID := NodeID(modulePath, p.PkgPath, NodeFile, fileQN)
	nodes.add(Node{
		ID:            fileID,
		Kind:          NodeFile,
		Name:          filepath.Base(rel),
		QualifiedName: fileQN,
		PackagePath:   p.PkgPath,
		FilePath:      rel,
		StartLine:     pos.Line,
		EndLine:       endPos.Line,
	})
	edges.add(Edge{
		ID:    EdgeID(pkgID, EdgeContains, fileID, rel, pos.Line),
		Kind:  EdgeContains,
		SrcID: pkgID,
		DstID: fileID,
		// File-level contains carries no start line of its own beyond the file's; leave 0.
	})

	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			extractFunc(p, d, fset, fileID, rel, modulePath, inTree, nodes, edges)
		case *ast.GenDecl:
			if d.Tok == token.TYPE {
				extractTypes(p, d, fset, fileID, rel, modulePath, nodes, edges)
			}
		}
	}
}

func extractFunc(
	p *packages.Package,
	fn *ast.FuncDecl,
	fset *token.FileSet,
	fileID, rel, modulePath string,
	inTree map[string]bool,
	nodes *nodeSet, edges *edgeSet,
) {
	startLine := fset.Position(fn.Pos()).Line
	endLine := fset.Position(fn.End()).Line
	name := fn.Name.Name

	if fn.Recv == nil || len(fn.Recv.List) == 0 {
		// Top-level function.
		qn := name
		fnID := NodeID(modulePath, p.PkgPath, NodeFunction, qn)
		nodes.add(Node{
			ID:            fnID,
			Kind:          NodeFunction,
			Name:          name,
			QualifiedName: qn,
			PackagePath:   p.PkgPath,
			FilePath:      rel,
			StartLine:     startLine,
			EndLine:       endLine,
		})
		edges.add(Edge{
			ID:        EdgeID(fileID, EdgeContains, fnID, rel, startLine),
			Kind:      EdgeContains,
			SrcID:     fileID,
			DstID:     fnID,
			FilePath:  rel,
			StartLine: startLine,
			EndLine:   endLine,
		})
		extractCalls(p, fn.Body, fset, fnID, rel, modulePath, inTree, edges)
		return
	}

	// Method: receiver type spelled with optional pointer prefix.
	recv := fn.Recv.List[0].Type
	recvName, isPtr := recvTypeName(recv)
	if recvName == "" {
		return // unparseable receiver — skip rather than emit garbage
	}
	recvDisplay := recvName
	if isPtr {
		recvDisplay = "*" + recvName
	}
	qn := "(" + recvDisplay + ")." + name
	methodID := NodeID(modulePath, p.PkgPath, NodeMethod, qn)
	nodes.add(Node{
		ID:            methodID,
		Kind:          NodeMethod,
		Name:          name,
		QualifiedName: qn,
		PackagePath:   p.PkgPath,
		FilePath:      rel,
		StartLine:     startLine,
		EndLine:       endLine,
		Metadata: map[string]any{
			"receiver":         recvName,
			"receiver_pointer": isPtr,
		},
	})
	edges.add(Edge{
		ID:        EdgeID(fileID, EdgeContains, methodID, rel, startLine),
		Kind:      EdgeContains,
		SrcID:     fileID,
		DstID:     methodID,
		FilePath:  rel,
		StartLine: startLine,
		EndLine:   endLine,
	})
	// has_method: type → method. The type itself may live in another file
	// or even another package, but the qualified name we built encodes
	// the same package as the receiver's syntactic name. Cross-package
	// embedded receivers don't exist in Go (methods can only be defined
	// on locally-declared types), so this is correct.
	typeQN := recvName
	typeID := NodeID(modulePath, p.PkgPath, NodeType, typeQN)
	edges.add(Edge{
		ID:        EdgeID(typeID, EdgeHasMethod, methodID, rel, startLine),
		Kind:      EdgeHasMethod,
		SrcID:     typeID,
		DstID:     methodID,
		FilePath:  rel,
		StartLine: startLine,
		EndLine:   endLine,
	})
	extractCalls(p, fn.Body, fset, methodID, rel, modulePath, inTree, edges)
}

// extractCalls walks the function body and emits one `calls` edge per
// resolvable call expression. Idempotent — re-extracting the same
// source yields the same edge IDs (start_line is the call expression's
// line, so unchanged call sites collide on PK).
//
// What gets resolved:
//   - Bare `Foo()` in the same package      → function node in p.PkgPath
//   - `pkg.Foo()` for an imported package   → function node in pkg
//   - `x.Method()` (method on a value/ptr)  → method node on the
//                                              receiver type's package
//   - `iface.Method()`                       → interface-method node
//
// What gets skipped (silently — these are not graph-noteworthy):
//   - Builtins (`len`, `make`, `new`, `append`, `panic`, …)
//   - Calls into packages outside the project (inTree miss)
//   - Function-typed fields / variables (no *types.Func resolution)
//   - Type conversions that happen to syntactically look like calls
func extractCalls(
	p *packages.Package,
	body *ast.BlockStmt,
	fset *token.FileSet,
	srcID, rel, modulePath string,
	inTree map[string]bool,
	edges *edgeSet,
) {
	if body == nil || p.TypesInfo == nil {
		return
	}
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		dstID := resolveCallee(p, call, modulePath, inTree)
		if dstID == "" {
			return true
		}
		line := fset.Position(call.Pos()).Line
		edges.add(Edge{
			ID:        EdgeID(srcID, EdgeCalls, dstID, rel, line),
			Kind:      EdgeCalls,
			SrcID:     srcID,
			DstID:     dstID,
			FilePath:  rel,
			StartLine: line,
			EndLine:   line,
		})
		return true
	})
}

// resolveCallee maps a *ast.CallExpr to a graph node ID, or "" if the
// callee is unresolvable / out of project scope. Mirrors the QN
// encoding used by extractFunc and extractInterfaceMethods so the
// edges line up with the dst nodes those emit.
func resolveCallee(p *packages.Package, call *ast.CallExpr, modulePath string, inTree map[string]bool) string {
	switch fun := call.Fun.(type) {

	case *ast.Ident:
		// Bare call: same-package function, dot-imported function, or builtin.
		obj := p.TypesInfo.Uses[fun]
		fn, ok := obj.(*types.Func)
		if !ok {
			return ""
		}
		if fn.Pkg() == nil {
			return "" // builtin
		}
		pkgPath := fn.Pkg().Path()
		if !inTree[pkgPath] {
			return ""
		}
		// Methods are never spelled bare (Go disallows it outside the
		// owning type's methods, where the method's receiver isn't
		// implicit at the call expression). So a *types.Func resolved
		// from a bare identifier is a top-level function.
		return NodeID(modulePath, pkgPath, NodeFunction, fn.Name())

	case *ast.SelectorExpr:
		// SelectorExpr covers two distinct call shapes:
		//   1) sel := TypesInfo.Selections[fun]  — method or field
		//      access on a value/ptr (x.M(), iface.M(), T.M for method
		//      expressions). When sel.Obj() is a *types.Func, it's a
		//      method call.
		//   2) Otherwise, fun.X is the imported package name and
		//      fun.Sel is the function within it.
		if sel := p.TypesInfo.Selections[fun]; sel != nil {
			fn, ok := sel.Obj().(*types.Func)
			if !ok {
				return ""
			}
			sig, ok := fn.Type().(*types.Signature)
			if !ok || sig.Recv() == nil {
				return ""
			}
			recv := sig.Recv().Type()
			isPtr := false
			if ptr, isP := recv.(*types.Pointer); isP {
				isPtr = true
				recv = ptr.Elem()
			}
			named, ok := recv.(*types.Named)
			if !ok {
				return ""
			}
			obj := named.Obj()
			if obj == nil || obj.Pkg() == nil {
				return ""
			}
			pkgPath := obj.Pkg().Path()
			if !inTree[pkgPath] {
				return ""
			}
			recvName := obj.Name()
			recvDisplay := recvName
			// Interface methods are encoded *without* the pointer
			// prefix (extractInterfaceMethods produces "(T).M"); for
			// concrete types we keep the pointer marker so the QN
			// matches what extractFunc emits.
			if _, isIface := named.Underlying().(*types.Interface); !isIface && isPtr {
				recvDisplay = "*" + recvName
			}
			qn := "(" + recvDisplay + ")." + fn.Name()
			return NodeID(modulePath, pkgPath, NodeMethod, qn)
		}
		// Package-qualified call: fun.X resolves to a PkgName.
		funObj, ok := p.TypesInfo.Uses[fun.Sel].(*types.Func)
		if !ok {
			return ""
		}
		if funObj.Pkg() == nil {
			return ""
		}
		pkgPath := funObj.Pkg().Path()
		if !inTree[pkgPath] {
			return ""
		}
		return NodeID(modulePath, pkgPath, NodeFunction, funObj.Name())
	}
	return ""
}

// recvTypeName extracts the receiver type name and whether it's a
// pointer receiver. Returns "" on unrecognized shapes (parameterized
// receivers from generics also land here; layer-1 ignores them).
func recvTypeName(expr ast.Expr) (string, bool) {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name, false
	case *ast.StarExpr:
		if id, ok := t.X.(*ast.Ident); ok {
			return id.Name, true
		}
		// Generic pointer receiver: *T[U]. The X is *ast.IndexExpr / *ast.IndexListExpr.
		if name, ok := indexBase(t.X); ok {
			return name, true
		}
	case *ast.IndexExpr, *ast.IndexListExpr:
		if name, ok := indexBase(t); ok {
			return name, false
		}
	}
	return "", false
}

func indexBase(expr ast.Expr) (string, bool) {
	switch t := expr.(type) {
	case *ast.IndexExpr:
		if id, ok := t.X.(*ast.Ident); ok {
			return id.Name, true
		}
	case *ast.IndexListExpr:
		if id, ok := t.X.(*ast.Ident); ok {
			return id.Name, true
		}
	}
	return "", false
}

func extractTypes(
	p *packages.Package,
	gd *ast.GenDecl,
	fset *token.FileSet,
	fileID, rel, modulePath string,
	nodes *nodeSet, edges *edgeSet,
) {
	for _, spec := range gd.Specs {
		ts, ok := spec.(*ast.TypeSpec)
		if !ok {
			continue
		}
		startLine := fset.Position(ts.Pos()).Line
		endLine := fset.Position(ts.End()).Line
		name := ts.Name.Name
		qn := name

		kind := NodeType
		switch ts.Type.(type) {
		case *ast.StructType:
			kind = NodeStruct
		case *ast.InterfaceType:
			kind = NodeInterface
		}
		typeID := NodeID(modulePath, p.PkgPath, NodeType, qn)
		nodes.add(Node{
			ID:            typeID,
			Kind:          kind,
			Name:          name,
			QualifiedName: qn,
			PackagePath:   p.PkgPath,
			FilePath:      rel,
			StartLine:     startLine,
			EndLine:       endLine,
		})
		edges.add(Edge{
			ID:        EdgeID(fileID, EdgeContains, typeID, rel, startLine),
			Kind:      EdgeContains,
			SrcID:     fileID,
			DstID:     typeID,
			FilePath:  rel,
			StartLine: startLine,
			EndLine:   endLine,
		})

		switch ut := ts.Type.(type) {
		case *ast.StructType:
			extractStructFields(p, ut, fset, typeID, name, rel, modulePath, nodes, edges)
		case *ast.InterfaceType:
			extractInterfaceMethods(p, ut, fset, typeID, name, rel, modulePath, nodes, edges)
		}
	}
}

func extractStructFields(
	p *packages.Package,
	st *ast.StructType,
	fset *token.FileSet,
	typeID, typeName, rel, modulePath string,
	nodes *nodeSet, edges *edgeSet,
) {
	if st.Fields == nil {
		return
	}
	for _, fld := range st.Fields.List {
		startLine := fset.Position(fld.Pos()).Line
		endLine := fset.Position(fld.End()).Line
		if len(fld.Names) == 0 {
			// Embedded field.
			embName, _ := recvTypeName(fld.Type) // reuse: handles T, *T, T[U]
			if embName == "" {
				continue
			}
			// The embedded type may live in a different package, but we
			// only resolve names syntactically here. Type-info-based
			// resolution lands with the cross-package implements/embeds
			// work in layer 2.
			dstID := NodeID(modulePath, p.PkgPath, NodeType, embName)
			edges.add(Edge{
				ID:        EdgeID(typeID, EdgeEmbeds, dstID, rel, startLine),
				Kind:      EdgeEmbeds,
				SrcID:     typeID,
				DstID:     dstID,
				FilePath:  rel,
				StartLine: startLine,
				EndLine:   endLine,
				Metadata: map[string]any{
					"embedded_name": embName,
				},
			})
			continue
		}
		for _, n := range fld.Names {
			fieldQN := typeName + "." + n.Name
			fieldID := NodeID(modulePath, p.PkgPath, NodeField, fieldQN)
			nodes.add(Node{
				ID:            fieldID,
				Kind:          NodeField,
				Name:          n.Name,
				QualifiedName: fieldQN,
				PackagePath:   p.PkgPath,
				FilePath:      rel,
				StartLine:     startLine,
				EndLine:       endLine,
			})
			edges.add(Edge{
				ID:        EdgeID(typeID, EdgeHasField, fieldID, rel, startLine),
				Kind:      EdgeHasField,
				SrcID:     typeID,
				DstID:     fieldID,
				FilePath:  rel,
				StartLine: startLine,
				EndLine:   endLine,
			})
		}
	}
}

func extractInterfaceMethods(
	p *packages.Package,
	it *ast.InterfaceType,
	fset *token.FileSet,
	typeID, typeName, rel, modulePath string,
	nodes *nodeSet, edges *edgeSet,
) {
	if it.Methods == nil {
		return
	}
	for _, m := range it.Methods.List {
		if len(m.Names) == 0 {
			// Embedded interface — emit as embeds edge.
			embName, _ := recvTypeName(m.Type)
			if embName == "" {
				continue
			}
			dstID := NodeID(modulePath, p.PkgPath, NodeType, embName)
			startLine := fset.Position(m.Pos()).Line
			edges.add(Edge{
				ID:        EdgeID(typeID, EdgeEmbeds, dstID, rel, startLine),
				Kind:      EdgeEmbeds,
				SrcID:     typeID,
				DstID:     dstID,
				FilePath:  rel,
				StartLine: startLine,
				Metadata: map[string]any{
					"embedded_name": embName,
					"on_interface":  true,
				},
			})
			continue
		}
		for _, n := range m.Names {
			startLine := fset.Position(m.Pos()).Line
			endLine := fset.Position(m.End()).Line
			qn := "(" + typeName + ")." + n.Name
			methodID := NodeID(modulePath, p.PkgPath, NodeMethod, qn)
			nodes.add(Node{
				ID:            methodID,
				Kind:          NodeMethod,
				Name:          n.Name,
				QualifiedName: qn,
				PackagePath:   p.PkgPath,
				FilePath:      rel,
				StartLine:     startLine,
				EndLine:       endLine,
				Metadata: map[string]any{
					"on_interface": true,
				},
			})
			edges.add(Edge{
				ID:        EdgeID(typeID, EdgeHasMethod, methodID, rel, startLine),
				Kind:      EdgeHasMethod,
				SrcID:     typeID,
				DstID:     methodID,
				FilePath:  rel,
				StartLine: startLine,
				EndLine:   endLine,
			})
		}
	}
}

// extractImplements uses type-checker info to emit implements edges for
// every concrete type in the package that satisfies a same-package interface.
// Both value (T) and pointer (*T) are tested so pointer-receiver method sets
// are correctly matched against interfaces.
func extractImplements(p *packages.Package, modulePath string, _ *nodeSet, edges *edgeSet) {
	if p.TypesInfo == nil || p.Types == nil {
		return
	}
	scope := p.Types.Scope()

	var concretes []*types.Named
	var ifaces []*types.Named
	for _, name := range scope.Names() {
		obj, ok := scope.Lookup(name).(*types.TypeName)
		if !ok {
			continue
		}
		named, ok := obj.Type().(*types.Named)
		if !ok {
			continue
		}
		if named.TypeParams() != nil {
			continue // skip generics
		}
		if _, ok := named.Underlying().(*types.Interface); ok {
			ifaces = append(ifaces, named)
		} else {
			concretes = append(concretes, named)
		}
	}

	for _, conc := range concretes {
		concType := conc.Obj().Type()
		ptrType := types.NewPointer(concType)
		for _, iface := range ifaces {
			ifaceT, ok := iface.Underlying().(*types.Interface)
			if !ok {
				continue
			}
			if !types.Implements(concType, ifaceT) && !types.Implements(ptrType, ifaceT) {
				continue
			}
			concName := conc.Obj().Name()
			ifaceName := iface.Obj().Name()
			concID := NodeID(modulePath, p.PkgPath, NodeType, concName)
			ifaceID := NodeID(modulePath, p.PkgPath, NodeType, ifaceName)
			edges.add(Edge{
				ID:    EdgeID(concID, EdgeImplements, ifaceID, "", 0),
				Kind:  EdgeImplements,
				SrcID: concID,
				DstID: ifaceID,
			})
		}
	}
}

// nodeSet / edgeSet keep insertion order while deduping by primary key.
// Order matters for tests (deterministic outputs) and exporter logs.

type nodeSet struct {
	order []string
	byID  map[string]Node
}

func newNodeSet() *nodeSet { return &nodeSet{byID: map[string]Node{}} }

func (s *nodeSet) add(n Node) {
	if _, ok := s.byID[n.ID]; ok {
		return
	}
	s.byID[n.ID] = n
	s.order = append(s.order, n.ID)
}

func (s *nodeSet) flatten() []Node {
	out := make([]Node, 0, len(s.order))
	for _, id := range s.order {
		out = append(out, s.byID[id])
	}
	return out
}

type edgeSet struct {
	order []string
	byID  map[string]Edge
}

func newEdgeSet() *edgeSet { return &edgeSet{byID: map[string]Edge{}} }

func (s *edgeSet) add(e Edge) {
	if _, ok := s.byID[e.ID]; ok {
		return
	}
	s.byID[e.ID] = e
	s.order = append(s.order, e.ID)
}

func (s *edgeSet) flatten() []Edge {
	out := make([]Edge, 0, len(s.order))
	for _, id := range s.order {
		out = append(out, s.byID[id])
	}
	return out
}
