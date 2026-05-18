package graph

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/tools/go/packages"
)

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
		extractPackage(p, projectRoot, modulePath, nodeSet, edgeSet)
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

func extractPackage(p *packages.Package, projectRoot, modulePath string, nodes *nodeSet, edges *edgeSet) {
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
		extractFile(p, file, fset, pkgID, projectRoot, modulePath, nodes, edges)
	}
}

func extractFile(
	p *packages.Package,
	file *ast.File,
	fset *token.FileSet,
	pkgID, projectRoot, modulePath string,
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
			extractFunc(p, d, fset, fileID, rel, modulePath, nodes, edges)
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
