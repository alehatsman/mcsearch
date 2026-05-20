package mcp

// server_graph.go holds the graph-only MCP tools: graph_deps,
// graph_callers, graph_callees. Each handler reads the static graph
// (graph_nodes / graph_edges) via loadGraphView and never touches the
// embedding or chat endpoints — making these the cheapest tools in
// the surface and useful as a precise fallback when semantic search
// drifts.

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/alehatsman/mcsearch/internal/graph"
	"github.com/alehatsman/mcsearch/internal/store"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// ─── tool: graph_deps ─────────────────────────────────────────────────────

type GraphDepsInput struct {
	Path        string `json:"path,omitempty" jsonschema:"relative file path inside the project — resolved to its package"`
	Package     string `json:"package,omitempty" jsonschema:"full package path (e.g. 'github.com/foo/bar/internal/baz'); takes precedence over path"`
	ProjectRoot string `json:"project_root,omitempty" jsonschema:"absolute path to the project root; defaults to the server's working directory"`
}

// GraphDep is a single import relationship: from_package depends on
// to_package. Layer 1 emits one edge per (importing-pkg, imported-pkg)
// pair, so the output is package-grained — not yet file-grained.
type GraphDep struct {
	FromPackage string `json:"from_package"`
	ToPackage   string `json:"to_package"`
}

type GraphDepsOutput struct {
	Status  string     `json:"status"` // "ok" | "no-index" | "no-graph" | "not-found" | "error"
	Hint    string     `json:"hint,omitempty"`
	Project string     `json:"project,omitempty"`
	Package string     `json:"package,omitempty"` // resolved package path the answer is for
	Imports []GraphDep `json:"imports,omitempty"`
}

func (s *Server) graphDeps(ctx context.Context, _ *sdk.CallToolRequest, in GraphDepsInput) (*sdk.CallToolResult, GraphDepsOutput, error) {
	if strings.TrimSpace(in.Path) == "" && strings.TrimSpace(in.Package) == "" {
		return nil, GraphDepsOutput{Status: "error", Hint: "pass `path` (a file inside the project) or `package` (full package path)"}, nil
	}
	p, hint := s.resolveProject(in.ProjectRoot)
	if hint != "" {
		return nil, GraphDepsOutput{Status: "error", Hint: hint}, nil
	}
	if _, err := os.Stat(p.DBPath); errors.Is(err, os.ErrNotExist) {
		return nil, GraphDepsOutput{Status: "no-index", Project: p.Root,
			Hint: fmt.Sprintf("no index for %s — run `mcsearch index %s` first.", p.Root, p.Root)}, nil
	}
	st, err := store.OpenWith(ctx, p.DBPath, s.StoreOpts)
	if err != nil {
		return nil, GraphDepsOutput{Status: "error", Hint: fmt.Sprintf("open index: %v", err)}, nil
	}
	defer st.Close()

	view, err := loadGraphView(ctx, st)
	if err != nil {
		return nil, GraphDepsOutput{Status: "error", Hint: fmt.Sprintf("load graph: %v", err)}, nil
	}
	if view == nil {
		return nil, GraphDepsOutput{Status: "no-graph", Project: p.Root,
			Hint: fmt.Sprintf("graph not indexed for %s — run `mcsearch index %s --graph=only`.", p.Root, p.Root)}, nil
	}

	// Resolve target package. `package` wins over `path` when both are set.
	pkg := strings.TrimSpace(in.Package)
	if pkg == "" {
		// Path → package via nodesByPath. Multiple files share a pkg
		// (and the import graph is per-pkg anyway) so any node in that
		// file is fine.
		nodes := view.nodesByPath[in.Path]
		if len(nodes) == 0 {
			return nil, GraphDepsOutput{Status: "not-found", Project: p.Root,
				Hint: fmt.Sprintf("path %q has no graph nodes — file may be outside the indexed languages (currently Go-only) or not yet indexed", in.Path)}, nil
		}
		for _, n := range nodes {
			if n.PackagePath != "" {
				pkg = n.PackagePath
				break
			}
		}
		if pkg == "" {
			return nil, GraphDepsOutput{Status: "not-found", Project: p.Root,
				Hint: fmt.Sprintf("path %q resolved to no package — likely a file outside any package", in.Path)}, nil
		}
	}

	// Collect import edges where src is the NodePackage for pkg.
	// Layer 1 emits imports at the package node, so this is a single
	// edges-by-src lookup once we have the package node ID.
	var pkgID string
	for _, n := range view.nodesByPackage[pkg] {
		if n.Kind == graph.NodePackage {
			pkgID = n.ID
			break
		}
	}
	if pkgID == "" {
		return nil, GraphDepsOutput{Status: "not-found", Project: p.Root,
			Hint: fmt.Sprintf("package %q has no node in the graph", pkg)}, nil
	}

	out := GraphDepsOutput{Status: "ok", Project: p.Root, Package: pkg}
	for _, e := range view.edgesBySrc[pkgID] {
		if e.Kind != graph.EdgeImports {
			continue
		}
		dst, ok := view.nodesByID[e.DstID]
		if !ok || dst.Kind != graph.NodeImport {
			continue
		}
		out.Imports = append(out.Imports, GraphDep{
			FromPackage: pkg,
			ToPackage:   dst.QualifiedName, // import nodes carry the import path here
		})
	}
	sort.Slice(out.Imports, func(i, j int) bool { return out.Imports[i].ToPackage < out.Imports[j].ToPackage })
	return nil, out, nil
}

// ─── tools: graph_callers / graph_callees ─────────────────────────────────

type CallEdgeInput struct {
	Name        string `json:"name" jsonschema:"symbol to query: bare ('Foo'), receiver-qualified ('(*Server).RunStdio'), or package-tail-qualified ('mcp.NewServer')"`
	Package     string `json:"package,omitempty" jsonschema:"optional package path filter when the same name is defined in multiple packages"`
	ProjectRoot string `json:"project_root,omitempty" jsonschema:"absolute path to the project root; defaults to the server's working directory"`
	K           int    `json:"k,omitempty" jsonschema:"max hits to return (default 12, max 50)"`
}

// CallSite is one calls-edge endpoint — the function on the other end
// of the edge, plus the file:line where the call expression sits.
type CallSite struct {
	QualifiedName string `json:"qualified_name"`
	Package       string `json:"package,omitempty"`
	Kind          string `json:"kind"` // "function" | "method" | "interface_method"
	Path          string `json:"path"`
	StartLine     int    `json:"start_line"`
	EndLine       int    `json:"end_line"`
	CallSitePath  string `json:"call_site_path,omitempty"` // file containing the call expression
	CallSiteLine  int    `json:"call_site_line,omitempty"` // line of the call expression
	// Role tags the peer the same way SearchHit.Role does: how this
	// function sits in the call graph. Empty for unremarkable peers.
	// See formatRole for the threshold/tiering rules.
	Role      string `json:"role,omitempty"`
	Content   string `json:"content,omitempty"`
	Truncated bool   `json:"truncated,omitempty"`
}

// TargetMatch is one resolved interpretation of the input `name`.
// Returned even when there's no calls activity, so the caller can
// disambiguate or confirm the resolution.
type TargetMatch struct {
	QualifiedName string `json:"qualified_name"`
	Package       string `json:"package,omitempty"`
	Kind          string `json:"kind"`
	Path          string `json:"path,omitempty"`
	StartLine     int    `json:"start_line,omitempty"`
}

type CallEdgeOutput struct {
	Status  string        `json:"status"` // "ok" | "no-index" | "no-graph" | "not-found" | "error"
	Hint    string        `json:"hint,omitempty"`
	Project string        `json:"project,omitempty"`
	Targets []TargetMatch `json:"targets,omitempty"`
	Hits    []CallSite    `json:"hits,omitempty"`
}

func (s *Server) graphCallers(ctx context.Context, req *sdk.CallToolRequest, in CallEdgeInput) (*sdk.CallToolResult, CallEdgeOutput, error) {
	return s.callEdges(ctx, in, true)
}

func (s *Server) graphCallees(ctx context.Context, req *sdk.CallToolRequest, in CallEdgeInput) (*sdk.CallToolResult, CallEdgeOutput, error) {
	return s.callEdges(ctx, in, false)
}

// callEdges is the shared body. callers=true walks edgesByDst (incoming
// calls); callers=false walks edgesBySrc (outgoing calls).
func (s *Server) callEdges(ctx context.Context, in CallEdgeInput, callers bool) (*sdk.CallToolResult, CallEdgeOutput, error) {
	if strings.TrimSpace(in.Name) == "" {
		return nil, CallEdgeOutput{Status: "error", Hint: "name is empty"}, nil
	}
	p, hint := s.resolveProject(in.ProjectRoot)
	if hint != "" {
		return nil, CallEdgeOutput{Status: "error", Hint: hint}, nil
	}
	if _, err := os.Stat(p.DBPath); errors.Is(err, os.ErrNotExist) {
		return nil, CallEdgeOutput{Status: "no-index", Project: p.Root,
			Hint: fmt.Sprintf("no index for %s — run `mcsearch index %s` first.", p.Root, p.Root)}, nil
	}
	st, err := store.OpenWith(ctx, p.DBPath, s.StoreOpts)
	if err != nil {
		return nil, CallEdgeOutput{Status: "error", Hint: fmt.Sprintf("open index: %v", err)}, nil
	}
	defer st.Close()

	view, err := loadGraphView(ctx, st)
	if err != nil {
		return nil, CallEdgeOutput{Status: "error", Hint: fmt.Sprintf("load graph: %v", err)}, nil
	}
	if view == nil {
		return nil, CallEdgeOutput{Status: "no-graph", Project: p.Root,
			Hint: fmt.Sprintf("graph not indexed for %s — run `mcsearch index %s --graph=only`.", p.Root, p.Root)}, nil
	}
	if len(view.edgesByKind[graph.EdgeCalls]) == 0 {
		return nil, CallEdgeOutput{Status: "no-graph", Project: p.Root,
			Hint: "graph has no `calls` edges — reindex the project with this release (`mcsearch index . --graph=only`) to extract them."}, nil
	}

	targets := resolveCallTargets(view, in.Name, in.Package)
	if len(targets) == 0 {
		return nil, CallEdgeOutput{Status: "not-found", Project: p.Root,
			Hint: fmt.Sprintf("no graph node matches name=%q — try the bare identifier or the receiver-qualified form like '(*Type).Method'", in.Name)}, nil
	}

	k := in.K
	if k <= 0 {
		k = 12
	}
	if k > 50 {
		k = 50
	}

	out := CallEdgeOutput{Status: "ok", Project: p.Root}
	for _, t := range targets {
		out.Targets = append(out.Targets, TargetMatch{
			QualifiedName: t.QualifiedName,
			Package:       t.PackagePath,
			Kind:          string(t.Kind),
			Path:          t.FilePath,
			StartLine:     t.StartLine,
		})
	}

	seen := map[string]bool{}
	for _, t := range targets {
		var edges []graphEdge
		if callers {
			edges = view.edgesByDst[t.ID]
		} else {
			edges = view.edgesBySrc[t.ID]
		}
		for _, e := range edges {
			if e.Kind != graph.EdgeCalls {
				continue
			}
			peerID := e.SrcID
			if !callers {
				peerID = e.DstID
			}
			peer, ok := view.nodesByID[peerID]
			if !ok {
				continue
			}
			// Dedup on (peer node id, call-site file+line). Different
			// call sites from the same caller are distinct hits.
			key := peer.ID + "@" + e.FilePath + ":" + fmt.Sprint(e.StartLine)
			if seen[key] {
				continue
			}
			seen[key] = true
			hit := CallSite{
				QualifiedName: peer.QualifiedName,
				Package:       peer.PackagePath,
				Kind:          string(peer.Kind),
				Path:          peer.FilePath,
				StartLine:     peer.StartLine,
				EndLine:       peer.EndLine,
				CallSitePath:  e.FilePath,
				CallSiteLine:  e.StartLine,
				Role:          formatRole(peer.Name, peer.InDegree, peer.OutDegree, peer.CrossPkgCallers),
			}
			out.Hits = append(out.Hits, hit)
			if len(out.Hits) >= k {
				break
			}
		}
		if len(out.Hits) >= k {
			break
		}
	}

	// Sort hits by peer centrality, then by path/line for determinism.
	// peerCentrality is a closure over view.nodesByID so we don't
	// re-resolve per hit. PageRank dominates; in_degree breaks ties
	// for peers that didn't pick up rank (e.g. callees with no
	// incoming edges in the indexed slice).
	peerCentrality := func(h CallSite) (float64, int) {
		// Resolve peer node by qualified name + package — the same key
		// we used when populating the hit.
		for _, n := range view.nodesByQualified[h.QualifiedName] {
			if n.PackagePath == h.Package {
				return n.PageRank, n.InDegree
			}
		}
		for _, n := range view.nodesByName[h.QualifiedName] {
			if n.PackagePath == h.Package {
				return n.PageRank, n.InDegree
			}
		}
		return 0, 0
	}
	sort.SliceStable(out.Hits, func(i, j int) bool {
		ai, aj := out.Hits[i], out.Hits[j]
		pi, di := peerCentrality(ai)
		pj, dj := peerCentrality(aj)
		if pi != pj {
			return pi > pj
		}
		if di != dj {
			return di > dj
		}
		if ai.Path != aj.Path {
			return ai.Path < aj.Path
		}
		if ai.StartLine != aj.StartLine {
			return ai.StartLine < aj.StartLine
		}
		return ai.CallSiteLine < aj.CallSiteLine
	})

	// Inline a short slice of each hit's containing function so the
	// agent doesn't need a follow-up Read for context. Same shape as
	// inlineContent's per-read budget for targeted intents.
	const (
		maxHitLines = 30
		maxHitBytes = 2 * 1024
	)
	for i := range out.Hits {
		abs := out.Hits[i].Path
		if !filepath.IsAbs(abs) {
			abs = filepath.Join(p.Root, abs)
		}
		content, truncated, err := readLineRange(abs, out.Hits[i].StartLine, out.Hits[i].EndLine, maxHitLines, maxHitBytes)
		if err == nil {
			out.Hits[i].Content = content
			out.Hits[i].Truncated = truncated
		}
	}

	return nil, out, nil
}

// resolveCallTargets maps the user-supplied `name` (and optional pkg
// filter) onto graph nodes. Recognised shapes, in order:
//
//	"Foo"                  — bare; matches NodeFunction / NodeMethod / NodeType by Name
//	"(*T).Foo" / "T.Foo"   — receiver-qualified; matches by QualifiedName
//	"pkg.Foo"              — package-tail-qualified; PackagePath must end with /pkg or equal pkg
//
// Multiple matches are returned so the caller can disambiguate. The
// optional `pkgFilter` collapses ambiguity by full package path.
func resolveCallTargets(view *graphView, name, pkgFilter string) []graphNode {
	name = strings.TrimSpace(name)
	pkgFilter = strings.TrimSpace(pkgFilter)
	if name == "" {
		return nil
	}
	want := func(n graphNode) bool {
		switch n.Kind {
		case graph.NodeFunction, graph.NodeMethod:
			return true
		default:
			return false
		}
	}
	pkgOK := func(n graphNode) bool {
		if pkgFilter == "" {
			return true
		}
		return n.PackagePath == pkgFilter
	}
	out := []graphNode{}
	seen := map[string]bool{}
	add := func(n graphNode) {
		if seen[n.ID] || !want(n) || !pkgOK(n) {
			return
		}
		seen[n.ID] = true
		out = append(out, n)
	}

	// 1) Exact QualifiedName match — covers "(*T).Foo", "T.Foo", and
	//    bare function names that happen to be unique within a pkg.
	for _, n := range view.nodesByQualified[name] {
		add(n)
	}
	// 2) Bare Name match — covers "Foo" both as a function name and
	//    as the method portion of "(*T).Foo" (graph stores Name="Foo"
	//    alongside QualifiedName="(*T).Foo").
	for _, n := range view.nodesByName[name] {
		add(n)
	}
	if len(out) > 0 {
		return out
	}

	// 3) "pkg.Foo" — split on the last dot and try pkg-tail matching.
	//    Only attempt when there's exactly one dot and the second
	//    segment looks like an identifier (no receiver parens).
	if i := strings.LastIndex(name, "."); i > 0 && !strings.ContainsAny(name, "()*") {
		pkgTail, bare := name[:i], name[i+1:]
		for _, n := range view.nodesByName[bare] {
			tail := n.PackagePath
			if j := strings.LastIndex(tail, "/"); j >= 0 {
				tail = tail[j+1:]
			}
			if tail == pkgTail {
				add(n)
			}
		}
	}
	return out
}
