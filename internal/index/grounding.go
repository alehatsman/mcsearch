package index

import (
	"context"
	"os"
	"path/filepath"
	"strings"

	"github.com/alehatsman/dex/internal/store"
)

// pkgGrounding is the graph-derived ground-truth context fed to
// summarizePackage. The zero value is a valid "no grounding" call —
// callers that can't (or don't) read the graph fall back to the
// ungrounded prompt rather than aborting.
type pkgGrounding struct {
	// Symbols holds the package's exported declarations from graph_nodes.
	// The prompt uses this list as a closed set: backtick-wrapped names
	// in the model's output must come from here.
	Symbols []string
	// ProjectImports holds imports of this package trimmed to the
	// project's module prefix (external/stdlib drops out). The prompt
	// uses this to constrain cross-package claims.
	ProjectImports []string
}

func (g pkgGrounding) empty() bool {
	return len(g.Symbols) == 0 && len(g.ProjectImports) == 0
}

// repoPkgGrounding is one row in repoGrounding: a directory plus a few
// of its top-PageRank symbols. Kept short so the inputs to summarizeRepo
// don't blow past the model's context.
type repoPkgGrounding struct {
	Dir        string
	TopSymbols []string
}

// repoGrounding feeds summarizeRepo. Empty Packages means no
// per-package grounding section is rendered.
type repoGrounding struct {
	Packages []repoPkgGrounding
}

func (g repoGrounding) empty() bool { return len(g.Packages) == 0 }

// fetchPackageGrounding pulls the per-dir exported symbols and the
// project-trimmed import list from the graph. Failures are
// best-effort — an empty grounding still produces a valid (ungrounded)
// prompt rather than blocking the summary cascade.
func (ix *Indexer) fetchPackageGrounding(ctx context.Context, dir, modPath string) pkgGrounding {
	var g pkgGrounding
	if syms, err := ix.Store.ExportedSymbolsByDir(ctx, dir); err == nil {
		g.Symbols = uniqueSymbolNames(syms)
	}
	if imps, err := ix.Store.ImportsForDir(ctx, dir); err == nil {
		g.ProjectImports = projectImports(imps, modPath)
	}
	return g
}

// fetchRepoGrounding builds one repoPkgGrounding per package directory
// using the same top-by-centrality cut that topRepoSummaryInput uses.
// Caller supplies the dir list (parallel to the pkgSummaries slice it
// passes to summarizeRepo) so the grounding stays aligned with the
// summary order the model sees.
func (ix *Indexer) fetchRepoGrounding(ctx context.Context, dirs []string) repoGrounding {
	const topK = 5
	var g repoGrounding
	for _, dir := range dirs {
		syms, err := ix.Store.TopCentralByDir(ctx, dir, topK, true)
		if err != nil || len(syms) == 0 {
			g.Packages = append(g.Packages, repoPkgGrounding{Dir: dir})
			continue
		}
		g.Packages = append(g.Packages, repoPkgGrounding{
			Dir:        dir,
			TopSymbols: uniqueSymbolNames(syms),
		})
	}
	return g
}

func uniqueSymbolNames(syms []store.GraphSymbol) []string {
	seen := make(map[string]bool, len(syms))
	out := make([]string, 0, len(syms))
	for _, s := range syms {
		if s.Name == "" || seen[s.Name] {
			continue
		}
		seen[s.Name] = true
		out = append(out, s.Name)
	}
	return out
}

// projectImports filters and trims imps to the project's module prefix.
// External (third-party / stdlib) entries are dropped because the
// prompt uses this set to constrain claims about sibling packages —
// stdlib imports add noise without grounding any cross-package claim.
// Returns nil when modPath is empty (non-Go projects: no grounding).
func projectImports(imps []string, modPath string) []string {
	if modPath == "" {
		return nil
	}
	prefix := modPath + "/"
	var out []string
	for _, imp := range imps {
		if strings.HasPrefix(imp, prefix) {
			out = append(out, strings.TrimPrefix(imp, prefix))
		}
	}
	return out
}

// readModulePath parses go.mod at root and returns the declared module
// path, or "" if go.mod is missing or unparseable. Duplicated from
// internal/guide so the index package doesn't take an upstream
// dependency on the renderer.
func readModulePath(root string) string {
	data, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if rest, ok := strings.CutPrefix(line, "module "); ok {
			return strings.TrimSpace(rest)
		}
	}
	return ""
}
