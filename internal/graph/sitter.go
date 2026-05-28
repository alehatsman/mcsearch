package graph

// This file defines the multi-language structural-graph extractor
// framework. Per-language extractors (Python, TS, Rust, …) plug in by
// implementing Extractor and calling Register from an init() in their
// own file. The framework walks the project tree once per Run, honoring
// ignore rules, parses each file with the right tree-sitter grammar,
// and hands the parsed tree to the matching extractor.
//
// Provenance: all edges emitted via this framework are stamped
// `metadata.provenance = "sitter"` so consumers (MCP layer, guide
// renderer, future LSP-as-consumer upgrade) can distinguish them from
// the type-resolved Go edges emitted by ExtractGo. Nodes are not
// stamped — `kind` already disambiguates and we mirror yaml.go's choice
// of leaving the language tag to the extractor.
//
// What the framework does NOT do:
//   - Cross-file name resolution. An extractor that wants to resolve
//     `foo.bar.baz()` to a node in another file does that itself,
//     typically by collecting declarations during ProcessFile and
//     emitting deferred edges in Finalize.
//   - Provide call-graph precision. Tree-sitter is name-based; dynamic
//     dispatch, monkey-patching, and re-exports are out of scope.
//     Type-resolved precision is the LSP-as-consumer upgrade lane.

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	sitter "github.com/smacker/go-tree-sitter"

	"github.com/alehatsman/dex/internal/ignore"
)

// FileInput is the per-file argument handed to Extractor.ProcessFile.
// Tree and Root are owned by the framework — the extractor MUST NOT
// call Tree.Close. Source remains valid for the duration of the call.
type FileInput struct {
	// RelPath is the project-root-relative path with forward slashes,
	// matching the form chunk rows and other extractors use.
	RelPath string
	// AbsPath is the absolute on-disk path; useful only when an
	// extractor needs to read sibling files.
	AbsPath string
	// Source is the file's bytes.
	Source []byte
	// Tree is the parsed syntax tree. Lifetime is the ProcessFile call.
	Tree *sitter.Tree
	// Root is shorthand for Tree.RootNode().
	Root *sitter.Node
}

// Extractor is implemented by per-language modules in this package.
// Lifecycle, per Run:
//
//	Init     → ProcessFile (per matching file) → Finalize
//
// The framework calls these in order. A fresh extractor instance may
// be used per Run (Finalize returns the accumulated graph); the
// registry holds factories, not singletons, so per-Run state stays
// isolated.
type Extractor interface {
	// Name is a short identifier used in logs and warnings (e.g.
	// "python", "ts", "rust").
	Name() string
	// Language returns the tree-sitter grammar to use for files with
	// the extensions this extractor claims.
	Language() *sitter.Language
	// Extensions returns the lowercase file extensions (with leading
	// dot) this extractor handles. Returning ".py" claims every file
	// whose filepath.Ext is ".py".
	Extensions() []string

	// Init is called once before any ProcessFile. projectRoot is
	// absolute. Use this to reset accumulator state.
	Init(ctx context.Context, projectRoot string) error
	// ProcessFile is called for every matching file the walker visits.
	// Implementations should be tolerant of partial trees (tree-sitter
	// returns a tree even on parse errors); silent skip is preferable
	// to a hard error since the framework treats an error here as
	// fatal for the whole pass.
	ProcessFile(ctx context.Context, in FileInput) error
	// Finalize is called once after the last ProcessFile. The returned
	// nodes/edges are folded into the per-run ExtractResult. Warnings
	// are non-fatal and surface in res.Warnings.
	Finalize(ctx context.Context) (nodes []Node, edges []Edge, warnings []string, err error)
}

// ExtractorFactory builds a fresh extractor instance per Run. We hold
// factories instead of singletons so concurrent Runs (tests, daemon
// + CLI) don't share mutable accumulator state.
type ExtractorFactory func() Extractor

// Registry maps file extensions to extractor factories. Most callers
// use DefaultRegistry (populated via Register from per-language init);
// tests pass their own Registry into ExtractSitterWith for isolation.
type Registry struct {
	mu        sync.RWMutex
	factories []ExtractorFactory
	// byExt is a derived lookup: each extension maps to the *index* in
	// factories. Multiple extensions can share one factory.
	byExt map[string]int
}

// NewRegistry returns an empty Registry. The framework treats an empty
// registry as a no-op (ExtractSitter returns an empty ExtractResult).
func NewRegistry() *Registry {
	return &Registry{byExt: map[string]int{}}
}

// Register adds f to r. Calling Register more than once for the same
// extension wins-last; this matters only for the edge case of a
// language extension supported by two extractors (don't do this).
func (r *Registry) Register(f ExtractorFactory) {
	if f == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	probe := f()
	if probe == nil {
		return
	}
	idx := len(r.factories)
	r.factories = append(r.factories, f)
	for _, ext := range probe.Extensions() {
		r.byExt[strings.ToLower(ext)] = idx
	}
}

// Extensions returns the sorted, deduped list of extensions handled by
// this registry. Used by the walker to skip files cheaply.
func (r *Registry) Extensions() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.byExt))
	for ext := range r.byExt {
		out = append(out, ext)
	}
	return out
}

// DefaultRegistry is the package-level registry. Per-language files
// (sitter_python.go, sitter_ts.go, …) call Register from init.
var DefaultRegistry = NewRegistry()

// Register installs f in DefaultRegistry. Per-language extractors call
// this from their own init() — see the database/sql driver pattern.
func Register(f ExtractorFactory) { DefaultRegistry.Register(f) }

// ExtractSitter is the main entry point. Walks projectRoot with the
// shared ignore matcher, parses each file matching a registered
// extension with the right tree-sitter grammar, and dispatches to the
// extractor. Returns an empty ExtractResult when no extractors are
// registered (the default state until per-language files land).
//
// Per-file parse failures and per-extractor warnings surface in
// res.Warnings; they do not abort the pass. Only context cancellation
// and a top-level walker error are returned as `err`.
func ExtractSitter(ctx context.Context, projectRoot string) (*ExtractResult, error) {
	return ExtractSitterWith(ctx, projectRoot, DefaultRegistry)
}

// ExtractSitterWith is ExtractSitter with an explicit registry. Used
// by tests that want hermetic, package-state-free dispatch.
func ExtractSitterWith(ctx context.Context, projectRoot string, reg *Registry) (*ExtractResult, error) {
	res := &ExtractResult{}
	if reg == nil {
		return res, nil
	}

	// Snapshot the registry's factories under the lock, then release.
	// Holding the lock across file I/O would serialize concurrent Runs
	// for no reason — once we have the snapshot the per-Run state is
	// entirely local.
	reg.mu.RLock()
	if len(reg.factories) == 0 {
		reg.mu.RUnlock()
		return res, nil
	}
	factories := make([]ExtractorFactory, len(reg.factories))
	copy(factories, reg.factories)
	extByName := make(map[string]int, len(reg.byExt))
	for ext, idx := range reg.byExt {
		extByName[ext] = idx
	}
	reg.mu.RUnlock()

	matcher, err := ignore.New(projectRoot)
	if err != nil {
		return nil, fmt.Errorf("ignore.New: %w", err)
	}

	// One live extractor instance per factory, lazily built when the
	// first file for its extension shows up. Skipping Init for unused
	// languages keeps walks of single-language projects cheap.
	extractors := make([]Extractor, len(factories))
	parsers := make([]*sitter.Parser, len(factories))
	defer func() {
		for _, p := range parsers {
			if p != nil {
				p.Close()
			}
		}
	}()

	walkErr := filepath.WalkDir(projectRoot, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		rel, relErr := filepath.Rel(projectRoot, path)
		if relErr != nil {
			return relErr
		}
		if rel == "." {
			return nil
		}
		if matcher.Match(rel, d.IsDir()) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(rel))
		idx, ok := extByName[ext]
		if !ok {
			return nil
		}

		if extractors[idx] == nil {
			ex := factories[idx]()
			if ex == nil {
				return nil
			}
			if initErr := ex.Init(ctx, projectRoot); initErr != nil {
				res.Warnings = append(res.Warnings,
					fmt.Sprintf("%s: init: %v", ex.Name(), initErr))
				// Mark as failed by leaving extractors[idx] nil and
				// removing the extension from the lookup so we don't
				// re-Init on every file.
				for e, i := range extByName {
					if i == idx {
						delete(extByName, e)
					}
				}
				return nil
			}
			p := sitter.NewParser()
			p.SetLanguage(ex.Language())
			extractors[idx] = ex
			parsers[idx] = p
		}

		src, readErr := os.ReadFile(path)
		if readErr != nil {
			res.Warnings = append(res.Warnings,
				fmt.Sprintf("%s: read %s: %v", extractors[idx].Name(), rel, readErr))
			return nil
		}
		tree, parseErr := parsers[idx].ParseCtx(ctx, nil, src)
		if parseErr != nil {
			res.Warnings = append(res.Warnings,
				fmt.Sprintf("%s: parse %s: %v", extractors[idx].Name(), rel, parseErr))
			return nil
		}
		// Tree must be released before the next ParseCtx on the same
		// parser; close at the end of this file's handling.
		func() {
			defer tree.Close()
			in := FileInput{
				RelPath: filepath.ToSlash(rel),
				AbsPath: path,
				Source:  src,
				Tree:    tree,
				Root:    tree.RootNode(),
			}
			if procErr := extractors[idx].ProcessFile(ctx, in); procErr != nil {
				res.Warnings = append(res.Warnings,
					fmt.Sprintf("%s: %s: %v", extractors[idx].Name(), in.RelPath, procErr))
			}
		}()
		return nil
	})
	if walkErr != nil {
		return nil, walkErr
	}

	// Drain each extractor that actually saw a file (the ones that did
	// not stay nil).
	for _, ex := range extractors {
		if ex == nil {
			continue
		}
		nodes, edges, warnings, finErr := ex.Finalize(ctx)
		if finErr != nil {
			res.Warnings = append(res.Warnings,
				fmt.Sprintf("%s: finalize: %v", ex.Name(), finErr))
			continue
		}
		res.Nodes = append(res.Nodes, nodes...)
		res.Edges = append(res.Edges, stampProvenance(edges, ex.Name())...)
		res.Warnings = append(res.Warnings, warnings...)
	}
	return res, nil
}

// stampProvenance sets edge.Metadata["provenance"] = "sitter" and
// edge.Metadata["sitter_lang"] = name on every edge whose metadata
// doesn't already pin a provenance. Mutates in place and returns the
// same slice for chained appends. The double tag lets the consumer
// distinguish "any tree-sitter edge" from "specifically the Python
// extractor's edge" without parsing edge IDs.
func stampProvenance(edges []Edge, name string) []Edge {
	for i := range edges {
		if edges[i].Metadata == nil {
			edges[i].Metadata = map[string]any{}
		}
		if _, ok := edges[i].Metadata["provenance"]; !ok {
			edges[i].Metadata["provenance"] = "sitter"
		}
		if _, ok := edges[i].Metadata["sitter_lang"]; !ok {
			edges[i].Metadata["sitter_lang"] = name
		}
	}
	return edges
}
