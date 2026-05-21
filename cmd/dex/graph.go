package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/alehatsman/dex/internal/graph"
	"github.com/alehatsman/dex/internal/mcp"
	"github.com/alehatsman/dex/internal/proj"
	"github.com/alehatsman/dex/internal/store"
)

// cmdGraph dispatches `dex graph <subcommand>`. The leaf
// subcommands (neighbors / deps / callers / callees / export) mirror
// the MCP `graph_*` tools 1:1 so CLI and MCP feel like the same tool.
func cmdGraph(ctx context.Context, args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("graph needs a subcommand: neighbors | deps | callers | callees | export")
	}
	sub, rest := args[0], args[1:]
	switch sub {
	case "index":
		return fmt.Errorf("`graph index` has been folded into `index` — use `dex index --graph=only <path>` (or just `dex index <path>`, which runs both phases)")
	case "neighbors":
		return cmdGraphNeighbors(ctx, rest)
	case "deps":
		return cmdGraphDeps(ctx, rest)
	case "callers":
		return cmdGraphCallers(ctx, rest)
	case "callees":
		return cmdGraphCallees(ctx, rest)
	case "export":
		return cmdGraphExport(ctx, rest)
	case "-h", "--help", "help":
		fmt.Fprintln(os.Stderr, `usage:
  dex graph neighbors <path> <file> <line>   vector neighbours of a chunk (MCP: graph_neighbors)
  dex graph deps      <path> [flags]         imports edges (MCP: graph_deps)
                                                  --file=<rel>  --package=<full>
  dex graph callers   <path> <name>          incoming calls edges (MCP: graph_callers)
                                                  --package=<pkg>  --k=<n>
  dex graph callees   <path> <name>          outgoing calls edges (MCP: graph_callees)
                                                  --package=<pkg>  --k=<n>
  dex graph export    <path> [--output=<dir>]
                                                  dump nodes/edges as JSONL

note:
  'graph index' is gone — use 'dex index --graph=only <path>'.
  Plain 'dex index <path>' runs both chunk and graph phases.`)
		return nil
	default:
		return fmt.Errorf("unknown graph subcommand: %s (have: neighbors, deps, callers, callees, export)", sub)
	}
}

func cmdGraphNeighbors(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("graph neighbors", flag.ContinueOnError)
	setHelp(fs,
		"Find chunks semantically related to a given chunk (MCP: graph_neighbors).",
		"dex graph neighbors [flags] <path> <file> <line>")
	k := fs.Int("k", 8, "number of related chunks to return (max 30)")
	format := fs.String("format", "text", "output format: text | json")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) != 3 {
		return fmt.Errorf("graph neighbors needs <path> <file> <line>")
	}
	line, err := parsePositiveInt("line", rest[2])
	if err != nil {
		return err
	}
	base, err := indexDir()
	if err != nil {
		return err
	}
	p, err := proj.Resolve(rest[0], base)
	if err != nil {
		return err
	}
	s, _ := newServerFromEnv(base)
	out, err := s.Related(ctx, mcp.RelatedInput{
		Path:        rest[1],
		StartLine:   line,
		ProjectRoot: p.Root,
		K:           *k,
	})
	if err != nil {
		return err
	}
	if *format == "json" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}
	printSearchHitResult(out.Status, out.Hint, out.Project, out.Hits)
	return nil
}

func cmdGraphDeps(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("graph deps", flag.ContinueOnError)
	setHelp(fs,
		"Return `imports` edges for a file or package (MCP: graph_deps).",
		"dex graph deps [flags] <path>")
	file := fs.String("file", "", "relative file path inside the project (resolved to its package)")
	pkg := fs.String("package", "", "full package path (e.g. 'github.com/foo/bar/internal/baz'); takes precedence over --file")
	format := fs.String("format", "text", "output format: text | json")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) != 1 {
		return fmt.Errorf("graph deps needs <path>")
	}
	if *file == "" && *pkg == "" {
		return fmt.Errorf("graph deps needs --file=<rel> or --package=<full>")
	}
	base, err := indexDir()
	if err != nil {
		return err
	}
	p, err := proj.Resolve(rest[0], base)
	if err != nil {
		return err
	}
	s, _ := newServerFromEnv(base)
	out, err := s.GraphDeps(ctx, mcp.GraphDepsInput{
		Path:        *file,
		Package:     *pkg,
		ProjectRoot: p.Root,
	})
	if err != nil {
		return err
	}
	if *format == "json" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}
	if out.Status != "ok" {
		fmt.Fprintf(os.Stderr, "status: %s\n", out.Status)
		if out.Hint != "" {
			fmt.Fprintf(os.Stderr, "hint:   %s\n", out.Hint)
		}
		return nil
	}
	fmt.Printf("package: %s\n", out.Package)
	if len(out.Imports) == 0 {
		fmt.Println("(no import edges)")
		return nil
	}
	for _, dep := range out.Imports {
		fmt.Printf("  → %s\n", dep.ToPackage)
	}
	return nil
}

func cmdGraphCallers(ctx context.Context, args []string) error {
	return runGraphCallEdges(ctx, args, true)
}

func cmdGraphCallees(ctx context.Context, args []string) error {
	return runGraphCallEdges(ctx, args, false)
}

func runGraphCallEdges(ctx context.Context, args []string, callers bool) error {
	name := "graph callees"
	rel := "callees"
	helpOneLiner := "Outgoing `calls` edges (MCP: graph_callees). Go-only today."
	if callers {
		name = "graph callers"
		rel = "callers"
		helpOneLiner = "Incoming `calls` edges (MCP: graph_callers). Go-only today."
	}
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	setHelp(fs, helpOneLiner, "dex "+name+" [flags] <path> <name>")
	k := fs.Int("k", 12, "max hits to return (default 12, max 50)")
	pkg := fs.String("package", "", "package path filter (when the same name is defined in multiple packages)")
	format := fs.String("format", "text", "output format: text | json")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) != 2 {
		return fmt.Errorf("%s needs <path> <name>", name)
	}
	base, err := indexDir()
	if err != nil {
		return err
	}
	p, err := proj.Resolve(rest[0], base)
	if err != nil {
		return err
	}
	in := mcp.CallEdgeInput{
		Name:        rest[1],
		Package:     *pkg,
		ProjectRoot: p.Root,
		K:           *k,
	}
	s, _ := newServerFromEnv(base)
	var out mcp.CallEdgeOutput
	if callers {
		out, err = s.GraphCallers(ctx, in)
	} else {
		out, err = s.GraphCallees(ctx, in)
	}
	if err != nil {
		return err
	}
	if *format == "json" {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}
	if out.Status != "ok" {
		fmt.Fprintf(os.Stderr, "status: %s\n", out.Status)
		if out.Hint != "" {
			fmt.Fprintf(os.Stderr, "hint:   %s\n", out.Hint)
		}
		return nil
	}
	if len(out.Targets) == 0 {
		fmt.Fprintln(os.Stderr, "no targets matched")
		return nil
	}
	fmt.Printf("targets (%d):\n", len(out.Targets))
	for _, t := range out.Targets {
		fmt.Printf("  %s  (%s)  %s\n", t.QualifiedName, t.Kind, t.Package)
	}
	fmt.Println()
	if len(out.Hits) == 0 {
		fmt.Printf("no %s\n", rel)
		return nil
	}
	fmt.Printf("%s (%d):\n", rel, len(out.Hits))
	for i, h := range out.Hits {
		loc := fmt.Sprintf("%s:%d", h.Path, h.StartLine)
		header := fmt.Sprintf("─── #%d %s  (%s)", i+1, h.QualifiedName, h.Kind)
		fmt.Println(header)
		fmt.Printf("  def: %s\n", loc)
		if h.CallSitePath != "" {
			fmt.Printf("  call site: %s:%d\n", h.CallSitePath, h.CallSiteLine)
		}
		if h.Role != "" {
			fmt.Printf("  role: %s\n", h.Role)
		}
		if h.Content != "" {
			for line := range strings.SplitSeq(strings.TrimRight(h.Content, "\n"), "\n") {
				fmt.Printf("  │ %s\n", line)
			}
			if h.Truncated {
				fmt.Println("  │ … (truncated; Read the file for the rest)")
			}
		}
		fmt.Println()
	}
	return nil
}

// parsePositiveInt is a tiny CLI helper for arg-parsing positional
// integers (e.g. `<line>`). Returns an error with the flag/arg name so
// the user knows which token failed.
func parsePositiveInt(name, raw string) (int, error) {
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 {
		return 0, fmt.Errorf("%s must be a positive integer (got %q)", name, raw)
	}
	return v, nil
}

// graphIndexResult is the JSON payload emitted by `index --graph=only --format=json`.
type graphIndexResult struct {
	Project    string   `json:"project"`
	Packages   int      `json:"packages"`
	Nodes      int64    `json:"nodes"`
	Edges      int64    `json:"edges"`
	Pruned     int64    `json:"pruned_nodes"`
	PrunedEdge int64    `json:"pruned_edges"`
	Linked     int      `json:"linked_to_chunks"`
	ElapsedMS  int64    `json:"elapsed_ms"`
	Warnings   []string `json:"warnings,omitempty"`
}

// runGraphPhase extracts the Go static graph for p and upserts into st.
// Shared by `index` (Phase 2) and `index --graph=only`.
func runGraphPhase(ctx context.Context, p *proj.Project, st *store.Store, verbose bool) (*graph.Stats, error) {
	gx := graph.New(p, graph.NewStoreAdapter(st), graph.Options{
		Verbose: verbose,
		Logger:  cliLogger(),
	})
	return gx.Run(ctx)
}

// reportGraphStats prints either a text summary or a JSON blob matching
// the old `graph index --format=json` schema, so existing scripts can
// migrate to `index --graph=only --format=json` without a payload change.
func reportGraphStats(project string, stats *graph.Stats, format string) error {
	switch format {
	case "json":
		out := graphIndexResult{
			Project:    project,
			Packages:   stats.Packages,
			Nodes:      stats.NodesUpserted,
			Edges:      stats.EdgesUpserted,
			Pruned:     stats.NodesPruned,
			PrunedEdge: stats.EdgesPruned,
			Linked:     stats.LinkedToChunks,
			ElapsedMS:  stats.Elapsed.Milliseconds(),
			Warnings:   stats.Warnings,
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	default:
		fmt.Printf("  graph: %d packages  %d nodes  %d edges  %d linked  pruned %d/%d  in %s\n",
			stats.Packages, stats.NodesUpserted, stats.EdgesUpserted,
			stats.LinkedToChunks, stats.NodesPruned, stats.EdgesPruned, stats.Elapsed)
		if len(stats.Warnings) > 0 {
			fmt.Printf("  warnings: %d\n", len(stats.Warnings))
			for _, w := range stats.Warnings {
				fmt.Printf("    %s\n", w)
			}
		}
		return nil
	}
}

func cmdGraphExport(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("graph export", flag.ContinueOnError)
	setHelp(fs,
		"Dump graph_nodes/graph_edges as JSONL.",
		"dex graph export [--output=<dir>] <path>")
	output := fs.String("output", "", "output directory (default: <project>/.dex/graph)")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) != 1 {
		return fmt.Errorf("graph export needs exactly one path argument")
	}

	base, err := indexDir()
	if err != nil {
		return err
	}
	p, err := proj.Resolve(rest[0], base)
	if err != nil {
		return err
	}
	if _, err := os.Stat(p.DBPath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("no index at %s — run `dex index %s` first", p.DBPath, p.Root)
		}
		return err
	}
	st, err := openStore(ctx, p.DBPath)
	if err != nil {
		return err
	}
	defer st.Close()

	outDir := *output
	if outDir == "" {
		outDir = filepath.Join(p.Root, ".dex", "graph")
	}
	if err := graph.ExportJSONL(ctx, graph.NewStoreAdapter(st), outDir); err != nil {
		return err
	}
	fmt.Printf("✓ graph exported to %s\n", outDir)
	fmt.Printf("  nodes: %s\n", filepath.Join(outDir, "nodes.jsonl"))
	fmt.Printf("  edges: %s\n", filepath.Join(outDir, "edges.jsonl"))
	return nil
}
