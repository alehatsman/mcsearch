package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/alehatsman/mcsearch/internal/graph"
	"github.com/alehatsman/mcsearch/internal/proj"
)

// cmdGraph dispatches `mcsearch graph <subcommand>`. Sub-subcommands
// keep the top-level `mcsearch` switch flat — adding more leaves
// (query, callers, callees, ...) in follow-up PRs lands here, not in
// the root switch.
func cmdGraph(ctx context.Context, args []string) error {
	if len(args) < 1 {
		return fmt.Errorf("graph needs a subcommand: index|export")
	}
	sub, rest := args[0], args[1:]
	switch sub {
	case "index":
		return cmdGraphIndex(ctx, rest)
	case "export":
		return cmdGraphExport(ctx, rest)
	case "-h", "--help", "help":
		fmt.Fprintln(os.Stderr, `usage:
  mcsearch graph index  <path>        build or refresh the Go static graph
  mcsearch graph export <path>        dump nodes/edges as JSONL

flags:
  mcsearch graph index <path> [-v] [--format=json]
  mcsearch graph export <path> [--output=<dir>] [--format=jsonl]`)
		return nil
	default:
		return fmt.Errorf("unknown graph subcommand: %s (have: index, export)", sub)
	}
}

// graphIndexResult is the JSON payload emitted by `graph index --format=json`.
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

func cmdGraphIndex(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("graph index", flag.ContinueOnError)
	verbose := fs.Bool("v", false, "verbose")
	force := fs.Bool("force", false, "bypass protected-path and git-tree guards")
	format := fs.String("format", "text", "output format: text | json")
	if err := fs.Parse(args); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) != 1 {
		return fmt.Errorf("graph index needs exactly one path argument")
	}

	base, err := indexDir()
	if err != nil {
		return err
	}
	p, err := proj.Resolve(rest[0], base)
	if err != nil {
		return err
	}
	if err := proj.CheckIndexable(p, *force); err != nil {
		return err
	}
	if err := p.EnsureCacheDir(); err != nil {
		return err
	}
	st, err := openStore(ctx, p.DBPath)
	if err != nil {
		return err
	}
	defer st.Close()

	gx := graph.New(p, graph.NewStoreAdapter(st), graph.Options{
		Verbose: *verbose,
		Logger:  cliLogger(),
	})
	stats, err := gx.Run(ctx)
	if err != nil {
		return err
	}
	if err := st.SetProjectRoot(ctx, p.Root); err != nil {
		return err
	}

	switch *format {
	case "json":
		out := graphIndexResult{
			Project:    p.Root,
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
		fmt.Printf("✓ graph indexed %s\n", p.Root)
		fmt.Printf("  packages: %d  nodes: %d  edges: %d  linked: %d  pruned: %d/%d  elapsed: %s\n",
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
	output := fs.String("output", "", "output directory (default: <project>/.mcsearch/graph)")
	format := fs.String("format", "jsonl", "output format: jsonl (dot lands in a follow-up PR)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) != 1 {
		return fmt.Errorf("graph export needs exactly one path argument")
	}
	if *format != "jsonl" {
		return fmt.Errorf("unsupported --format=%s (this PR ships jsonl only)", *format)
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
			return fmt.Errorf("no index at %s — run `mcsearch graph index %s` first", p.DBPath, p.Root)
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
		outDir = filepath.Join(p.Root, ".mcsearch", "graph")
	}
	if err := graph.ExportJSONL(ctx, graph.NewStoreAdapter(st), outDir); err != nil {
		return err
	}
	fmt.Printf("✓ graph exported to %s\n", outDir)
	fmt.Printf("  nodes: %s\n", filepath.Join(outDir, "nodes.jsonl"))
	fmt.Printf("  edges: %s\n", filepath.Join(outDir, "edges.jsonl"))
	return nil
}
