package main

// `dex serve` — HTTP transport for the same primitives the stdio MCP
// server exposes. Boots a long-lived daemon serving a fixed list of
// project roots, optionally gated by a bearer token. The MCP server
// itself is reused (same handlers, same auto-watcher); only the
// transport changes.

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/alehatsman/dex/internal/mcp"
)

func cmdServe(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	setHelp(fs,
		"Run dex as an HTTP daemon serving one or more pre-indexed projects.",
		"dex serve [--addr :8080] --project /path/to/repo [--project /path2 ...]")

	addr := fs.String("addr", ":8080",
		"TCP listen address. Without DEX_SERVE_TOKEN, only loopback binds (127.0.0.1, ::1, localhost) are accepted.")
	var projectFlag stringSlice
	fs.Var(&projectFlag, "project",
		"Project root to serve. Repeatable. Each path must already be indexed (run `dex index <path>` first).")

	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
		return err
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("serve takes no positional args (got %v)", fs.Args())
	}
	if len(projectFlag) == 0 {
		return fmt.Errorf("at least one --project is required")
	}

	base, err := indexDir()
	if err != nil {
		return err
	}

	registry, err := mcp.BuildProjectRegistry(projectFlag)
	if err != nil {
		return fmt.Errorf("project registry: %w", err)
	}
	if warnings := mcp.PreflightProjects(registry, base); len(warnings) > 0 {
		for _, w := range warnings {
			fmt.Fprintf(os.Stderr, "warning: %s\n", w)
		}
	}

	token := strings.TrimSpace(os.Getenv("DEX_SERVE_TOKEN"))

	// Reuse the same Server wiring the stdio MCP path uses — embed
	// client, chat client, autowatch config, etc. Only the transport
	// differs.
	srv, _ := newServerFromEnv(base)

	fmt.Printf("dex serve\n")
	fmt.Printf("  addr=%s  projects=%d  auth=%v\n", *addr, len(registry), token != "")
	for id, root := range registry {
		fmt.Printf("  %s  %s\n", id[:12], root)
	}

	return srv.RunHTTP(ctx, mcp.RunHTTPOptions{
		Addr:     *addr,
		Token:    token,
		Projects: registry,
		Logger:   cliLogger(),
	})
}

// stringSlice satisfies flag.Value so --project can be repeated.
type stringSlice []string

func (s *stringSlice) String() string     { return strings.Join(*s, ",") }
func (s *stringSlice) Set(v string) error { *s = append(*s, v); return nil }
