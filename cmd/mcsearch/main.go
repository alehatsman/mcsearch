// mcsearch — local semantic-search helper for Claude Code.
//
// Subcommands:
//   index <path>            Build or refresh the per-project index.
//   query <path> <query...> Search an existing index from the terminal.
//   status [<path>]         Show endpoint health and indexed projects.
//   nuke <path>             Delete the on-disk index for a project.
//   watch <path>            Keep the index fresh as files change.
//   clone <src> <dst>       Seed dst's index from src's (worktrees).
//   mcp                     Run as an MCP server over stdio.
//   version                 Print the build version.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/alehatsman/mcsearch/internal/embed"
	"github.com/alehatsman/mcsearch/internal/ignore"
	"github.com/alehatsman/mcsearch/internal/index"
	"github.com/alehatsman/mcsearch/internal/mcp"
	"github.com/alehatsman/mcsearch/internal/proj"
	"github.com/alehatsman/mcsearch/internal/store"
	"github.com/alehatsman/mcsearch/internal/watch"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	cmd, args := os.Args[1], os.Args[2:]

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	var err error
	switch cmd {
	case "index":
		err = cmdIndex(ctx, args)
	case "query":
		err = cmdQuery(ctx, args)
	case "status":
		err = cmdStatus(ctx, args)
	case "nuke":
		err = cmdNuke(ctx, args)
	case "mcp":
		err = cmdMCP(ctx, args)
	case "watch":
		err = cmdWatch(ctx, args)
	case "clone":
		err = cmdClone(ctx, args)
	case "version", "-V", "--version":
		fmt.Println(mcp.Version)
		return
	case "-h", "--help", "help":
		usage()
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", cmd)
		usage()
		os.Exit(2)
	}
	if err != nil {
		// `-h` returns flag.ErrHelp via flag.ContinueOnError. The FlagSet
		// already printed its usage block; suppress the redundant
		// "flag: help requested" line and exit cleanly.
		if errors.Is(err, flag.ErrHelp) {
			return
		}
		// SIGINT/SIGTERM cancel ctx; report a friendlier exit (130 is
		// the conventional shell code for SIGINT).
		if errors.Is(err, context.Canceled) {
			fmt.Fprintln(os.Stderr, "interrupted")
			os.Exit(130)
		}
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `usage:
  mcsearch index <path>             build or refresh the index for a project
  mcsearch query <path> <query>     return top-k chunks for a query
  mcsearch status [<path>]          show endpoint health and project stats
  mcsearch nuke   <path>            delete the on-disk index for a project
  mcsearch mcp                      run as an MCP server over stdio
  mcsearch watch  <path>            keep the index fresh as files change
  mcsearch clone  <src> <dst>       seed dst's index from src's (e.g. for a
                                    new git worktree); follow with
                                    `+"`mcsearch index <dst>`"+` to reconcile
                                    any chunks that differ between the two.

env:
  MCSEARCH_EMBED_URL      default http://127.0.0.1:8082
  MCSEARCH_EMBED_MODEL    default Qwen/Qwen3-Embedding-4B
  MCSEARCH_EMBED_BATCH    default 32
  MCSEARCH_EMBED_TIMEOUT  default 60s (Go duration)
  MCSEARCH_INDEX_DIR      default ~/.cache/mcsearch`)
}

// ─── env helpers ──────────────────────────────────────────────────────────

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func indexDir() (string, error) {
	if v := os.Getenv("MCSEARCH_INDEX_DIR"); v != "" {
		return v, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".cache", "mcsearch"), nil
}

func newEmbedClient() *embed.Client {
	url := envOr("MCSEARCH_EMBED_URL", "http://127.0.0.1:8082")
	model := envOr("MCSEARCH_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
	batch, _ := strconv.Atoi(envOr("MCSEARCH_EMBED_BATCH", "32"))
	timeout, err := time.ParseDuration(envOr("MCSEARCH_EMBED_TIMEOUT", "60s"))
	if err != nil {
		timeout = 60 * time.Second
	}
	return embed.New(url, model, batch, timeout)
}

// ─── index ─────────────────────────────────────────────────────────────────

func cmdIndex(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("index", flag.ContinueOnError)
	verbose := fs.Bool("v", false, "verbose")
	if err := fs.Parse(args); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) != 1 {
		return fmt.Errorf("index needs exactly one path argument")
	}
	base, err := indexDir()
	if err != nil {
		return err
	}
	p, err := proj.Resolve(rest[0], base)
	if err != nil {
		return err
	}
	if err := p.EnsureCacheDir(); err != nil {
		return err
	}
	st, err := store.Open(ctx, p.DBPath)
	if err != nil {
		return err
	}
	defer st.Close()
	ig, err := ignore.New(p.Root)
	if err != nil {
		return err
	}
	ix := index.New(p, st, newEmbedClient(), ig, index.Options{Verbose: *verbose})
	if err := ix.Run(ctx); err != nil {
		return err
	}
	stats, err := st.Stats(ctx)
	if err != nil {
		return err
	}
	fmt.Printf("✓ indexed %s\n", p.Root)
	fmt.Printf("  chunks: %d  files: %d  dim: %d\n", stats.Chunks, stats.Files, stats.Dim)
	return nil
}

// ─── query ─────────────────────────────────────────────────────────────────

func cmdQuery(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("query", flag.ContinueOnError)
	k := fs.Int("k", 8, "number of results to return")
	if err := fs.Parse(args); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) < 2 {
		return fmt.Errorf("query needs <path> <query...>")
	}
	path := rest[0]
	q := strings.Join(rest[1:], " ")
	if strings.TrimSpace(q) == "" {
		return fmt.Errorf("query is empty — pass a natural-language description or code fragment")
	}

	base, err := indexDir()
	if err != nil {
		return err
	}
	p, err := proj.Resolve(path, base)
	if err != nil {
		return err
	}
	if _, err := os.Stat(p.DBPath); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("no index for %s — run `mcsearch index %s` first", p.Root, p.Root)
		}
		return err
	}
	st, err := store.Open(ctx, p.DBPath)
	if err != nil {
		return err
	}
	defer st.Close()
	em := newEmbedClient()
	vecs, err := em.Embed(ctx, []string{q})
	if err != nil {
		return err
	}
	hits, err := st.Search(ctx, vecs[0], *k)
	if err != nil {
		return err
	}
	if len(hits) == 0 {
		fmt.Fprintln(os.Stderr, "no results")
		return nil
	}
	for i, h := range hits {
		fmt.Printf("─── #%d %s:%d-%d  (%s)  score=%.4f\n", i+1, h.Path, h.StartLine, h.EndLine, h.Kind, h.Score)
		fmt.Println(truncate(h.Content, 1500))
		fmt.Println()
	}
	return nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "\n…(truncated)"
}

// ─── status ────────────────────────────────────────────────────────────────

func cmdStatus(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("status", flag.ContinueOnError)
	if err := fs.Parse(args); err != nil {
		return err
	}
	rest := fs.Args()
	em := newEmbedClient()
	checkCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	if err := em.Health(checkCtx); err != nil {
		fmt.Printf("embed endpoint: %s   UNREACHABLE (%v)\n", em.BaseURL, err)
	} else {
		fmt.Printf("embed endpoint: %s   ok\n", em.BaseURL)
	}
	fmt.Printf("model: %s\n", em.Model)
	fmt.Printf("mcsearch version: %s\n", mcp.Version)

	base, err := indexDir()
	if err != nil {
		return err
	}
	fmt.Printf("index dir: %s\n", base)
	if len(rest) == 1 {
		// Per-project status
		p, err := proj.Resolve(rest[0], base)
		if err != nil {
			return err
		}
		if _, err := os.Stat(p.DBPath); err != nil {
			if os.IsNotExist(err) {
				fmt.Printf("project: %s\n  no index — run `mcsearch index %s`\n", p.Root, p.Root)
				return nil
			}
			return err
		}
		st, err := store.Open(ctx, p.DBPath)
		if err != nil {
			return err
		}
		defer st.Close()
		stats, err := st.Stats(ctx)
		if err != nil {
			return err
		}
		fmt.Printf("project: %s\n", p.Root)
		fmt.Printf("  chunks: %d  files: %d  dim: %d  last_indexed: %s\n",
			stats.Chunks, stats.Files, stats.Dim, formatTime(stats.LastIndex))
		return nil
	}
	// All-project summary by scanning index dir
	entries, err := os.ReadDir(base)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Println("no projects indexed yet")
			return nil
		}
		return err
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dbPath := filepath.Join(base, e.Name(), "index.db")
		if _, err := os.Stat(dbPath); err != nil {
			continue
		}
		short := e.Name()
		if len(short) > 12 {
			short = short[:12]
		}
		st, err := store.Open(ctx, dbPath)
		if err != nil {
			fmt.Printf("  %s  CORRUPT (%v)\n", short, err)
			continue
		}
		stats, _ := st.Stats(ctx)
		st.Close()
		fmt.Printf("  %s  chunks=%d files=%d dim=%d  last=%s\n",
			short, stats.Chunks, stats.Files, stats.Dim, formatTime(stats.LastIndex))
	}
	return nil
}

func formatTime(t time.Time) string {
	if t.IsZero() {
		return "never"
	}
	return t.Format(time.RFC3339)
}

// ─── nuke ──────────────────────────────────────────────────────────────────

func cmdNuke(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("nuke", flag.ContinueOnError)
	if err := fs.Parse(args); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) != 1 {
		return fmt.Errorf("nuke needs exactly one path argument")
	}
	base, err := indexDir()
	if err != nil {
		return err
	}
	p, err := proj.Resolve(rest[0], base)
	if err != nil {
		return err
	}
	if _, err := os.Stat(p.CacheDir); err != nil {
		if os.IsNotExist(err) {
			fmt.Printf("nothing to remove: no index for %s\n", p.Root)
			return nil
		}
		return err
	}
	if err := os.RemoveAll(p.CacheDir); err != nil {
		return err
	}
	fmt.Printf("✓ removed index for %s\n", p.Root)
	return nil
}

// ─── watch ─────────────────────────────────────────────────────────────────

func cmdWatch(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("watch", flag.ContinueOnError)
	verbose := fs.Bool("v", false, "verbose")
	debounce := fs.Duration("debounce", 500*time.Millisecond, "quiet window before re-indexing")
	if err := fs.Parse(args); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) != 1 {
		return fmt.Errorf("watch needs exactly one path argument")
	}
	base, err := indexDir()
	if err != nil {
		return err
	}
	p, err := proj.Resolve(rest[0], base)
	if err != nil {
		return err
	}
	if err := p.EnsureCacheDir(); err != nil {
		return err
	}
	st, err := store.Open(ctx, p.DBPath)
	if err != nil {
		return err
	}
	defer st.Close()
	ig, err := ignore.New(p.Root)
	if err != nil {
		return err
	}
	ix := index.New(p, st, newEmbedClient(), ig, index.Options{Verbose: *verbose})
	w := watch.New(ix, ig, p.Root, watch.Options{Debounce: *debounce, Verbose: *verbose})
	fmt.Fprintf(os.Stderr, "mcsearch watching %s (debounce=%s)\n", p.Root, *debounce)
	return w.Run(ctx)
}

// ─── clone ─────────────────────────────────────────────────────────────────

// cmdClone seeds dst's per-project cache from src's. Useful when the same
// repository is checked out in multiple locations (e.g. git worktrees,
// branch-per-folder workflows). Chunks are keyed by (relative path,
// content sha1), so the copied index is correct for any file that exists
// at the same path with the same content in dst; differing files get
// reconciled on the next `mcsearch index <dst>` (incremental — only
// changed chunks are re-embedded).
func cmdClone(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("clone", flag.ContinueOnError)
	force := fs.Bool("force", false, "overwrite dst's index if it already exists")
	if err := fs.Parse(args); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) != 2 {
		return fmt.Errorf("clone needs <src-path> <dst-path>")
	}
	base, err := indexDir()
	if err != nil {
		return err
	}
	src, err := proj.Resolve(rest[0], base)
	if err != nil {
		return fmt.Errorf("resolve src: %w", err)
	}
	dst, err := proj.Resolve(rest[1], base)
	if err != nil {
		return fmt.Errorf("resolve dst: %w", err)
	}
	if src.ID == dst.ID {
		return fmt.Errorf("src and dst resolve to the same project root (%s); nothing to clone", src.Root)
	}
	if _, err := os.Stat(src.DBPath); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("src has no index at %s — run `mcsearch index %s` first", src.DBPath, src.Root)
		}
		return err
	}
	if _, err := os.Stat(dst.DBPath); err == nil {
		if !*force {
			return fmt.Errorf("dst already has an index at %s — pass --force to overwrite or `mcsearch nuke %s` first", dst.DBPath, dst.Root)
		}
		if err := os.RemoveAll(dst.CacheDir); err != nil {
			return fmt.Errorf("remove existing dst cache: %w", err)
		}
	}
	if err := dst.EnsureCacheDir(); err != nil {
		return err
	}
	// Copy index.db. SQLite WAL files are not copied — they're either
	// already checkpointed (idle index) or will be rebuilt on next open.
	if err := copyFile(src.DBPath, dst.DBPath); err != nil {
		return fmt.Errorf("copy index: %w", err)
	}
	fmt.Printf("✓ cloned %s → %s\n", src.Root, dst.Root)
	fmt.Printf("  next: `mcsearch index %s` will reconcile any files that differ between the two trees (incremental — only changed chunks are re-embedded).\n", dst.Root)
	return nil
}

func copyFile(srcPath, dstPath string) error {
	in, err := os.Open(srcPath)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dstPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

// ─── mcp ───────────────────────────────────────────────────────────────────

func cmdMCP(ctx context.Context, args []string) error {
	base, err := indexDir()
	if err != nil {
		return err
	}
	srv := &mcp.Server{
		EmbedClient: newEmbedClient(),
		IndexDir:    base,
	}
	return srv.RunStdio(ctx)
}

