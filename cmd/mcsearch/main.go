// mcsearch — local semantic-search helper for Claude Code.
//
// Subcommands:
//
//	index <path>              Build or refresh the per-project index (chunk + graph).
//	query <path> <query...>   Search an existing index from the terminal.
//	context <path> <q...>     One-shot context router (semantic + symbol + graph).
//	generate <path> <prompt>  Generate code grounded in the project's index.
//	status [<path>]           Show endpoint health and indexed projects.
//	env                       Print effective env-var configuration with sources.
//	compact <path>            Concatenate indexable files for LLM prompts (alias: bundle).
//	nuke <path>               Delete the on-disk index for a project.
//	reindex <path>|--all      Drop and re-embed (one project, or every known project).
//	watch <path>              Keep the index fresh as files change.
//	clone <src> <dst>         Seed dst's index from src's (worktrees).
//	graph export <path>       Dump nodes/edges as JSONL.
//	mcp                       Run as an MCP server over stdio.
//	version                   Print the build version.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/alehatsman/mcsearch/internal/chat"
	"github.com/alehatsman/mcsearch/internal/embed"
	"github.com/alehatsman/mcsearch/internal/graph"
	"github.com/alehatsman/mcsearch/internal/ignore"
	"github.com/alehatsman/mcsearch/internal/index"
	"github.com/alehatsman/mcsearch/internal/mcp"
	"github.com/alehatsman/mcsearch/internal/proj"
	"github.com/alehatsman/mcsearch/internal/rerank"
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
	case "context":
		err = cmdContext(ctx, args)
	case "generate":
		err = cmdGenerate(ctx, args)
	case "status":
		err = cmdStatus(ctx, args)
	case "env":
		err = cmdEnv(ctx, args)
	case "compact", "bundle":
		err = cmdCompact(ctx, args)
	case "nuke":
		err = cmdNuke(ctx, args)
	case "reindex":
		err = cmdReindex(ctx, args)
	case "mcp":
		err = cmdMCP(ctx, args)
	case "watch":
		err = cmdWatch(ctx, args)
	case "clone":
		err = cmdClone(ctx, args)
	case "graph":
		err = cmdGraph(ctx, args)
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
                                    (runs chunk+embed AND Go static graph phases;
                                    use --graph=off to skip graph, --graph=only
                                    to refresh just the graph layer)
  mcsearch query <path> <query>     return top-k chunks for a query
  mcsearch context <path> <question>  one-shot router: picks intent, fuses
                                    semantic + symbol (+ graph when it lands)
                                    and returns suggested_reads + a prose
                                    next_action. Use this BEFORE grep loops.
                                    Flags: --intent, --k, --format=text|json
  mcsearch generate <path> <prompt> generate code grounded in the project's
                                    index (RAG: top-k chunks → chat endpoint)
  mcsearch status [<path>]          show endpoint health and project stats
  mcsearch env                      print effective env-var config with sources
                                    (--all to include tuning knobs, --doc for
                                    descriptions, --format=text|json)
  mcsearch compact <path>           concatenate all indexable files under <path>
                                    to stdout with `+"`===== <relpath> =====`"+`
                                    headers — for pasting into LLM prompts.
                                    Honors .gitignore/.mcsearch-ignore and
                                    skips binaries + secret-shaped files.
                                    Flags: --out FILE, --max-bytes N, --strip
  mcsearch nuke   <path>            delete the on-disk index for a project
  mcsearch reindex <path>           drop and re-embed a project from scratch
  mcsearch reindex --all --yes      drop and re-embed every known project
                                    (skips indexes from before this feature;
                                    those need one fresh `+"`mcsearch index <path>`"+`)
  mcsearch mcp                      run as an MCP server over stdio
  mcsearch watch  <path>            keep the index fresh as files change
  mcsearch clone  <src> <dst>       seed dst's index from src's (e.g. for a
                                    new git worktree); follow with
                                    `+"`mcsearch index <dst>`"+` to reconcile
                                    any chunks that differ between the two.
  mcsearch graph export <path>      dump graph_nodes/graph_edges as JSONL
                                    (--output=<dir> defaults to <path>/.mcsearch/graph)
                                    (the graph itself is built by `+"`mcsearch index`"+`)

env:
  Run `+"`mcsearch env`"+` for the effective configuration. The 5 vars that
  matter for 80% of setups: MCSEARCH_EMBED_URL, MCSEARCH_EMBED_MODEL,
  MCSEARCH_INDEX_DIR, MCSEARCH_CHAT_URL, MCSEARCH_CHAT_MODEL.
  Tuning knobs (timeouts, batch sizes, optional rerank/compress/draft
  endpoints) — see docs/tuning.md or run `+"`mcsearch env --all --doc`"+`.

flags:
  mcsearch query --rerank=off <path> "..."     skip rerank for this call
  mcsearch query --format=json <path> "..."    emit hits as JSON
  mcsearch query --explain <path> "..."        show per-chunk score breakdown and stage timings`)
}

// validIntent reports whether s is one of the strategies the context
// router accepts. Empty string means "auto" and is allowed.
func validIntent(s string) bool {
	switch s {
	case "", "auto", "behavior_search", "symbol_lookup", "callers", "callees",
		"architecture", "package_topology", "editing_context":
		return true
	}
	return false
}

// boolFlag duck-types the stdlib's unexported flag.boolFlag interface so
// reorderFlags can tell standalone boolean flags (`-v`) from flags that
// consume a value as the next token (`--rerank off`).
type boolFlag interface {
	flag.Value
	IsBoolFlag() bool
}

// reorderFlags moves every flag-shaped token to the front of args so
// flag.Parse sees them even when the user typed them after positional
// args. Without this, Go's flag package silently stops parsing at the
// first non-flag arg and quietly drops every flag that follows — a
// real footgun for invocations like `mcsearch query <path> "q" --k=3`.
//
// Uses the FlagSet to detect which flags consume a separate-token value
// (so `--rerank off` is treated as one flag/value pair, not flag plus
// stray positional). `--` ends flag scanning, matching stdlib behavior.
func reorderFlags(fs *flag.FlagSet, args []string) []string {
	var flags, positional []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--" {
			positional = append(positional, args[i:]...)
			break
		}
		if !strings.HasPrefix(a, "-") || a == "-" {
			positional = append(positional, a)
			continue
		}
		flags = append(flags, a)
		if strings.Contains(a, "=") {
			continue
		}
		name := strings.TrimLeft(a, "-")
		f := fs.Lookup(name)
		if f == nil {
			// Unknown flag — let fs.Parse raise the error.
			continue
		}
		if bf, ok := f.Value.(boolFlag); ok && bf.IsBoolFlag() {
			continue
		}
		if i+1 < len(args) {
			flags = append(flags, args[i+1])
			i++
		}
	}
	return append(flags, positional...)
}

// setHelp wires `<cmd> -h` to print a one-line summary, a usage pattern
// showing positional args, and the auto-generated flag defaults. The
// default flag.FlagSet usage prints only flags and omits everything
// the user actually needs to invoke the command.
func setHelp(fs *flag.FlagSet, oneLiner, usagePattern string) {
	fs.Usage = func() {
		out := fs.Output()
		fmt.Fprintln(out, oneLiner)
		fmt.Fprintln(out)
		fmt.Fprintln(out, "usage:")
		fmt.Fprintln(out, "  "+usagePattern)
		hasFlags := false
		fs.VisitAll(func(*flag.Flag) { hasFlags = true })
		if hasFlags {
			fmt.Fprintln(out)
			fmt.Fprintln(out, "flags:")
			fs.PrintDefaults()
		}
	}
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

// storeOpts reads runtime tweaks from the environment so every code
// path that opens a Store sees the same configuration.
func storeOpts() store.Options {
	opts := store.Options{
		DisableBM25:    os.Getenv("MCSEARCH_DISABLE_BM25") == "1",
		RerankPool:     rerankPool(),
		MaxHitsPerFile: maxHitsPerFile(),
	}
	// Assign through a typed-nil check: a (*rerank.Client)(nil) stored
	// in the Reranker interface field would still compare != nil, and
	// store.Search would dispatch into a nil receiver.
	if rc := newRerankClient(); rc != nil {
		opts.Reranker = rc
	}
	return opts
}

func parseDuration(envVar, raw string, def time.Duration) time.Duration {
	d, err := time.ParseDuration(raw)
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: %s=%q is not a Go duration; using %s\n", envVar, raw, def)
		return def
	}
	return d
}

// maxHitsPerFile reads MCSEARCH_MAX_HITS_PER_FILE from the environment.
// Zero means no per-file cap (default). Positive values enforce result
// diversity — useful when a single heavily-matched file would otherwise
// dominate the top-k results.
func maxHitsPerFile() int {
	raw := os.Getenv("MCSEARCH_MAX_HITS_PER_FILE")
	if raw == "" {
		return 0
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		fmt.Fprintf(os.Stderr, "warning: MCSEARCH_MAX_HITS_PER_FILE=%q is not a non-negative integer; ignoring\n", raw)
		return 0
	}
	return n
}

func openStore(ctx context.Context, dbPath string) (*store.Store, error) {
	return store.OpenWith(ctx, dbPath, storeOpts())
}

// cliLogger returns a stderr text logger. Used for the CLI commands
// (index/watch) so verbose output goes to stderr without polluting
// stdout (which the MCP server uses for JSON-RPC).
func cliLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
}

func newEmbedClient() *embed.Client {
	url := envOr("MCSEARCH_EMBED_URL", "http://127.0.0.1:8082")
	model := envOr("MCSEARCH_EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
	rawBatch := envOr("MCSEARCH_EMBED_BATCH", "32")
	batch, err := strconv.Atoi(rawBatch)
	if err != nil || batch <= 0 {
		fmt.Fprintf(os.Stderr, "warning: MCSEARCH_EMBED_BATCH=%q is not a positive integer; using 32\n", rawBatch)
		batch = 32
	}
	conc := envInt("MCSEARCH_EMBED_CONCURRENCY", 4)
	timeout := parseDuration("MCSEARCH_EMBED_TIMEOUT", envOr("MCSEARCH_EMBED_TIMEOUT", "60s"), 60*time.Second)
	return embed.NewWithConcurrency(url, model, batch, conc, timeout)
}

func newChatClient() *chat.Client {
	url := envOr("MCSEARCH_CHAT_URL", "http://127.0.0.1:8081")
	model := envOr("MCSEARCH_CHAT_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
	timeout := parseDuration("MCSEARCH_CHAT_TIMEOUT", envOr("MCSEARCH_CHAT_TIMEOUT", "120s"), 120*time.Second)
	return chat.New(url, model, timeout)
}

// newRerankClient returns a rerank.HealthChecker (either the Cohere-compatible
// *rerank.Client or the decoder-style *rerank.ChatReranker), or nil when
// reranking is disabled. Rerank is OFF by default — empty MCSEARCH_RERANK_URL
// or MCSEARCH_DISABLE_RERANK=1 yields nil; store.Search treats nil as
// "skip the stage".
//
// MCSEARCH_RERANK_STYLE selects the backend:
//
//	"cohere" (default) — Cohere-compatible /rerank endpoint (TEI, Infinity,
//	                     vLLM with a cross-encoder model like bge-reranker-v2-m3)
//	"chat"             — chat-completions + logprobs (vLLM serving a decoder-style
//	                     reranker like Qwen3-Reranker-4B)
func newRerankClient() rerank.HealthChecker {
	url := os.Getenv("MCSEARCH_RERANK_URL")
	if url == "" {
		return nil
	}
	if os.Getenv("MCSEARCH_DISABLE_RERANK") == "1" {
		return nil
	}
	model := envOr("MCSEARCH_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
	timeout := parseDuration("MCSEARCH_RERANK_TIMEOUT", envOr("MCSEARCH_RERANK_TIMEOUT", "5s"), 5*time.Second)
	if os.Getenv("MCSEARCH_RERANK_STYLE") == "chat" {
		rawConc := envOr("MCSEARCH_RERANK_CONCURRENCY", "4")
		concurrency, cerr := strconv.Atoi(rawConc)
		if cerr != nil || concurrency <= 0 {
			fmt.Fprintf(os.Stderr, "warning: MCSEARCH_RERANK_CONCURRENCY=%q is not a positive integer; using 4\n", rawConc)
			concurrency = 4
		}
		return rerank.NewChat(url, model, concurrency, timeout)
	}
	return rerank.New(url, model, timeout)
}

// chatClientFromEnv creates a *chat.Client from environment variables.
// Returns nil if urlEnv is unset (opt-in feature). modelFallback is used
// when modelEnv is also unset.
func chatClientFromEnv(urlEnv, modelEnv, timeoutEnv, modelFallback string, defTimeout time.Duration) *chat.Client {
	url := os.Getenv(urlEnv)
	if url == "" {
		return nil
	}
	model := envOr(modelEnv, modelFallback)
	timeout := parseDuration(timeoutEnv, envOr(timeoutEnv, defTimeout.String()), defTimeout)
	return chat.New(url, model, timeout)
}

func newCompressClient() *chat.Client {
	return chatClientFromEnv("MCSEARCH_COMPRESS_URL", "MCSEARCH_COMPRESS_MODEL", "MCSEARCH_COMPRESS_TIMEOUT",
		envOr("MCSEARCH_CHAT_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"), 30*time.Second)
}

func newDraftClient() *chat.Client {
	return chatClientFromEnv("MCSEARCH_DRAFT_URL", "MCSEARCH_DRAFT_MODEL", "MCSEARCH_DRAFT_TIMEOUT",
		envOr("MCSEARCH_CHAT_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"), 120*time.Second)
}

// newSummaryClient builds the chat client used for index-time
// summaries (file / chunk / package / repo). Per-chunk and per-file
// summaries are short (≤ 400 tokens) and dominated by call count, so
// users typically point this at a smaller, faster model than the main
// chat leg used by generate / ask_codebase. Falls back to the main
// chat client when MCSEARCH_SUMMARY_URL is unset.
func newSummaryClient() *chat.Client {
	if os.Getenv("MCSEARCH_SUMMARY_URL") == "" {
		return newChatClient()
	}
	return chatClientFromEnv("MCSEARCH_SUMMARY_URL", "MCSEARCH_SUMMARY_MODEL", "MCSEARCH_SUMMARY_TIMEOUT",
		envOr("MCSEARCH_CHAT_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"), 120*time.Second)
}

// envInt reads a positive integer env var with a default.
// Non-positive or unparsable values fall back to def with a warning.
func envInt(name string, def int) int {
	raw := os.Getenv(name)
	if raw == "" {
		return def
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		fmt.Fprintf(os.Stderr, "warning: %s=%q is not a non-negative integer; using %d\n", name, raw, def)
		return def
	}
	return n
}

// rerankPool reads the candidate-pool cap from the environment.
// Default 40, clamped to [1, 100]. Larger = better recall but slower
// cross-encoder call. Only consulted when a Reranker is wired.
func rerankPool() int {
	raw := envOr("MCSEARCH_RERANK_POOL", "40")
	pool, err := strconv.Atoi(raw)
	if err != nil || pool <= 0 {
		fmt.Fprintf(os.Stderr, "warning: MCSEARCH_RERANK_POOL=%q is not a positive integer; using 40\n", raw)
		pool = 40
	}
	if pool > 100 {
		pool = 100
	}
	return pool
}

// ─── index ─────────────────────────────────────────────────────────────────

func cmdIndex(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("index", flag.ContinueOnError)
	setHelp(fs,
		"Build or refresh the per-project index (chunks + Go static graph).",
		"mcsearch index [flags] <path>")
	verbose := fs.Bool("v", false, "verbose")
	force := fs.Bool("force", false, "bypass protected-path and git-tree guards")
	summarize := fs.Bool("summarize", false, "generate per-file and per-chunk summaries via the chat endpoint (auto-enabled when MCSEARCH_SUMMARY_URL is set)")
	summarizeDefer := fs.Bool("summarize-defer", false, "queue summaries into pending_summaries instead of generating them inline; `mcsearch summarize` (or watch idle) drains the queue later. Implies --summarize. Chat endpoint not required at index time.")
	graphMode := fs.String("graph", "on", "graph phase: on|off|only ('on' runs both phases, 'off' skips graph, 'only' skips chunk/embed and just refreshes the graph)")
	format := fs.String("format", "text", "output format: text|json")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
		return err
	}
	switch *graphMode {
	case "on", "off", "only":
	default:
		return fmt.Errorf("invalid --graph=%s (want on|off|only)", *graphMode)
	}
	switch *format {
	case "text", "json":
	default:
		return fmt.Errorf("unknown --format=%s (want text|json)", *format)
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

	// Phase 1: chunk + embed (skipped when --graph=only).
	if *graphMode != "only" {
		ig, err := ignore.New(p.Root)
		if err != nil {
			return err
		}
		opts := index.Options{
			Verbose:     *verbose,
			Logger:      cliLogger(),
			Concurrency: envInt("MCSEARCH_INDEX_CONCURRENCY", 0),
		}
		// Auto-enable summarize when a dedicated summary endpoint is
		// configured. MCSEARCH_CHAT_URL is NOT a trigger — users often
		// set it for generate/ask_codebase without wanting per-chunk
		// chat calls on every index. Set --summarize, --summarize-defer,
		// or MCSEARCH_SUMMARY_URL explicitly to opt in.
		//
		// --summarize-defer implies --summarize and routes job dispatch
		// through pending_summaries instead of inline chat calls; the
		// chat client is unused at index time but we still wire it so
		// the future `mcsearch summarize` drainer can reuse the same
		// Options shape.
		if *summarize || *summarizeDefer || os.Getenv("MCSEARCH_SUMMARY_URL") != "" {
			opts.Summarize = true
			opts.DeferSummaries = *summarizeDefer
			opts.Chat = newSummaryClient()
			opts.SummaryConcurrency = envInt("MCSEARCH_SUMMARY_CONCURRENCY", 4)
			opts.ChunkSummaryMinLines = envInt("MCSEARCH_CHUNK_SUMMARY_MIN_LINES", 0)
		}
		ix := index.New(p, st, newEmbedClient(), ig, opts)
		if err := ix.Run(ctx); err != nil {
			return err
		}
	}
	if err := st.SetProjectRoot(ctx, p.Root); err != nil {
		return err
	}

	// Phase 2: graph extraction (skipped when --graph=off).
	// In --graph=only mode the user explicitly asked for the graph, so a
	// failure is hard. In default mode the chunk phase has already
	// succeeded, so we warn-and-continue — losing the graph shouldn't
	// invalidate a fresh embed pass.
	var gstats *graph.Stats
	if *graphMode != "off" {
		s, gerr := runGraphPhase(ctx, p, st, *verbose)
		if gerr != nil {
			if *graphMode == "only" {
				return gerr
			}
			fmt.Fprintf(os.Stderr, "⚠ graph phase failed: %v (chunk index is still usable)\n", gerr)
		} else {
			gstats = s
		}
	}

	if *graphMode == "only" {
		// Mirror the old `graph index` output shape so existing scripts
		// piping --format=json keep parsing.
		return reportGraphStats(p.Root, gstats, *format)
	}
	stats, err := st.Stats(ctx)
	if err != nil {
		return err
	}
	if *format == "json" {
		return reportIndexResult(p.Root, stats, gstats)
	}
	fmt.Printf("✓ indexed %s\n", p.Root)
	fmt.Printf("  chunks: %d  files: %d  dim: %d\n", stats.Chunks, stats.Files, stats.Dim)
	if gstats != nil {
		_ = reportGraphStats(p.Root, gstats, "text")
	}
	return nil
}

// indexResult is the JSON payload emitted by `index --format=json`
// (combined chunk + graph stats). The Graph field is omitted when
// the graph phase was skipped or failed.
type indexResult struct {
	Project string            `json:"project"`
	Chunks  int               `json:"chunks"`
	Files   int               `json:"files"`
	Dim     int               `json:"dim"`
	Graph   *graphIndexResult `json:"graph,omitempty"`
}

func reportIndexResult(project string, s store.Stats, g *graph.Stats) error {
	out := indexResult{
		Project: project,
		Chunks:  s.Chunks,
		Files:   s.Files,
		Dim:     s.Dim,
	}
	if g != nil {
		out.Graph = &graphIndexResult{
			Project:    project,
			Packages:   g.Packages,
			Nodes:      g.NodesUpserted,
			Edges:      g.EdgesUpserted,
			Pruned:     g.NodesPruned,
			PrunedEdge: g.EdgesPruned,
			Linked:     g.LinkedToChunks,
			ElapsedMS:  g.Elapsed.Milliseconds(),
			Warnings:   g.Warnings,
		}
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(out)
}

// ─── query ─────────────────────────────────────────────────────────────────

func cmdQuery(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("query", flag.ContinueOnError)
	setHelp(fs,
		"Search an existing index from the terminal (top-k chunks for a query).",
		"mcsearch query [flags] <path> <query...>")
	k := fs.Int("k", 8, "number of results to return")
	rerankFlag := fs.String("rerank", "", "set to 'off' to skip the rerank stage for this query (no effect when MCSEARCH_RERANK_URL is unset)")
	format := fs.String("format", "text", "output format: text | json")
	explain := fs.Bool("explain", false, "show per-chunk score breakdown and stage timings")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
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
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("no index for %s — run `mcsearch index %s` first", p.Root, p.Root)
		}
		return err
	}
	opts := storeOpts()
	if *rerankFlag == "off" {
		opts.Reranker = nil
	}
	st, err := store.OpenWith(ctx, p.DBPath, opts)
	if err != nil {
		return err
	}
	defer st.Close()
	em := newEmbedClient()
	t0 := time.Now()
	vecs, err := em.Embed(ctx, []string{q})
	embedDur := time.Since(t0)
	if err != nil {
		return err
	}
	t1 := time.Now()
	hits, err := st.Search(ctx, vecs[0], q, *k)
	searchDur := time.Since(t1)
	if err != nil {
		return err
	}
	if len(hits) == 0 {
		fmt.Fprintln(os.Stderr, "no results")
		return nil
	}
	switch *format {
	case "json":
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(hitsToJSON(hits))
	case "", "text":
		for i, h := range hits {
			loc := fmt.Sprintf("%s:%d-%d", h.Path, h.StartLine, h.EndLine)
			if h.Name != "" {
				loc = h.Name + "  " + loc
			}
			header := fmt.Sprintf("─── #%d %s  (%s)", i+1, loc, h.Kind)
			if *explain {
				scores := fmt.Sprintf("sem=%.4f", h.Score)
				if h.BM25Score > 0 {
					scores += fmt.Sprintf("  bm25=%.4f", h.BM25Score)
				}
				if h.RRFScore > 0 {
					scores += fmt.Sprintf("  rrf=%.4f", h.RRFScore)
				}
				if h.RerankScore > 0 {
					scores += fmt.Sprintf("  rerank=%.4f", h.RerankScore)
				}
				fmt.Println(header)
				fmt.Println("  " + scores)
			} else {
				header += fmt.Sprintf("  score=%.4f", h.Score)
				if h.RerankScore > 0 {
					header += fmt.Sprintf("  rerank=%.4f", h.RerankScore)
				}
				fmt.Println(header)
			}
			fmt.Println(truncate(h.Content, 1500))
			fmt.Println()
		}
		if *explain {
			fmt.Fprintf(os.Stderr, "timing:  embed=%dms  search=%dms  total=%dms\n",
				embedDur.Milliseconds(), searchDur.Milliseconds(), (embedDur + searchDur).Milliseconds())
		}
		return nil
	default:
		return fmt.Errorf("unknown format %q (want text|json)", *format)
	}
}

// ─── context ──────────────────────────────────────────────────────────────

// cmdContext mirrors the `mcsearch_context` MCP tool so agents and
// humans share one implementation. The flag surface maps 1-to-1 onto
// mcp.ContextInput so a CLI invocation can stand in for a tool call
// when an agent is offline.
func cmdContext(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("context", flag.ContinueOnError)
	setHelp(fs,
		"One-shot context router — composes semantic + symbol + graph; emit suggested_reads + next_action. Use this BEFORE grep loops.",
		"mcsearch context [flags] <path> <question...>")
	intent := fs.String("intent", "", "force a strategy: auto|behavior_search|symbol_lookup|callers|callees|architecture|package_topology|editing_context")
	k := fs.Int("k", 8, "max hits per lane (capped at 30)")
	format := fs.String("format", "text", "output format: text | json")
	noInline := fs.Bool("no-inline", false, "skip inlining raw file contents into suggested_reads (stored chunk/file summaries are still emitted; use --format=json to inspect)")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
		return err
	}
	if !validIntent(*intent) {
		return fmt.Errorf("invalid --intent=%q (want one of: auto, behavior_search, symbol_lookup, callers, callees, architecture, package_topology, editing_context)", *intent)
	}
	rest := fs.Args()
	if len(rest) < 2 {
		return fmt.Errorf("context needs <path> <question...>")
	}
	path := rest[0]
	question := strings.Join(rest[1:], " ")

	base, err := indexDir()
	if err != nil {
		return err
	}
	p, err := proj.Resolve(path, base)
	if err != nil {
		return err
	}

	s, _ := newServerFromEnv(base)
	_, out, err := s.ContextRouter(ctx, mcp.ContextInput{
		Project:  p.Root,
		Question: question,
		Intent:   *intent,
		K:        *k,
		NoInline: *noInline,
	})
	if err != nil {
		return err
	}

	switch *format {
	case "json":
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	case "", "text":
		printContextText(out)
		return nil
	default:
		return fmt.Errorf("unknown format %q (want text|json)", *format)
	}
}

// ─── generate ──────────────────────────────────────────────────────────────

func cmdGenerate(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("generate", flag.ContinueOnError)
	setHelp(fs,
		"Generate code grounded in the project's index (RAG: top-k chunks → chat endpoint).",
		"mcsearch generate [flags] <path> <prompt...>")
	k := fs.Int("k", 8, "number of RAG chunks to retrieve as context")
	noRAG := fs.Bool("no-rag", false, "skip retrieval; send prompt to the chat endpoint without project context")
	system := fs.String("system", "", "override the default system prompt")
	temp := fs.Float64("temperature", 0, "sampling temperature (0 = server default)")
	maxTok := fs.Int("max-tokens", 0, "max tokens to generate (0 = server default)")
	showCtx := fs.Bool("show-context", false, "print the chunks fed as context before the model output")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
		return err
	}
	rest := fs.Args()
	if len(rest) < 2 {
		return fmt.Errorf("generate needs <path> <prompt...>")
	}
	path := rest[0]
	prompt := strings.Join(rest[1:], " ")
	if strings.TrimSpace(prompt) == "" {
		return fmt.Errorf("prompt is empty")
	}

	base, err := indexDir()
	if err != nil {
		return err
	}
	p, err := proj.Resolve(path, base)
	if err != nil {
		return err
	}

	var hits []store.Hit
	if !*noRAG {
		if _, err := os.Stat(p.DBPath); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return fmt.Errorf("no index for %s — run `mcsearch index %s` first, or pass --no-rag to skip retrieval", p.Root, p.Root)
			}
			return err
		}
		st, err := openStore(ctx, p.DBPath)
		if err != nil {
			return err
		}
		em := newEmbedClient()
		vecs, err := em.Embed(ctx, []string{prompt})
		if err != nil {
			st.Close()
			return fmt.Errorf("embed: %w", err)
		}
		hits, err = st.Search(ctx, vecs[0], prompt, *k)
		st.Close()
		if err != nil {
			return fmt.Errorf("search: %w", err)
		}
	}

	sysPrompt := *system
	if strings.TrimSpace(sysPrompt) == "" {
		sysPrompt = "You are a precise coding assistant. " +
			"When CONTEXT chunks from the user's project are provided, ground your answer in them — " +
			"reference real symbols, paths, and conventions rather than inventing names. " +
			"Output code in fenced blocks; keep prose minimal."
	}

	userContent := prompt
	if len(hits) > 0 {
		userContent = store.FormatHits(hits) + "\n\n---\n\n" + prompt
	}

	if *showCtx && len(hits) > 0 {
		fmt.Fprintln(os.Stderr, "─── context fed to the model ───")
		for i, h := range hits {
			fmt.Fprintf(os.Stderr, "#%d %s:%d-%d  (%s)  score=%.4f\n", i+1, h.Path, h.StartLine, h.EndLine, h.Kind, h.Score)
		}
		fmt.Fprintln(os.Stderr, "────────────────────────────────")
	}

	cc := newChatClient()
	resp, err := cc.Generate(ctx, []chat.Message{
		{Role: "system", Content: sysPrompt},
		{Role: "user", Content: userContent},
	}, chat.Options{
		Temperature: float32(*temp),
		MaxTokens:   *maxTok,
	})
	if err != nil {
		return err
	}
	fmt.Println(resp.Content)
	if resp.FinishReason != "" && resp.FinishReason != "stop" {
		fmt.Fprintf(os.Stderr, "\n(finish_reason=%s)\n", resp.FinishReason)
	}
	return nil
}

// ─── status ────────────────────────────────────────────────────────────────

func cmdStatus(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("status", flag.ContinueOnError)
	setHelp(fs,
		"Show endpoint health and project stats (chunks/files/graph). Optional path narrows to one project.",
		"mcsearch status [<path>]")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
		return err
	}
	rest := fs.Args()
	checkCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	printEndpoints(checkCtx)
	fmt.Printf("mcsearch %s\n", mcp.Version)

	base, err := indexDir()
	if err != nil {
		return err
	}

	if len(rest) == 1 {
		// Per-project status
		p, err := proj.Resolve(rest[0], base)
		if err != nil {
			return err
		}
		if _, err := os.Stat(p.DBPath); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				fmt.Printf("\n%s\n  not indexed — run: mcsearch index %s\n", p.Root, p.Root)
				return nil
			}
			return err
		}
		st, err := openStore(ctx, p.DBPath)
		if err != nil {
			return err
		}
		defer st.Close()
		stats, err := st.Stats(ctx)
		if err != nil {
			return err
		}
		nodes, edges, gerr := st.GraphStats(ctx)
		fmt.Printf("\n%s\n", p.Root)
		fmt.Printf("  %d chunks  %d files  dim=%d\n", stats.Chunks, stats.Files, stats.Dim)
		if gerr == nil && (nodes > 0 || edges > 0) {
			fmt.Printf("  %d graph nodes  %d graph edges\n", nodes, edges)
		}
		if stats.LastIndex.IsZero() {
			fmt.Println("  last indexed: unknown (run mcsearch index to refresh)")
		} else if time.Since(stats.LastIndex) > 24*time.Hour {
			fmt.Printf("  last indexed: %s  ⚠ stale — run: mcsearch index %s\n",
				relativeTime(stats.LastIndex), p.Root)
		} else {
			fmt.Printf("  last indexed: %s\n", relativeTime(stats.LastIndex))
		}
		return nil
	}

	// All-project summary
	entries, err := os.ReadDir(base)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			fmt.Printf("\nindex dir: %s\nno projects indexed yet\n", base)
			return nil
		}
		return err
	}

	type row struct {
		root    string
		chunks  int
		files   int
		nodes   int64
		edges   int64
		last    time.Time
		corrupt bool
		empty   bool
	}
	results := make([]row, len(entries))
	sem := make(chan struct{}, 8)
	var wg sync.WaitGroup
	for i, e := range entries {
		if !e.IsDir() {
			continue
		}
		dbPath := filepath.Join(base, e.Name(), "index.db")
		if _, err := os.Stat(dbPath); err != nil {
			continue
		}
		wg.Add(1)
		go func(idx int, name, path string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			st, err := store.Open(ctx, path)
			if err != nil {
				results[idx] = row{root: fmt.Sprintf("(corrupt cache: %s)", name), corrupt: true}
				return
			}
			stats, _ := st.Stats(ctx)
			root, _ := st.ProjectRoot(ctx)
			nodes, edges, _ := st.GraphStats(ctx)
			st.Close()
			if stats.Chunks == 0 {
				results[idx] = row{empty: true}
				return
			}
			if root == "" {
				root = "? (run mcsearch index <path> to tag)"
			}
			results[idx] = row{
				root:   root,
				chunks: stats.Chunks,
				files:  stats.Files,
				nodes:  nodes,
				edges:  edges,
				last:   stats.LastIndex,
			}
		}(i, e.Name(), dbPath)
	}
	wg.Wait()

	var rows []row
	var empties int
	for _, r := range results {
		switch {
		case r.empty:
			empties++
		case r.root != "":
			rows = append(rows, r)
		}
	}

	if len(rows) == 0 && empties == 0 {
		fmt.Printf("\nindex dir: %s\nno projects indexed yet\n", base)
		return nil
	}

	// Compute width of the path column for alignment.
	maxRoot := 0
	for _, r := range rows {
		if len(r.root) > maxRoot {
			maxRoot = len(r.root)
		}
	}
	if maxRoot > 60 {
		maxRoot = 60
	}

	fmt.Println()
	for _, r := range rows {
		if r.corrupt {
			fmt.Printf("  %-*s  CORRUPT\n", maxRoot, r.root)
			continue
		}
		var when string
		switch {
		case r.last.IsZero():
			when = "no timestamp"
		case time.Since(r.last) > 24*time.Hour:
			when = "⚠ " + relativeTime(r.last)
		default:
			when = relativeTime(r.last)
		}
		fmt.Printf("  %-*s  %5d chunks  %4d files  %s\n",
			maxRoot, r.root, r.chunks, r.files, when)
		if r.nodes > 0 || r.edges > 0 {
			fmt.Printf("  %-*s  %5d nodes   %4d edges (graph)\n",
				maxRoot, "", r.nodes, r.edges)
		}
	}
	if empties > 0 {
		noun := "index"
		if empties != 1 {
			noun = "indexes"
		}
		fmt.Printf("  (%d empty %s skipped)\n", empties, noun)
	}
	return nil
}

// ─── nuke ──────────────────────────────────────────────────────────────────

func cmdNuke(_ context.Context, args []string) error {
	fs := flag.NewFlagSet("nuke", flag.ContinueOnError)
	setHelp(fs,
		"Delete the on-disk index for a project (irreversible).",
		"mcsearch nuke <path>")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
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
		if errors.Is(err, os.ErrNotExist) {
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

// ─── reindex ───────────────────────────────────────────────────────────────

func cmdReindex(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("reindex", flag.ContinueOnError)
	setHelp(fs,
		"Drop and re-embed a project from scratch (or every known project with --all --yes).",
		"mcsearch reindex [flags] <path>  |  mcsearch reindex --all --yes")
	all := fs.Bool("all", false, "drop and re-index every known project under MCSEARCH_INDEX_DIR")
	yes := fs.Bool("yes", false, "confirm the destructive sweep required by --all")
	verbose := fs.Bool("v", false, "verbose")
	force := fs.Bool("force", false, "bypass protected-path and git-tree guards")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
		return err
	}
	base, err := indexDir()
	if err != nil {
		return err
	}
	rest := fs.Args()

	if *all {
		if len(rest) != 0 {
			return fmt.Errorf("reindex --all takes no path argument")
		}
		if !*yes {
			return fmt.Errorf("reindex --all drops every project index and re-embeds from scratch; pass --yes to confirm")
		}
		roots, err := knownProjectRoots(ctx, base)
		if err != nil {
			return err
		}
		if len(roots) == 0 {
			fmt.Printf("nothing to reindex under %s\n", base)
			return nil
		}
		var failed []string
		for _, root := range roots {
			fmt.Printf("→ reindexing %s\n", root)
			if err := reindexOne(ctx, root, base, *verbose, *force); err != nil {
				fmt.Fprintf(os.Stderr, "  ✗ %v\n", err)
				failed = append(failed, root)
			}
		}
		if len(failed) > 0 {
			return fmt.Errorf("%d of %d project(s) failed to reindex", len(failed), len(roots))
		}
		return nil
	}

	if len(rest) != 1 {
		return fmt.Errorf("reindex needs exactly one path argument (or --all)")
	}
	return reindexOne(ctx, rest[0], base, *verbose, *force)
}

// reindexOne drops the existing per-project cache dir and re-runs the
// indexer from scratch. Used by both `reindex <path>` and the loop in
// `reindex --all`.
func reindexOne(ctx context.Context, root, base string, verbose, force bool) error {
	p, err := proj.Resolve(root, base)
	if err != nil {
		return err
	}
	if err := proj.CheckIndexable(p, force); err != nil {
		return err
	}
	if _, err := os.Stat(p.CacheDir); err == nil {
		if err := os.RemoveAll(p.CacheDir); err != nil {
			return err
		}
	} else if !errors.Is(err, os.ErrNotExist) {
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
	ig, err := ignore.New(p.Root)
	if err != nil {
		return err
	}
	ix := index.New(p, st, newEmbedClient(), ig, index.Options{Verbose: verbose, Logger: cliLogger(), Concurrency: envInt("MCSEARCH_INDEX_CONCURRENCY", 0)})
	if err := ix.Run(ctx); err != nil {
		return err
	}
	if err := st.SetProjectRoot(ctx, p.Root); err != nil {
		return err
	}
	gstats, gerr := runGraphPhase(ctx, p, st, verbose)
	if gerr != nil {
		fmt.Fprintf(os.Stderr, "⚠ graph phase failed for %s: %v (chunk index is still usable)\n", p.Root, gerr)
	}
	stats, err := st.Stats(ctx)
	if err != nil {
		return err
	}
	fmt.Printf("✓ reindexed %s\n", p.Root)
	fmt.Printf("  chunks: %d  files: %d  dim: %d\n", stats.Chunks, stats.Files, stats.Dim)
	if gstats != nil {
		_ = reportGraphStats(p.Root, gstats, "text")
	}
	return nil
}

// knownProjectRoots walks the index dir, opening each per-project index
// and reading the recorded `project_root` meta. Entries written before
// that meta existed are reported to stderr and skipped — the user can
// `mcsearch nuke <path>` + `mcsearch index <path>` once to re-record it.
func knownProjectRoots(ctx context.Context, base string) ([]string, error) {
	entries, err := os.ReadDir(base)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var roots []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dbPath := filepath.Join(base, e.Name(), "index.db")
		if _, err := os.Stat(dbPath); err != nil {
			continue
		}
		st, err := openStore(ctx, dbPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: skipping %s: open: %v\n", e.Name(), err)
			continue
		}
		root, err := st.ProjectRoot(ctx)
		st.Close()
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: skipping %s: %v\n", e.Name(), err)
			continue
		}
		if root == "" {
			fmt.Fprintf(os.Stderr, "warning: skipping %s: no recorded project_root (pre-migration index)\n", e.Name())
			continue
		}
		roots = append(roots, root)
	}
	return roots, nil
}

// ─── watch ─────────────────────────────────────────────────────────────────

func cmdWatch(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("watch", flag.ContinueOnError)
	setHelp(fs,
		"Keep the index fresh as files change (foreground; runs chunk + graph after each debounce).",
		"mcsearch watch [flags] <path>")
	verbose := fs.Bool("v", false, "verbose")
	force := fs.Bool("force", false, "bypass protected-path and git-tree guards")
	debounce := fs.Duration("debounce", 500*time.Millisecond, "quiet window before re-indexing")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
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
	ig, err := ignore.New(p.Root)
	if err != nil {
		return err
	}
	logger := cliLogger()
	ix := index.New(p, st, newEmbedClient(), ig, index.Options{Verbose: *verbose, Logger: logger, Concurrency: envInt("MCSEARCH_INDEX_CONCURRENCY", 0)})
	// Refresh the Go static graph after each chunk-index flush. The
	// graph layer lives in the same SQLite file, so the chunk run has
	// already released the writer when this fires.
	afterIndex := func(c context.Context) error {
		_, err := runGraphPhase(c, p, st, *verbose)
		return err
	}
	w := watch.New(ix, ig, p.Root, watch.Options{
		Debounce:   *debounce,
		Verbose:    *verbose,
		Logger:     logger,
		AfterIndex: afterIndex,
	})
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
	setHelp(fs,
		"Seed dst's index from src's (e.g. for a new git worktree). Follow with `mcsearch index <dst>` to reconcile.",
		"mcsearch clone [flags] <src-path> <dst-path>")
	force := fs.Bool("force", false, "overwrite dst's index if it already exists")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
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
		if errors.Is(err, os.ErrNotExist) {
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
	// Re-tag project_root so `reindex --all` / status see this cache
	// as belonging to dst, not src. A subsequent `mcsearch index <dst>`
	// would also do this, but tagging now keeps the cache discoverable
	// even before the first reconcile.
	if err := retagProjectRoot(ctx, dst.DBPath, dst.Root); err != nil {
		return fmt.Errorf("retag project root: %w", err)
	}
	fmt.Printf("✓ cloned %s → %s\n", src.Root, dst.Root)
	fmt.Printf("  next: `mcsearch index %s` will reconcile any files that differ between the two trees (incremental — only changed chunks are re-embedded).\n", dst.Root)
	return nil
}

// retagProjectRoot opens the cloned DB just long enough to overwrite
// the project_root meta key, so the dst cache no longer claims to be
// src's index.
func retagProjectRoot(ctx context.Context, dbPath, root string) error {
	st, err := store.Open(ctx, dbPath)
	if err != nil {
		return err
	}
	defer st.Close()
	return st.SetProjectRoot(ctx, root)
}

func copyFile(srcPath, dstPath string) error {
	// Hard-link is instant when src and dst are on the same filesystem.
	if err := os.Link(srcPath, dstPath); err == nil {
		return nil
	}
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
	fs := flag.NewFlagSet("mcp", flag.ContinueOnError)
	setHelp(fs,
		"Run mcsearch as an MCP server over stdio (canonical entrypoint for Claude Code).",
		"mcsearch mcp")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
		return err
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("mcp takes no arguments (got %v)", fs.Args())
	}
	base, err := indexDir()
	if err != nil {
		return err
	}
	srv, _ := newServerFromEnv(base)
	return srv.RunStdio(ctx)
}

// newServerFromEnv builds a fully-wired *mcp.Server from the current
// environment. Used by both `cmdMCP` (stdio server) and `cmdContext`
// (one-shot CLI invocation of the context router). The HTTP clients
// are lazy — they don't dial until invoked — so wiring all of them
// is cheap even when the context router only uses Embed.
//
// Returns the shared rerank client as the second value so callers
// that need it for separate purposes (e.g. health reporting) don't
// have to redundantly construct another instance.
func newServerFromEnv(base string) (*mcp.Server, rerank.HealthChecker) {
	rerankClient := newRerankClient()
	opts := storeOpts()
	if rerankClient != nil {
		opts.Reranker = rerankClient
	}
	srv := &mcp.Server{
		EmbedClient:    newEmbedClient(),
		ChatClient:     newChatClient(),
		RerankClient:   rerankClient,
		CompressClient: newCompressClient(),
		DraftClient:    newDraftClient(),
		IndexDir:       base,
		StoreOpts:      opts,
	}
	return srv, rerankClient
}
