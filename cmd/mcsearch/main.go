// mcsearch — local semantic-search helper for Claude Code.
//
// Subcommands:
//
//	index <path>             Build or refresh the per-project index.
//	query <path> <query...>  Search an existing index from the terminal.
//	generate <path> <prompt> Generate code grounded in the project's index.
//	status [<path>]          Show endpoint health and indexed projects.
//	nuke <path>              Delete the on-disk index for a project.
//	watch <path>             Keep the index fresh as files change.
//	clone <src> <dst>        Seed dst's index from src's (worktrees).
//	mcp                      Run as an MCP server over stdio.
//	version                  Print the build version.
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
	"syscall"
	"time"

	"github.com/alehatsman/mcsearch/internal/chat"
	"github.com/alehatsman/mcsearch/internal/embed"
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
	case "generate":
		err = cmdGenerate(ctx, args)
	case "status":
		err = cmdStatus(ctx, args)
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
  mcsearch generate <path> <prompt> generate code grounded in the project's
                                    index (RAG: top-k chunks → chat endpoint)
  mcsearch status [<path>]          show endpoint health and project stats
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

env:
  MCSEARCH_EMBED_URL          default http://127.0.0.1:8082
  MCSEARCH_EMBED_MODEL        default Qwen/Qwen3-Embedding-4B
  MCSEARCH_EMBED_BATCH        default 32
  MCSEARCH_EMBED_TIMEOUT      default 60s (Go duration)
  MCSEARCH_CHAT_URL           default http://127.0.0.1:8081
  MCSEARCH_CHAT_MODEL         default Qwen/Qwen2.5-Coder-7B-Instruct
  MCSEARCH_CHAT_TIMEOUT       default 120s (Go duration)
  MCSEARCH_RERANK_URL         unset by default; set to enable reranking
  MCSEARCH_RERANK_STYLE       "cohere" (default) or "chat" (decoder-style, e.g. Qwen3-Reranker-4B via vLLM)
  MCSEARCH_RERANK_MODEL       default BAAI/bge-reranker-v2-m3
  MCSEARCH_RERANK_POOL        default 40 (candidates before rerank; clamped to 1..100)
  MCSEARCH_RERANK_TIMEOUT     default 5s (Go duration)
  MCSEARCH_RERANK_CONCURRENCY default 4 (concurrent scoring goroutines, chat style only)
  MCSEARCH_DISABLE_RERANK     set to 1 to short-circuit rerank even if URL is set
  MCSEARCH_COMPRESS_URL       unset by default; set to enable context compression for ask/generate
  MCSEARCH_COMPRESS_MODEL     default: value of MCSEARCH_CHAT_MODEL
  MCSEARCH_COMPRESS_TIMEOUT   default 30s (Go duration)
  MCSEARCH_DRAFT_URL          unset by default; set to enable speculative local draft for generate_code
  MCSEARCH_DRAFT_MODEL        default: value of MCSEARCH_CHAT_MODEL
  MCSEARCH_DRAFT_TIMEOUT      default 120s (Go duration)
  MCSEARCH_INDEX_DIR          default ~/.cache/mcsearch
  MCSEARCH_DISABLE_VEC_CACHE  set to 1 to skip the in-RAM vector cache
  MCSEARCH_DISABLE_BM25       set to 1 to disable the lexical (BM25) leg
  MCSEARCH_MAX_HITS_PER_FILE  max hits per file in search results (0 = no cap)

flags:
  mcsearch query --rerank=off <path> "..."     skip rerank for this call
  mcsearch query --format=json <path> "..."    emit hits as JSON
  mcsearch query --explain <path> "..."        show per-chunk score breakdown and stage timings`)
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
		DisableVecCache: os.Getenv("MCSEARCH_DISABLE_VEC_CACHE") == "1",
		DisableBM25:     os.Getenv("MCSEARCH_DISABLE_BM25") == "1",
		RerankPool:      rerankPool(),
		MaxHitsPerFile:  maxHitsPerFile(),
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
	timeout := parseDuration("MCSEARCH_EMBED_TIMEOUT", envOr("MCSEARCH_EMBED_TIMEOUT", "60s"), 60*time.Second)
	return embed.New(url, model, batch, timeout)
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

// newCompressClient returns a chat client aimed at a fast local model
// that distils retrieved chunks before they enter the chat model's
// context window. Returns nil when MCSEARCH_COMPRESS_URL is unset,
// disabling the compression stage (today's default behaviour).
func newCompressClient() *chat.Client {
	url := os.Getenv("MCSEARCH_COMPRESS_URL")
	if url == "" {
		return nil
	}
	model := envOr("MCSEARCH_COMPRESS_MODEL", envOr("MCSEARCH_CHAT_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"))
	timeout := parseDuration("MCSEARCH_COMPRESS_TIMEOUT", envOr("MCSEARCH_COMPRESS_TIMEOUT", "30s"), 30*time.Second)
	return chat.New(url, model, timeout)
}

// newDraftClient returns a chat client for the speculative local draft
// stage of generate_code. Returns nil when MCSEARCH_DRAFT_URL is unset,
// leaving generate_code on the standard single-model RAG path.
func newDraftClient() *chat.Client {
	url := os.Getenv("MCSEARCH_DRAFT_URL")
	if url == "" {
		return nil
	}
	model := envOr("MCSEARCH_DRAFT_MODEL", envOr("MCSEARCH_CHAT_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"))
	timeout := parseDuration("MCSEARCH_DRAFT_TIMEOUT", envOr("MCSEARCH_DRAFT_TIMEOUT", "120s"), 120*time.Second)
	return chat.New(url, model, timeout)
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
	verbose := fs.Bool("v", false, "verbose")
	force := fs.Bool("force", false, "bypass protected-path and git-tree guards")
	summarize := fs.Bool("summarize", false, "also generate a per-file summary via the chat endpoint (slow; opt-in)")
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
	opts := index.Options{Verbose: *verbose, Logger: cliLogger()}
	if *summarize {
		opts.Summarize = true
		opts.Chat = newChatClient()
	}
	ix := index.New(p, st, newEmbedClient(), ig, opts)
	if err := ix.Run(ctx); err != nil {
		return err
	}
	if err := st.SetProjectRoot(ctx, p.Root); err != nil {
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
	rerankFlag := fs.String("rerank", "", "set to 'off' to skip the rerank stage for this query (no effect when MCSEARCH_RERANK_URL is unset)")
	format := fs.String("format", "text", "output format: text | json")
	explain := fs.Bool("explain", false, "show per-chunk score breakdown and stage timings")
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

// queryJSONHit is the wire shape for `mcsearch query --format=json`.
// Mirrors mcp.SearchHit so the two CLI/MCP surfaces stay aligned.
type queryJSONHit struct {
	Path        string  `json:"path"`
	Kind        string  `json:"kind"`
	StartLine   int     `json:"start_line"`
	EndLine     int     `json:"end_line"`
	Score       float32 `json:"score"`
	BM25Score   float32 `json:"bm25_score,omitempty"`
	RRFScore    float32 `json:"rrf_score,omitempty"`
	RerankScore float32 `json:"rerank_score,omitempty"`
	Content     string  `json:"content"`
}

func hitsToJSON(hits []store.Hit) []queryJSONHit {
	out := make([]queryJSONHit, len(hits))
	for i, h := range hits {
		out[i] = queryJSONHit{
			Path:        h.Path,
			Kind:        h.Kind,
			StartLine:   h.StartLine,
			EndLine:     h.EndLine,
			Score:       h.Score,
			BM25Score:   h.BM25Score,
			RRFScore:    h.RRFScore,
			RerankScore: h.RerankScore,
			Content:     h.Content,
		}
	}
	return out
}

// ─── generate ──────────────────────────────────────────────────────────────

func cmdGenerate(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("generate", flag.ContinueOnError)
	k := fs.Int("k", 8, "number of RAG chunks to retrieve as context")
	noRAG := fs.Bool("no-rag", false, "skip retrieval; send prompt to the chat endpoint without project context")
	system := fs.String("system", "", "override the default system prompt")
	temp := fs.Float64("temperature", 0, "sampling temperature (0 = server default)")
	maxTok := fs.Int("max-tokens", 0, "max tokens to generate (0 = server default)")
	showCtx := fs.Bool("show-context", false, "print the chunks fed as context before the model output")
	if err := fs.Parse(args); err != nil {
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

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	// Snap to a UTF-8 boundary so we don't cut a multi-byte rune in
	// half and emit invalid UTF-8 to the terminal.
	cut := n
	for cut > 0 && (s[cut]&0xC0) == 0x80 {
		cut--
	}
	return s[:cut] + "\n…(truncated)"
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
	embedOK := em.Health(checkCtx) == nil
	if embedOK {
		fmt.Printf("embed  %s  %s  ok\n", em.BaseURL, em.Model)
	} else {
		fmt.Printf("embed  %s  %s  UNREACHABLE\n", em.BaseURL, em.Model)
	}
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
		fmt.Printf("\n%s\n", p.Root)
		fmt.Printf("  %d chunks  %d files  dim=%d\n", stats.Chunks, stats.Files, stats.Dim)
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
		last    time.Time
		corrupt bool
	}
	var rows []row
	var empties int
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dbPath := filepath.Join(base, e.Name(), "index.db")
		if _, err := os.Stat(dbPath); err != nil {
			continue
		}
		st, err := store.Open(ctx, dbPath)
		if err != nil {
			rows = append(rows, row{root: e.Name()[:min(12, len(e.Name()))], corrupt: true})
			continue
		}
		stats, _ := st.Stats(ctx)
		root, _ := st.ProjectRoot(ctx)
		st.Close()
		if stats.Chunks == 0 {
			empties++
			continue
		}
		if root == "" {
			root = "? (run mcsearch index <path> to tag)"
		}
		rows = append(rows, row{
			root:   root,
			chunks: stats.Chunks,
			files:  stats.Files,
			last:   stats.LastIndex,
		})
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

// relativeTime formats a timestamp as a human-friendly relative string
// ("just now", "5m ago", "2h ago", "3d ago", or a date for old entries).
func relativeTime(t time.Time) string {
	if t.IsZero() {
		return "never"
	}
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	case d < 7*24*time.Hour:
		return fmt.Sprintf("%dd ago", int(d.Hours()/24))
	default:
		return t.Format("2006-01-02")
	}
}

// ─── nuke ──────────────────────────────────────────────────────────────────

func cmdNuke(_ context.Context, args []string) error {
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
	all := fs.Bool("all", false, "drop and re-index every known project under MCSEARCH_INDEX_DIR")
	yes := fs.Bool("yes", false, "confirm the destructive sweep required by --all")
	verbose := fs.Bool("v", false, "verbose")
	force := fs.Bool("force", false, "bypass protected-path and git-tree guards")
	if err := fs.Parse(args); err != nil {
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
	ix := index.New(p, st, newEmbedClient(), ig, index.Options{Verbose: verbose, Logger: cliLogger()})
	if err := ix.Run(ctx); err != nil {
		return err
	}
	if err := st.SetProjectRoot(ctx, p.Root); err != nil {
		return err
	}
	stats, err := st.Stats(ctx)
	if err != nil {
		return err
	}
	fmt.Printf("✓ reindexed %s\n", p.Root)
	fmt.Printf("  chunks: %d  files: %d  dim: %d\n", stats.Chunks, stats.Files, stats.Dim)
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
	verbose := fs.Bool("v", false, "verbose")
	force := fs.Bool("force", false, "bypass protected-path and git-tree guards")
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
	ix := index.New(p, st, newEmbedClient(), ig, index.Options{Verbose: *verbose, Logger: logger})
	w := watch.New(ix, ig, p.Root, watch.Options{Debounce: *debounce, Verbose: *verbose, Logger: logger})
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
func cmdClone(_ context.Context, args []string) error {
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
	fs := flag.NewFlagSet("mcp", flag.ContinueOnError)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() > 0 {
		return fmt.Errorf("mcp takes no arguments (got %v)", fs.Args())
	}
	base, err := indexDir()
	if err != nil {
		return err
	}
	// Build the rerank client once and share it with both the Server
	// (for `status` health reporting) and Store.Options (for the
	// query-time rerank stage). storeOpts() would call newRerankClient
	// internally, but reusing one instance avoids the redundant HTTP
	// client allocation.
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
	return srv.RunStdio(ctx)
}
