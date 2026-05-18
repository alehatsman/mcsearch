// Package index orchestrates the walk → chunk → embed → upsert pipeline.
package index

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/alehatsman/mcsearch/internal/chat"
	"github.com/alehatsman/mcsearch/internal/chunk"
	"github.com/alehatsman/mcsearch/internal/embed"
	"github.com/alehatsman/mcsearch/internal/ignore"
	"github.com/alehatsman/mcsearch/internal/proj"
	"github.com/alehatsman/mcsearch/internal/store"
)

// summarizeCap bounds the slice we send to the chat endpoint per file.
// Beyond this, summary quality degrades and latency spikes on local
// hardware; whole-repo overviews belong in ask_codebase, not here.
const summarizeCap = 64 * 1024

// chunkSummaryMinLines: skip per-chunk summaries for tiny chunks (< this
// many lines) — they're too short to need prose distillation and the
// full source text is better context than a summary.
const chunkSummaryMinLines = 30

// Options controls one index run.
type Options struct {
	MaxFileSize int64        // skip files larger than this (bytes); 0 = 1 MB default
	Verbose     bool         // emit per-event log lines (skip/embedding/etc.)
	Logger      *slog.Logger // destination for log output; nil = io.Discard
	// Summarize, when true and Chat is non-nil, generates a one-paragraph
	// per-file summary alongside the normal chunks. Summaries are stored
	// as `kind="file_summary"` chunks keyed by SHA of the file's first
	// summarizeCap bytes — so a re-index over unchanged content skips
	// the chat call entirely. Off by default: each summary costs one
	// chat round-trip per file.
	Summarize bool
	Chat      *chat.Client
}

// Indexer is the entry point.
type Indexer struct {
	Proj    *proj.Project
	Store   *store.Store
	Embed   *embed.Client
	Ignore  *ignore.Matcher
	Options Options
}

func New(p *proj.Project, st *store.Store, em *embed.Client, ig *ignore.Matcher, opt Options) *Indexer {
	if opt.MaxFileSize <= 0 {
		opt.MaxFileSize = 1 << 20 // 1 MB
	}
	if opt.Logger == nil {
		opt.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	return &Indexer{Proj: p, Store: st, Embed: em, Ignore: ig, Options: opt}
}

// Run walks the project, chunks new/changed files, embeds, and upserts.
// Files unchanged since the last index get their last_seen_at bumped but
// are not re-embedded. Stale rows (files removed) are pruned at the end.
//
// Mtime fast-path: if a file's mtime is <= the previous run's
// last_indexed_at, we know the content is identical to what we
// processed last time. We TouchPath() all of its chunks in one UPDATE
// and skip the read+parse+SHA work entirely — turning the no-change
// re-index from O(files × parse) into O(files × stat + 1 UPDATE).
func (ix *Indexer) Run(ctx context.Context) error {
	startTime := time.Now()
	var (
		toEmbed     []pending
		seen        int
		skipped     int
		mtimeSkips  int
	)

	prevStats, statsErr := ix.Store.Stats(ctx)
	var lastIndexed time.Time
	if statsErr == nil {
		lastIndexed = prevStats.LastIndex
	}

	err := filepath.WalkDir(ix.Proj.Root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if ix.Options.Verbose {
				ix.Options.Logger.Info("walk error", "path", path, "err", err)
			}
			return nil
		}
		// Honour context cancellation between files — useful for Ctrl-C
		// in CLI mode and for shutdown in watch mode.
		if cerr := ctx.Err(); cerr != nil {
			return cerr
		}
		rel, _ := filepath.Rel(ix.Proj.Root, path)
		if rel == "." {
			return nil
		}
		if ix.Ignore.Match(rel, d.IsDir()) {
			if d.IsDir() {
				// If the user just added e.g. `node_modules/` to
				// `.gitignore` between runs, the chunks under that
				// directory must be evicted on this pass — otherwise
				// they'd live forever in the index because the
				// directory is no longer walked. Drop them by path
				// prefix; the walk continues skipping the subtree.
				_ = ix.Store.DeletePathPrefix(ctx, rel+"/")
				return filepath.SkipDir
			}
			// Newly-ignored single file: drop its chunks.
			_ = ix.Store.DeletePath(ctx, rel)
			return nil
		}
		if d.IsDir() {
			return nil
		}
		// Skip symlinks. They risk (a) double-indexing the same content
		// under both the link path and the target path, and (b)
		// silently pulling content from outside the project root for
		// links into the file system. Operators who want symlinked
		// trees indexed should follow them at the shell level.
		if d.Type()&os.ModeSymlink != 0 {
			return nil
		}
		if !ignore.IndexableExt(path) && !ignore.IndexableBasename(path) {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		if info.Size() > ix.Options.MaxFileSize {
			skipped++
			if ix.Options.Verbose {
				ix.Options.Logger.Info("skip (too large)", "path", rel, "size", info.Size())
			}
			return nil
		}
		// Mtime fast-path: file hasn't changed since the last successful
		// index → just bump last_seen_at on its existing chunks.
		// Skipped when -summarize is set: we need to read the file to
		// discover which chunks still lack a summary and generate them.
		// The SHA-based summary cache prevents re-generating existing ones.
		if !ix.Options.Summarize && !lastIndexed.IsZero() && !info.ModTime().After(lastIndexed) {
			rows, terr := ix.Store.TouchPath(ctx, rel, startTime)
			if terr == nil && rows > 0 {
				seen += int(rows)
				mtimeSkips++
				return nil
			}
			// rows==0: never indexed before, fall through to slow path.
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil
		}
		if ignore.LooksBinary(data) {
			skipped++
			return nil
		}
		// Skip files whose content matches a known secret pattern —
		// but allow-list test fixtures, which routinely embed fake
		// credentials as inputs to their own detection logic.
		if !ignore.IsTestPath(rel) && ignore.LooksLikeSecret(data) {
			ix.Options.Logger.Warn("skip (matches secret pattern)", "path", rel)
			skipped++
			return nil
		}

		chunks, err := chunk.Chunks(ctx, rel, data)
		if err != nil {
			if ix.Options.Verbose {
				ix.Options.Logger.Info("chunk error", "path", rel, "err", err)
			}
			return nil
		}
		seen += len(chunks)

		existing, err := ix.Store.ExistingSHAs(ctx, rel)
		if err != nil {
			return err
		}
		for _, c := range chunks {
			sha := chunkSHA(c.Content)
			if existing[sha] {
				if err := ix.Store.TouchSeen(ctx, rel, sha, startTime); err != nil {
					return err
				}
				continue
			}
			toEmbed = append(toEmbed, pending{rel: rel, chunk: c, sha: sha})
		}

		// Per-file summary, opt-in. SHA is computed on the slice we'd
		// actually send to the model (so a file growing past the cap
		// doesn't re-summarize on every run). If we already have a
		// summary chunk for this exact slice, just bump last_seen_at
		// and skip the chat round-trip.
		if ix.Options.Summarize && ix.Options.Chat != nil && len(chunks) > 0 {
			slice := data
			truncated := false
			if len(slice) > summarizeCap {
				slice = slice[:summarizeCap]
				truncated = true
			}
			fileSHA := chunkSHA(string(slice))
			if existing[fileSHA] {
				if err := ix.Store.TouchSeen(ctx, rel, fileSHA, startTime); err != nil {
					return err
				}
				seen++
			} else {
				summary, err := summarizeFile(ctx, ix.Options.Chat, rel, slice)
				if err != nil {
					ix.Options.Logger.Warn("summarize failed", "path", rel, "err", err)
				} else if strings.TrimSpace(summary) != "" {
					if ix.Options.Verbose && truncated {
						ix.Options.Logger.Info("summarize truncated", "path", rel, "size", len(data))
					}
					toEmbed = append(toEmbed, pending{
						rel: rel,
						chunk: chunk.Chunk{
							Path:      rel,
							Kind:      "file_summary",
							StartLine: 1,
							EndLine:   lineCount(data),
							Content:   summary,
						},
						sha: fileSHA,
					})
					seen++
				}
			}
		}
		// Per-chunk summary, opt-in. Only structural chunks (functions,
		// methods, classes) with ≥ chunkSummaryMinLines lines get summaries;
		// tiny helpers, windows, and orphans aren't worth the round-trip.
		// SHA is keyed on the chunk source text so cache invalidation is
		// automatic when the function body changes.
		if ix.Options.Summarize && ix.Options.Chat != nil {
			for _, c := range chunks {
				if !isStructural(c.Kind) || (c.EndLine-c.StartLine+1) < chunkSummaryMinLines {
					continue
				}
				sumSHA := chunkSHA("chunk_summary:" + c.Content)
				if existing[sumSHA] {
					if err := ix.Store.TouchSeen(ctx, rel, sumSHA, startTime); err != nil {
						return err
					}
					seen++
					continue
				}
				summary, err := summarizeChunk(ctx, ix.Options.Chat, rel, c)
				if err != nil {
					ix.Options.Logger.Warn("chunk summarize failed", "path", rel, "start", c.StartLine, "err", err)
					continue
				}
				if strings.TrimSpace(summary) == "" {
					continue
				}
				toEmbed = append(toEmbed, pending{
					rel: rel,
					chunk: chunk.Chunk{
						Path:      rel,
						Kind:      "chunk_summary",
						StartLine: c.StartLine,
						EndLine:   c.EndLine,
						Content:   summary,
					},
					sha: sumSHA,
				})
				seen++
			}
		}
		// Old rows for this file whose SHA disappeared get pruned at the
		// end via PruneUnseen — they never had last_seen_at bumped on this
		// run.
		return nil
	})
	if err != nil {
		return fmt.Errorf("walk: %w", err)
	}

	if len(toEmbed) > 0 {
		if ix.Options.Verbose {
			ix.Options.Logger.Info("embedding chunks", "count", len(toEmbed))
		}
		// Embed and upsert one batch at a time. If a later batch fails
		// (timeout, embedding service crash), earlier batches survive
		// in the store and the next index run skips them via
		// content-sha matching — no wasted GPU time on retry.
		batchSize := ix.Embed.Batch
		if batchSize <= 0 {
			batchSize = 32
		}
		for start := 0; start < len(toEmbed); start += batchSize {
			end := start + batchSize
			if end > len(toEmbed) {
				end = len(toEmbed)
			}
			batch := toEmbed[start:end]
			texts := make([]string, len(batch))
			for i, p := range batch {
				texts[i] = p.chunk.EmbedText()
			}
			vecs, err := ix.Embed.Embed(ctx, texts)
			if err != nil {
				return fmt.Errorf("embed: %w", err)
			}
			rows := make([]store.PendingChunk, len(batch))
			for i, p := range batch {
				rows[i] = store.PendingChunk{
					Path:       p.rel,
					Kind:       p.chunk.Kind,
					StartLine:  p.chunk.StartLine,
					EndLine:    p.chunk.EndLine,
					ContentSHA: p.sha,
					Content:    p.chunk.Content,
					Vec:        vecs[i],
				}
			}
			if err := ix.Store.UpsertMany(ctx, rows, startTime); err != nil {
				return err
			}
		}
	}

	pruned, err := ix.Store.PruneUnseen(ctx, startTime)
	if err != nil {
		return err
	}
	if ix.Options.Verbose && pruned > 0 {
		ix.Options.Logger.Info("pruned stale chunks (files removed since last index)", "count", pruned)
	}
	if err := ix.Store.SetLastIndexedAt(ctx, startTime); err != nil {
		return err
	}
	if ix.Options.Verbose {
		ix.Options.Logger.Info("indexed",
			"chunks_seen", seen,
			"files_fast_path", mtimeSkips,
			"embedded", len(toEmbed),
			"pruned", pruned,
			"skipped", skipped)
	}
	return nil
}

type pending struct {
	rel   string
	chunk chunk.Chunk
	sha   string
}

func chunkSHA(content string) string {
	h := sha1.Sum([]byte(content))
	return hex.EncodeToString(h[:])
}

// isStructural returns true for chunk kinds that represent a named code
// entity (function, method, class, type, impl block, etc.). Returns false
// for window/orphan/summary pseudo-kinds that don't warrant their own
// prose summary.
func isStructural(kind string) bool {
	switch kind {
	case "window", "orphan", "file_summary", "chunk_summary":
		return false
	default:
		return true
	}
}

// summarizeChunk asks the chat endpoint for a 1–2 sentence description of
// a single function, method, or class. Returns the summary text or an error.
// Caller logs and skips on error so one bad chunk doesn't break a whole run.
func summarizeChunk(ctx context.Context, cc *chat.Client, rel string, c chunk.Chunk) (string, error) {
	const system = "You are a code summarizer. Given a single function, method, or class, " +
		"write 1–2 sentences describing what it does. " +
		"Lead with the identifier name. " +
		"State its purpose, key parameters, and return value or notable side effects. " +
		"Use present tense. No prose padding, no restating the prompt, no code blocks."
	user := fmt.Sprintf("FILE: %s (lines %d–%d, kind: %s)\n\n```\n%s\n```",
		rel, c.StartLine, c.EndLine, c.Kind, c.Content)
	resp, err := cc.Generate(ctx, []chat.Message{
		{Role: "system", Content: system},
		{Role: "user", Content: user},
	}, chat.Options{MaxTokens: 150})
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(resp.Content), nil
}

// summarizeFile asks the chat endpoint for a tight, retrieval-friendly
// summary of one file. Returns the summary text or an error if the
// chat call fails. Caller decides whether the failure is fatal — the
// indexer logs and skips so one bad file doesn't break a whole run.
func summarizeFile(ctx context.Context, cc *chat.Client, rel string, data []byte) (string, error) {
	const system = "You are a code summarizer. Summarize this single file in 2-4 sentences so a reader can decide whether to open it. " +
		"Lead with what the file does. Name the exported types and functions verbatim. Note any non-obvious side effects or invariants. " +
		"No prose padding, no apologies, no restating the prompt."
	user := fmt.Sprintf("FILE: %s\n\n```\n%s\n```", rel, data)
	resp, err := cc.Generate(ctx, []chat.Message{
		{Role: "system", Content: system},
		{Role: "user", Content: user},
	}, chat.Options{MaxTokens: 300})
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

// lineCount returns the number of lines in data, counting a trailing
// newline as the terminator of the previous line rather than the start
// of an empty one (so file_summary EndLine matches what an editor
// would show for a typical POSIX file).
func lineCount(data []byte) int {
	if len(data) == 0 {
		return 0
	}
	n := 1
	for _, b := range data {
		if b == '\n' {
			n++
		}
	}
	if data[len(data)-1] == '\n' {
		n--
	}
	return n
}
