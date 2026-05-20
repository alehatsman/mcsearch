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
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/alehatsman/mcsearch/internal/chat"
	"github.com/alehatsman/mcsearch/internal/chunk"
	"github.com/alehatsman/mcsearch/internal/embed"
	"github.com/alehatsman/mcsearch/internal/ignore"
	"github.com/alehatsman/mcsearch/internal/proj"
	"github.com/alehatsman/mcsearch/internal/store"
	"golang.org/x/sync/errgroup"
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
	// SummaryConcurrency caps in-flight chat calls for per-chunk
	// summaries within a single file. <=1 = sequential (preserves
	// existing behaviour). Set higher to overlap inference with HTTP
	// RTT on a local Ollama/vLLM that can serve concurrent requests.
	SummaryConcurrency int
	// DeferSummaries, when true (alongside Summarize), changes Pass 3
	// from "run chat inline" to "enqueue a pending_summaries row and
	// return immediately." The index call no longer blocks on summary
	// generation; a separate `mcsearch index summarize` drainer (or watch
	// idle ticks) picks the queue up later. Package and repo summaries
	// are skipped entirely in this mode — they have cascading data
	// dependencies on file_summary chunks that don't exist yet at
	// queue time, so the drainer generates them after file/chunk jobs
	// drain. Chat is not required when DeferSummaries is true.
	DeferSummaries bool
	// ChunkSummaryMinLines overrides the package default
	// (chunkSummaryMinLines). 0 = use default. Raise to cut chunk-summary
	// volume on large repos by skipping medium-sized functions too.
	ChunkSummaryMinLines int
	// Concurrency caps the number of parallel file readers / chunkers in
	// Pass 1. <=0 = runtime.GOMAXPROCS(0). The walker itself stays
	// single-threaded (directory IO is cheap and serializes well with
	// inline mtime fast-path UPDATEs); only the expensive per-file work —
	// ReadFile, binary/secret detection, tree-sitter parse — runs on
	// workers.
	Concurrency int
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
// slowFile holds one file that survived all walk-phase filters and needs
// SHA-based deduplication against the store.
type slowFile struct {
	rel    string
	data   []byte
	chunks []chunk.Chunk
}

func (ix *Indexer) Run(ctx context.Context) error {
	startTime := time.Now()
	var (
		toEmbed            []pending
		seen               int
		summariesGenerated int
		summariesQueued    int
	)

	// summarizeWanted gates every plan-phase code path that decides what
	// summary chunks are missing. It's true when either:
	//   - inline mode: the user wants summaries AND we have a chat client
	//     to call now, OR
	//   - defer mode: the user wants summaries but is happy to queue them
	//     for a later drainer (no chat client needed at index time).
	summarizeWanted := ix.Options.Summarize && (ix.Options.Chat != nil || ix.Options.DeferSummaries)

	prevStats, statsErr := ix.Store.Stats(ctx)
	var lastIndexed time.Time
	if statsErr == nil {
		lastIndexed = prevStats.LastIndex
	}

	// Pre-walk: when --summarize is on, fetch every existing file_summary
	// row's content_sha1 in one query. The walker uses this to decide
	// whether each unchanged file is fully covered by the previous
	// summarize run (file_summary present → fast-path eligible) and to
	// recover the per-file SHA that Pass 5 needs as input to its
	// package_summary cache key. Without the pre-fetch we'd either
	// disable the fast-path under --summarize (the historical behaviour,
	// wasteful) or do N round-trips during the walk.
	var existingFileSummarySHAs map[string]string
	if summarizeWanted {
		shas, err := ix.Store.FileSummarySHAs(ctx)
		if err != nil {
			return fmt.Errorf("prefetch file_summary SHAs: %w", err)
		}
		existingFileSummarySHAs = shas
	}

	// pkgFiles tracks per-directory file entries for package summary
	// generation in Pass 5. Only populated when Summarize is on.
	// Initialized before the walk so the fast-path branch can populate
	// entries for files that bypass slowFiles. Mutated only by the
	// walker goroutine (the workers don't touch it), so no lock needed.
	type pkgFileEntry struct {
		path    string
		fileSHA string
	}
	var pkgFiles map[string][]pkgFileEntry
	if summarizeWanted {
		pkgFiles = make(map[string][]pkgFileEntry, len(existingFileSummarySHAs))
	}
	// fastDirs tracks dirs that contributed only via the summarize
	// fast-path (no slow-path siblings). Pass 2's batch-SHA query needs
	// to include them so Pass 5 can cache-hit their package_summary
	// without a needless chat regeneration. Walker-only mutation.
	fastDirs := make(map[string]bool)

	// Pass 1: walk the tree.
	//
	// The walker (this goroutine) does only cheap work — directory
	// traversal, ignore checks, symlink/extension/size filters, and the
	// inline mtime fast-path (one SQL UPDATE per unchanged file).
	// Keeping the fast-path inline avoids contending workers on the
	// SQLite writer lock for what is already a serial bottleneck.
	//
	// Expensive per-file work — ReadFile, LooksBinary/LooksLikeSecret,
	// tree-sitter Chunks — is fanned out to a pool of workers sized by
	// Options.Concurrency (default GOMAXPROCS). Each worker holds its
	// own tree-sitter parser via chunk.Chunks() so they don't share
	// state. Order of slowFiles is non-deterministic; subsequent passes
	// don't depend on it.
	conc := ix.Options.Concurrency
	if conc <= 0 {
		conc = runtime.GOMAXPROCS(0)
	}
	type pathTask struct {
		rel  string
		path string
	}
	pathCh := make(chan pathTask, conc*4)
	resultCh := make(chan slowFile, conc*4)
	var (
		skipped    atomic.Int64
		mtimeSkips atomic.Int64
	)

	var workers sync.WaitGroup
	for i := 0; i < conc; i++ {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for task := range pathCh {
				data, err := os.ReadFile(task.path)
				if err != nil {
					continue
				}
				if ignore.LooksBinary(data) {
					skipped.Add(1)
					continue
				}
				// Skip files whose content matches a known secret
				// pattern — but allow-list test fixtures, which
				// routinely embed fake credentials as inputs to
				// their own detection logic.
				if !ignore.IsTestPath(task.rel) && ignore.LooksLikeSecret(data) {
					ix.Options.Logger.Warn("skip (matches secret pattern)", "path", task.rel)
					skipped.Add(1)
					continue
				}
				chunks, err := chunk.Chunks(ctx, task.rel, data)
				if err != nil {
					if ix.Options.Verbose {
						ix.Options.Logger.Info("chunk error", "path", task.rel, "err", err)
					}
					continue
				}
				select {
				case resultCh <- slowFile{rel: task.rel, data: data, chunks: chunks}:
				case <-ctx.Done():
					return
				}
			}
		}()
	}

	// Collector drains resultCh into slowFiles. Runs on its own
	// goroutine so the walker can keep pushing into pathCh without
	// deadlocking against a full resultCh.
	var slowFiles []slowFile
	var collector sync.WaitGroup
	collector.Add(1)
	go func() {
		defer collector.Done()
		for sf := range resultCh {
			slowFiles = append(slowFiles, sf)
		}
	}()

	walkErr := filepath.WalkDir(ix.Proj.Root, func(path string, d fs.DirEntry, err error) error {
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
			skipped.Add(1)
			if ix.Options.Verbose {
				ix.Options.Logger.Info("skip (too large)", "path", rel, "size", info.Size())
			}
			return nil
		}
		// Mtime fast-path: file hasn't changed since the last successful
		// index → just bump last_seen_at on its existing chunks.
		// Under --summarize the fast-path requires that the previous run
		// produced a file_summary row for this path. If so, the same
		// TouchPath bumps the file_summary, chunk_summary and code chunks
		// in one shot (they all share `path = rel`), and we recover the
		// stored file_summary SHA so Pass 5's package_summary cache key
		// still resolves cleanly. If not, fall through to the slow path
		// so the missing summaries get generated.
		//
		// Caveat: this trusts that whatever chunk_summary set the previous
		// run produced is still complete. Lowering MCSEARCH_CHUNK_SUMMARY_MIN_LINES
		// between runs leaves previously-too-short chunks without summaries
		// — operators changing the threshold should `mcsearch reindex` once.
		fastPathSummary := ""
		canFastPath := !ix.Options.Summarize
		if !canFastPath {
			if sha, ok := existingFileSummarySHAs[rel]; ok {
				canFastPath = true
				fastPathSummary = sha
			}
		}
		if canFastPath && !lastIndexed.IsZero() && !info.ModTime().After(lastIndexed) {
			rows, terr := ix.Store.TouchPath(ctx, rel, startTime)
			if terr == nil && rows > 0 {
				seen += int(rows)
				mtimeSkips.Add(1)
				if ix.Options.Summarize && fastPathSummary != "" {
					dir := filepath.Dir(rel)
					pkgFiles[dir] = append(pkgFiles[dir], pkgFileEntry{path: rel, fileSHA: fastPathSummary})
					fastDirs[dir] = true
				}
				return nil
			}
			// rows==0: never indexed before, fall through to slow path.
		}
		select {
		case pathCh <- pathTask{rel: rel, path: path}:
		case <-ctx.Done():
			return ctx.Err()
		}
		return nil
	})
	close(pathCh)
	workers.Wait()
	close(resultCh)
	collector.Wait()
	if walkErr != nil {
		return fmt.Errorf("walk: %w", walkErr)
	}

	// Pass 2: one batch query for all slow-path files instead of N per-file
	// queries. Also include unique package dirs so we can check package
	// summary cache without extra round-trips in Pass 5. Dirs that hosted
	// only fast-path-summarize files (no slow-path siblings) are included
	// too, so their package_summary cache key still resolves.
	slowPaths := make([]string, len(slowFiles))
	for i, sf := range slowFiles {
		slowPaths[i] = sf.rel
	}
	allPaths := slowPaths
	if summarizeWanted {
		dirSet := make(map[string]bool, len(slowFiles)/4+len(fastDirs))
		for _, sf := range slowFiles {
			if len(sf.chunks) > 0 {
				dirSet[filepath.Dir(sf.rel)] = true
			}
		}
		for d := range fastDirs {
			dirSet[d] = true
		}
		for d := range dirSet {
			allPaths = append(allPaths, d)
		}
		allPaths = append(allPaths, ".") // repo-level summary cache entry
	}
	existingBatch, err := ix.Store.ExistingSHAsBatch(ctx, allPaths)
	if err != nil {
		return fmt.Errorf("existing SHAs: %w", err)
	}

	// Pass 3a: serial plan phase.
	//
	// Walk slowFiles once to resolve the cheap, IO-bound work — chunk SHA
	// lookups, TouchSeen UPDATEs for cache hits, and file_summary cache
	// resolution (so pkgFiles is fully populated before Pass 5). Chat
	// calls are NOT issued here. Instead, each cache-miss summary is
	// queued as a job for the parallel execute phase below.
	//
	// Keeping TouchSeen calls serial avoids piling contention on
	// SQLite's writer lock; only the chat-bound work crosses the
	// concurrency boundary.
	type fileSummaryJob struct {
		rel       string
		slice     []byte
		fileSHA   string
		endLine   int
		truncated bool
		size      int
	}
	type chunkSummaryJob struct {
		rel    string
		c      chunk.Chunk
		sumSHA string
	}
	var (
		fileSummaryJobs  []fileSummaryJob
		chunkSummaryJobs []chunkSummaryJob
	)

	for _, sf := range slowFiles {
		if err := ctx.Err(); err != nil {
			return err
		}
		existing := existingBatch[sf.rel]
		seen += len(sf.chunks)

		for _, c := range sf.chunks {
			sha := chunkSHA(c.Content)
			if existing[sha] {
				if err := ix.Store.TouchSeen(ctx, sf.rel, sha, c.Name, c.StartLine, c.EndLine, startTime); err != nil {
					return err
				}
				continue
			}
			toEmbed = append(toEmbed, pending{rel: sf.rel, chunk: c, sha: sha})
		}

		// Per-file summary, opt-in. SHA is computed on the slice we'd
		// actually send to the model (so a file growing past the cap
		// doesn't re-summarize on every run). If we already have a
		// summary chunk for this exact slice, just bump last_seen_at
		// and skip the chat round-trip; otherwise queue a job.
		if summarizeWanted && len(sf.chunks) > 0 {
			slice := sf.data
			truncated := false
			if len(slice) > summarizeCap {
				slice = slice[:summarizeCap]
				truncated = true
			}
			fileSHA := chunkSHA(string(slice))
			if pkgFiles != nil {
				dir := filepath.Dir(sf.rel)
				pkgFiles[dir] = append(pkgFiles[dir], pkgFileEntry{path: sf.rel, fileSHA: fileSHA})
			}
			if existing[fileSHA] {
				if err := ix.Store.TouchSeen(ctx, sf.rel, fileSHA, "", 0, 0, startTime); err != nil {
					return err
				}
				seen++
			} else {
				fileSummaryJobs = append(fileSummaryJobs, fileSummaryJob{
					rel:       sf.rel,
					slice:     slice,
					fileSHA:   fileSHA,
					endLine:   chunk.LineCount(sf.data),
					truncated: truncated,
					size:      len(sf.data),
				})
			}
		}
		// Per-chunk summary, opt-in. Only structural chunks (functions,
		// methods, classes) with ≥ minLines lines get summaries;
		// tiny helpers, windows, and orphans aren't worth the round-trip.
		// SHA is keyed on the chunk source text so cache invalidation is
		// automatic when the function body changes.
		if summarizeWanted {
			minLines := ix.Options.ChunkSummaryMinLines
			if minLines <= 0 {
				minLines = chunkSummaryMinLines
			}
			for _, c := range sf.chunks {
				if !isStructural(c.Kind) || (c.EndLine-c.StartLine+1) < minLines {
					continue
				}
				sumSHA := chunkSHA(chunk.KindChunkSummary + ":" + c.Content)
				if existing[sumSHA] {
					if err := ix.Store.TouchSeen(ctx, sf.rel, sumSHA, "", 0, 0, startTime); err != nil {
						return err
					}
					seen++
					continue
				}
				chunkSummaryJobs = append(chunkSummaryJobs, chunkSummaryJob{rel: sf.rel, c: c, sumSHA: sumSHA})
			}
		}
		// Old rows for this file whose SHA disappeared get pruned at the
		// end via PruneUnseen — they never had last_seen_at bumped on this
		// run.
	}

	// Pass 3b: execute phase. Two strategies, selected by Options.DeferSummaries:
	//
	// Inline (default): one global errgroup processes every cache-miss
	// summary across every slowFile. Bounded by SummaryConcurrency so a
	// local chat endpoint isn't overwhelmed. Workers are independent: file
	// and chunk summaries draw from the same pool, so a file with no chunk
	// summaries doesn't sit idle while a sibling churns through 50. Chat
	// failures log-and-skip (return nil) so one bad job doesn't poison
	// the whole run; the parent ctx still aborts cleanly on cancellation
	// via egctx.
	//
	// Deferred: no chat calls at all. Each job is INSERTed into
	// pending_summaries (idempotent on (path,kind,content_sha1) so a
	// re-run doesn't multiply rows). Index returns fast; the drainer
	// processes the queue later.
	if len(fileSummaryJobs) > 0 || len(chunkSummaryJobs) > 0 {
		if ix.Options.DeferSummaries {
			for _, j := range fileSummaryJobs {
				if err := ix.Store.EnqueuePendingSummary(ctx, store.PendingSummary{
					Path:       j.rel,
					Kind:       chunk.KindFileSummary,
					ContentSHA: j.fileSHA,
					StartLine:  1,
					EndLine:    j.endLine,
				}, startTime); err != nil {
					return fmt.Errorf("enqueue file summary: %w", err)
				}
				summariesQueued++
			}
			for _, j := range chunkSummaryJobs {
				sourceSHA := chunkSHA(j.c.Content)
				if err := ix.Store.EnqueuePendingSummary(ctx, store.PendingSummary{
					Path:       j.rel,
					Kind:       chunk.KindChunkSummary,
					ContentSHA: j.sumSHA,
					StartLine:  j.c.StartLine,
					EndLine:    j.c.EndLine,
					ChunkKind:  j.c.Kind,
					ChunkName:  j.c.Name,
					SourceSHA:  sourceSHA,
				}, startTime); err != nil {
					return fmt.Errorf("enqueue chunk summary: %w", err)
				}
				summariesQueued++
			}
		} else {
			conc := ix.Options.SummaryConcurrency
			if conc < 1 {
				conc = 1
			}
			fileResults := make([]*pending, len(fileSummaryJobs))
			chunkResults := make([]*pending, len(chunkSummaryJobs))
			eg, egctx := errgroup.WithContext(ctx)
			eg.SetLimit(conc)
			for i := range fileSummaryJobs {
				j := fileSummaryJobs[i]
				eg.Go(func() error {
					summary, err := summarizeFile(egctx, ix.Options.Chat, j.rel, j.slice)
					if err != nil {
						ix.Options.Logger.Warn("summarize failed", "path", j.rel, "err", err)
						return nil
					}
					if strings.TrimSpace(summary) == "" {
						return nil
					}
					if ix.Options.Verbose && j.truncated {
						ix.Options.Logger.Info("summarize truncated", "path", j.rel, "size", j.size)
					}
					fileResults[i] = &pending{
						rel: j.rel,
						chunk: chunk.Chunk{
							Path:      j.rel,
							Kind:      chunk.KindFileSummary,
							StartLine: 1,
							EndLine:   j.endLine,
							Content:   summary,
						},
						sha: j.fileSHA,
					}
					return nil
				})
			}
			for i := range chunkSummaryJobs {
				j := chunkSummaryJobs[i]
				eg.Go(func() error {
					summary, err := summarizeChunk(egctx, ix.Options.Chat, j.rel, j.c)
					if err != nil {
						ix.Options.Logger.Warn("chunk summarize failed", "path", j.rel, "start", j.c.StartLine, "err", err)
						return nil
					}
					if strings.TrimSpace(summary) == "" {
						return nil
					}
					chunkResults[i] = &pending{
						rel: j.rel,
						chunk: chunk.Chunk{
							Path:      j.rel,
							Kind:      chunk.KindChunkSummary,
							StartLine: j.c.StartLine,
							EndLine:   j.c.EndLine,
							Content:   summary,
						},
						sha: j.sumSHA,
					}
					return nil
				})
			}
			if err := eg.Wait(); err != nil {
				return err
			}
			for _, p := range fileResults {
				if p != nil {
					toEmbed = append(toEmbed, *p)
					seen++
					summariesGenerated++
				}
			}
			for _, p := range chunkResults {
				if p != nil {
					toEmbed = append(toEmbed, *p)
					seen++
					summariesGenerated++
				}
			}
		}
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
					Name:       p.chunk.Name,
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

	// Pass 5: package summaries — one per directory, generated from the
	// file summaries stored in the previous passes. Runs after embedding
	// so file_summary rows are already committed and queryable.
	//
	// Plan + execute split mirrors Pass 3: serial walk does cache-hit
	// TouchSeen and SHA computation; parallel errgroup runs the chat
	// calls (and the read of FileSummariesForPaths, which goes through
	// SQLite's connection pool and is safe to call concurrently).
	//
	// In defer mode this pass is skipped entirely — package summaries
	// have a cascading data dependency on file_summary chunks that
	// don't exist yet at queue time (they're sitting in
	// pending_summaries). The drainer regenerates package summaries
	// after the file/chunk jobs drain.
	if len(pkgFiles) > 0 && !ix.Options.DeferSummaries {
		type pkgJob struct {
			dir       string
			filePaths []string
			pkgSHA    string
		}
		var pkgJobs []pkgJob
		for dir, entries := range pkgFiles {
			if ctx.Err() != nil {
				break
			}
			// Stable cache key: SHA of sorted per-file SHAs.
			shas := make([]string, len(entries))
			filePaths := make([]string, len(entries))
			for i, e := range entries {
				shas[i] = e.fileSHA
				filePaths[i] = e.path
			}
			sort.Strings(shas)
			pkgSHA := chunkSHA(strings.Join(shas, ":"))
			if existingBatch[dir][pkgSHA] {
				if err := ix.Store.TouchSeen(ctx, dir, pkgSHA, "", 0, 0, startTime); err != nil {
					return err
				}
				continue
			}
			pkgJobs = append(pkgJobs, pkgJob{dir: dir, filePaths: filePaths, pkgSHA: pkgSHA})
		}

		var pkgEmbed []pending
		if len(pkgJobs) > 0 {
			conc := ix.Options.SummaryConcurrency
			if conc < 1 {
				conc = 1
			}
			pkgResults := make([]*pending, len(pkgJobs))
			eg, egctx := errgroup.WithContext(ctx)
			eg.SetLimit(conc)
			for i := range pkgJobs {
				j := pkgJobs[i]
				eg.Go(func() error {
					fileSummaries, err := ix.Store.FileSummariesForPaths(egctx, j.filePaths)
					if err != nil || len(fileSummaries) == 0 {
						return nil
					}
					summary, err := summarizePackage(egctx, ix.Options.Chat, j.dir, fileSummaries)
					if err != nil {
						ix.Options.Logger.Warn("package summarize failed", "dir", j.dir, "err", err)
						return nil
					}
					if strings.TrimSpace(summary) == "" {
						return nil
					}
					pkgResults[i] = &pending{
						rel: j.dir,
						chunk: chunk.Chunk{
							Path:    j.dir,
							Kind:    chunk.KindPackageSummary,
							Content: summary,
						},
						sha: j.pkgSHA,
					}
					return nil
				})
			}
			if err := eg.Wait(); err != nil {
				return err
			}
			for _, p := range pkgResults {
				if p != nil {
					pkgEmbed = append(pkgEmbed, *p)
					summariesGenerated++
				}
			}
		}
		if len(pkgEmbed) > 0 {
			if ix.Options.Verbose {
				ix.Options.Logger.Info("embedding package summaries", "count", len(pkgEmbed))
			}
			texts := make([]string, len(pkgEmbed))
			for i, p := range pkgEmbed {
				texts[i] = p.chunk.EmbedText()
			}
			vecs, err := ix.Embed.Embed(ctx, texts)
			if err != nil {
				ix.Options.Logger.Warn("package summary embed failed", "err", err)
			} else {
				rows := make([]store.PendingChunk, len(pkgEmbed))
				for i, p := range pkgEmbed {
					rows[i] = store.PendingChunk{
						Path:       p.rel,
						Kind:       p.chunk.Kind,
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
	}

	// Pass 6: repository summary — one per project, generated from all
	// package summaries. Runs after Pass 5 so package_summary rows are
	// committed. Stored with path="." so PruneUnseen leaves it alone.
	//
	// Like Pass 5, skipped in defer mode — depends on package_summary
	// chunks that the drainer will produce.
	if pkgFiles != nil && ctx.Err() == nil && !ix.Options.DeferSummaries {
		pkgSummaries, err := ix.Store.AllSummariesByKind(ctx, chunk.KindPackageSummary)
		if err == nil && len(pkgSummaries) > 0 {
			repoSHA := chunkSHA(strings.Join(pkgSummaries, "\x00"))
			if existingBatch["."][repoSHA] {
				if err := ix.Store.TouchSeen(ctx, ".", repoSHA, "", 0, 0, startTime); err != nil {
					return err
				}
			} else {
				summary, err := summarizeRepo(ctx, ix.Options.Chat, pkgSummaries)
				if err != nil {
					ix.Options.Logger.Warn("repo summarize failed", "err", err)
				} else if strings.TrimSpace(summary) != "" {
					vecs, err := ix.Embed.Embed(ctx, []string{chunk.KindRepoSummary + "\n" + summary})
					if err != nil {
						ix.Options.Logger.Warn("repo summary embed failed", "err", err)
					} else {
						rows := []store.PendingChunk{{
							Path:       ".",
							Kind:       chunk.KindRepoSummary,
							ContentSHA: repoSHA,
							Content:    summary,
							Vec:        vecs[0],
						}}
						if err := ix.Store.UpsertMany(ctx, rows, startTime); err != nil {
							return err
						}
					}
				}
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
			"files_fast_path", mtimeSkips.Load(),
			"embedded", len(toEmbed),
			"summaries_generated", summariesGenerated,
			"summaries_queued", summariesQueued,
			"pruned", pruned,
			"skipped", skipped.Load())
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
	case chunk.KindWindow, chunk.KindOrphan, chunk.KindFileSummary, chunk.KindChunkSummary, chunk.KindPackageSummary, chunk.KindRepoSummary:
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

// summarizePackage asks the chat endpoint for a package-level overview built
// from the prose summaries of its constituent files. Returns the summary text
// or an error if the chat call fails. Caller logs and skips on error.
func summarizePackage(ctx context.Context, cc *chat.Client, dir string, fileSummaries []string) (string, error) {
	const system = "You are a code summarizer. Given prose summaries of all files in a code package or directory, " +
		"write 2-4 sentences describing what the package does. " +
		"Name the package path. Describe its public role in the system. " +
		"Mention key exported types and functions. Note dependencies or notable constraints. " +
		"No prose padding, no apologies, no restating the prompt."
	user := fmt.Sprintf("PACKAGE: %s\n\nFILE SUMMARIES:\n%s", dir, strings.Join(fileSummaries, "\n\n---\n\n"))
	resp, err := cc.Generate(ctx, []chat.Message{
		{Role: "system", Content: system},
		{Role: "user", Content: user},
	}, chat.Options{MaxTokens: 300})
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(resp.Content), nil
}

// summarizeRepo asks the chat endpoint for a top-level overview of the
// whole repository, built from the per-package summaries. Stored once per
// project and re-generated only when any package summary changes.
func summarizeRepo(ctx context.Context, cc *chat.Client, pkgSummaries []string) (string, error) {
	const system = "You are a code summarizer. Given prose summaries of every package in a repository, " +
		"write 3-5 sentences describing what the repository does overall. " +
		"Name the top-level packages and their roles. Describe the main data flow or pipeline. " +
		"Note any architectural constraints or key invariants. " +
		"No prose padding, no apologies, no restating the prompt."
	user := fmt.Sprintf("PACKAGE SUMMARIES:\n%s", strings.Join(pkgSummaries, "\n\n---\n\n"))
	resp, err := cc.Generate(ctx, []chat.Message{
		{Role: "system", Content: system},
		{Role: "user", Content: user},
	}, chat.Options{MaxTokens: 400})
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
