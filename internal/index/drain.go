package index

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/alehatsman/dex/internal/chunk"
	"github.com/alehatsman/dex/internal/store"
	"golang.org/x/sync/errgroup"
)

// DrainPendingSummariesBatch processes up to `max` rows from
// pending_summaries (file_summary + chunk_summary kinds). Pass 0 for
// "no limit" — drain everything currently queued.
//
// Returns (generated, remaining, err):
//   - generated: summaries upserted this call (excludes stale-drops
//     and cache hits).
//   - remaining: queue depth observed AFTER this batch. Caller can
//     loop while remaining > 0 to bound per-call latency.
//
// Does NOT cascade. Callers that want package_summary / repo_summary
// refreshed must invoke CascadePackageRepoSummaries separately —
// typically once the queue reaches remaining == 0.
//
// Cancellation is cooperative: per-row chat calls honour ctx, and
// rows already upserted + deleted from the queue stay committed even
// if ctx ends mid-batch. This makes the batch a safe unit of work for
// a watcher's idle hook to schedule and preempt.
func (ix *Indexer) DrainPendingSummariesBatch(ctx context.Context, max int) (generated, remaining int, err error) {
	if ix.Options.Chat == nil {
		return 0, 0, fmt.Errorf("DrainPendingSummariesBatch: chat client not configured")
	}
	// Same embed-model gate as Run: the drainer also embeds and upserts.
	if err := ix.Store.EnsureEmbedModel(ctx, ix.Embed.ModelName()); err != nil {
		return 0, 0, err
	}
	startTime := time.Now()

	pending, err := ix.Store.ListPendingSummaries(ctx, max)
	if err != nil {
		return 0, 0, fmt.Errorf("list pending: %w", err)
	}
	if len(pending) == 0 {
		return 0, 0, nil
	}
	ix.Options.Logger.Info("drain: batch starting", "pending", len(pending), "max", max)

	conc := ix.Options.SummaryConcurrency
	if conc < 1 {
		conc = 1
	}
	results := make([]*drainResult, len(pending))
	stale := make([]int64, 0)
	var staleMu sync.Mutex
	addStale := func(id int64) {
		staleMu.Lock()
		stale = append(stale, id)
		staleMu.Unlock()
	}

	eg, egctx := errgroup.WithContext(ctx)
	eg.SetLimit(conc)
	for i := range pending {
		p := pending[i]
		eg.Go(func() error {
			switch p.Kind {
			case chunk.KindFileSummary:
				res, stalehit, err := ix.processFileSummary(egctx, p)
				if err != nil {
					ix.Options.Logger.Warn("file summary drain failed", "id", p.ID, "path", p.Path, "err", err)
					_ = ix.Store.BumpPendingAttempts(ctx, p.ID, err.Error())
					return nil
				}
				if stalehit {
					addStale(p.ID)
					return nil
				}
				if res != nil {
					results[i] = res
				}
			case chunk.KindChunkSummary:
				res, stalehit, err := ix.processChunkSummary(egctx, p)
				if err != nil {
					ix.Options.Logger.Warn("chunk summary drain failed", "id", p.ID, "path", p.Path, "start", p.StartLine, "err", err)
					_ = ix.Store.BumpPendingAttempts(ctx, p.ID, err.Error())
					return nil
				}
				if stalehit {
					addStale(p.ID)
					return nil
				}
				if res != nil {
					results[i] = res
				}
			default:
				ix.Options.Logger.Warn("unknown pending kind", "id", p.ID, "kind", p.Kind)
				_ = ix.Store.BumpPendingAttempts(ctx, p.ID, "unknown kind")
			}
			return nil
		})
	}
	if err := eg.Wait(); err != nil {
		return 0, 0, err
	}

	// Compact successful results, then embed + upsert in batches.
	successful := make([]*drainResult, 0, len(results))
	for _, r := range results {
		if r != nil {
			successful = append(successful, r)
		}
	}

	if len(successful) > 0 {
		batchSize := ix.Embed.Batch
		if batchSize <= 0 {
			batchSize = 32
		}
		totalBatches := (len(successful) + batchSize - 1) / batchSize
		ix.Options.Logger.Info("drain: embedding summaries",
			"chunks", len(successful),
			"batches", totalBatches,
			"batch_size", batchSize)
		for start := 0; start < len(successful); start += batchSize {
			end := start + batchSize
			if end > len(successful) {
				end = len(successful)
			}
			batch := successful[start:end]
			texts := make([]string, len(batch))
			for i, r := range batch {
				texts[i] = r.chunk.EmbedText()
			}
			batchStart := time.Now()
			vecs, embErr := ix.Embed.Embed(ctx, texts)
			if embErr != nil {
				return generated, 0, fmt.Errorf("embed: %w", embErr)
			}
			rows := make([]store.PendingChunk, len(batch))
			for i, r := range batch {
				rows[i] = store.PendingChunk{
					Path:       r.chunk.Path,
					Kind:       r.chunk.Kind,
					Name:       r.chunk.Name,
					StartLine:  r.chunk.StartLine,
					EndLine:    r.chunk.EndLine,
					ContentSHA: r.sha,
					Content:    r.chunk.Content,
					Vec:        vecs[i],
				}
			}
			if upErr := ix.Store.UpsertMany(ctx, rows, startTime); upErr != nil {
				return generated, 0, fmt.Errorf("upsert: %w", upErr)
			}
			// Only delete pending rows after the upsert succeeds.
			for _, r := range batch {
				if delErr := ix.Store.DeletePendingSummary(ctx, r.pendingID); delErr != nil {
					return generated, 0, fmt.Errorf("delete pending: %w", delErr)
				}
			}
			generated += len(batch)
			ix.Options.Logger.Info("drain: embed batch",
				"batch", start/batchSize+1,
				"of", totalBatches,
				"chunks", len(batch),
				"took", time.Since(batchStart).Round(time.Millisecond))
		}
	}

	// Drop stale rows (source content changed since enqueue). The next
	// index --summarize-defer run will re-enqueue with the new SHA.
	for _, id := range stale {
		if err := ix.Store.DeletePendingSummary(ctx, id); err != nil {
			return generated, 0, fmt.Errorf("delete stale pending: %w", err)
		}
	}
	if len(stale) > 0 {
		ix.Options.Logger.Info("drain: dropped stale rows", "count", len(stale))
	}

	remaining, _ = ix.Store.CountPendingSummaries(ctx)
	if generated > 0 {
		_ = ix.Store.SetLastSummarizedAt(ctx, time.Now())
	}
	ix.Options.Logger.Info("drain: batch done",
		"generated", generated,
		"stale_dropped", len(stale),
		"remaining", remaining,
		"elapsed", time.Since(startTime).Round(time.Millisecond))
	return generated, remaining, nil
}

// CascadePackageRepoSummaries regenerates any missing package_summary
// and repo_summary chunks from the current file_summary state of the
// chunks table. No-op when no file_summary chunks exist yet.
//
// Exposed so external callers (e.g. the watcher's idle hook) can run
// the cascade independently of the per-batch drainer — typically once
// DrainPendingSummariesBatch reports remaining == 0.
func (ix *Indexer) CascadePackageRepoSummaries(ctx context.Context) (int, error) {
	if ix.Options.Chat == nil {
		return 0, fmt.Errorf("CascadePackageRepoSummaries: chat client not configured")
	}
	if err := ix.Store.EnsureEmbedModel(ctx, ix.Embed.ModelName()); err != nil {
		return 0, err
	}
	gen, err := ix.cascadePackageAndRepo(ctx, time.Now())
	if err == nil && gen > 0 {
		_ = ix.Store.SetLastSummarizedAt(ctx, time.Now())
	}
	return gen, err
}

// IdleSummaryDrainer returns a callback suitable for
// watch.Options.OnIdle: it drains pending_summaries in batches of
// batchSize and cascades package + repo summaries once the queue is
// empty. Returns nil when the Indexer has no chat client configured
// (caller must fall through with OnIdle=nil).
//
// Stop conditions encoded in the callback:
//   - queue empty → cascade then signal done=true.
//   - batch made no progress (chat endpoint dead → all rows fail) →
//     done=true so we don't busy-loop; the next flush re-arms.
//   - underlying batch errors → (true, err); the watcher logs and
//     stops the cycle.
//
// Shared by `dex watch` and the MCP auto-watcher.
func (ix *Indexer) IdleSummaryDrainer(batchSize int) func(context.Context) (bool, error) {
	if ix.Options.Chat == nil {
		return nil
	}
	if batchSize <= 0 {
		batchSize = 10
	}
	logger := ix.Options.Logger
	verbose := ix.Options.Verbose
	return func(ctx context.Context) (bool, error) {
		before, _ := ix.Store.CountPendingSummaries(ctx)
		gen, after, err := ix.DrainPendingSummariesBatch(ctx, batchSize)
		if err != nil {
			return true, err
		}
		if after == 0 {
			cascadeGen, err := ix.CascadePackageRepoSummaries(ctx)
			if err != nil {
				return true, err
			}
			if verbose && (gen > 0 || cascadeGen > 0) {
				logger.Info("idle drain: complete", "summaries", gen, "cascade", cascadeGen)
			}
			return true, nil
		}
		if after >= before {
			logger.Warn("idle drain: no progress, stopping cycle", "remaining", after)
			return true, nil
		}
		if verbose {
			logger.Info("idle drain: batch", "generated", gen, "remaining", after)
		}
		return false, nil
	}
}

// DrainPendingSummaries drains the entire queue then cascades. This is
// the all-in-one entry point used by `dex index summarize`; callers
// that need to yield between rows (a watcher's idle hook, for
// example) should compose DrainPendingSummariesBatch with
// CascadePackageRepoSummaries instead.
func (ix *Indexer) DrainPendingSummaries(ctx context.Context) (int, error) {
	if ix.Options.Chat == nil {
		return 0, fmt.Errorf("DrainPendingSummaries: chat client not configured")
	}
	total := 0
	for {
		gen, remaining, err := ix.DrainPendingSummariesBatch(ctx, 0)
		if err != nil {
			return total, err
		}
		total += gen
		if remaining == 0 {
			break
		}
		// max=0 drains everything in one call, so the loop normally exits
		// on the first iteration. Kept as a safety net in case future
		// row-filtering (e.g. attempts-based backoff) causes the same
		// batch to leave rows behind.
	}
	ix.Options.Logger.Info("drain: cascading package + repo summaries")
	cascadeGen, err := ix.CascadePackageRepoSummaries(ctx)
	if err != nil {
		return total, err
	}
	return total + cascadeGen, nil
}

// processFileSummary handles one pending file_summary row. Returns
// (result, stale, err). `stale=true` means the file's current content
// no longer matches what was queued — the drainer should drop the row.
func (ix *Indexer) processFileSummary(ctx context.Context, p store.PendingSummary) (*drainResult, bool, error) {
	fullPath := filepath.Join(ix.Proj.Root, p.Path)
	data, err := os.ReadFile(fullPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, true, nil // file was removed; drop pending
		}
		return nil, false, fmt.Errorf("read %s: %w", fullPath, err)
	}
	slice := data
	if len(slice) > summarizeCap {
		slice = slice[:summarizeCap]
	}
	currentSHA := chunkSHA(string(slice))
	if currentSHA != p.ContentSHA {
		return nil, true, nil // file changed; drop pending
	}
	summary, err := summarizeFile(ctx, ix.Options.Chat, ix.Options.SummaryModels.File, p.Path, slice)
	if err != nil {
		return nil, false, err
	}
	if strings.TrimSpace(summary) == "" {
		return nil, true, nil // empty response; drop rather than retry
	}
	endLine := p.EndLine
	if endLine <= 0 {
		endLine = chunk.LineCount(data)
	}
	return &drainResult{
		pendingID: p.ID,
		chunk: chunk.Chunk{
			Path:      p.Path,
			Kind:      chunk.KindFileSummary,
			StartLine: 1,
			EndLine:   endLine,
			Content:   summary,
		},
		sha: p.ContentSHA,
	}, false, nil
}

// processChunkSummary handles one pending chunk_summary row. The
// source chunk's content is looked up by (path, source_sha1) — if it's
// no longer in the chunks table, the source has been pruned (file
// changed or removed), so we drop the row as stale.
func (ix *Indexer) processChunkSummary(ctx context.Context, p store.PendingSummary) (*drainResult, bool, error) {
	content, err := ix.Store.ChunkContent(ctx, p.Path, p.SourceSHA)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, true, nil // source chunk gone; drop pending
		}
		return nil, false, err
	}
	// Recompute the expected sumSHA from the recovered content; if it
	// doesn't match what was queued, something is inconsistent and we
	// drop the row rather than upsert under the wrong identity.
	expectedSumSHA := chunkSHA(chunk.KindChunkSummary + ":" + content)
	if expectedSumSHA != p.ContentSHA {
		return nil, true, nil
	}
	sourceChunk := chunk.Chunk{
		Path:      p.Path,
		Kind:      p.ChunkKind,
		Name:      p.ChunkName,
		StartLine: p.StartLine,
		EndLine:   p.EndLine,
		Content:   content,
	}
	summary, err := summarizeChunk(ctx, ix.Options.Chat, ix.Options.SummaryModels.Chunk, p.Path, sourceChunk)
	if err != nil {
		return nil, false, err
	}
	if strings.TrimSpace(summary) == "" {
		return nil, true, nil
	}
	return &drainResult{
		pendingID: p.ID,
		chunk: chunk.Chunk{
			Path:      p.Path,
			Kind:      chunk.KindChunkSummary,
			StartLine: p.StartLine,
			EndLine:   p.EndLine,
			Content:   summary,
		},
		sha: p.ContentSHA,
	}, false, nil
}

// drainResult mirrors the anonymous struct used by DrainPendingSummaries.
// Lifted to package scope so the processX helpers can return it.
type drainResult struct {
	pendingID int64
	chunk     chunk.Chunk
	sha       string
}

// cascadePackageAndRepo regenerates any missing package_summary and
// repo_summary chunks based on the current file_summary / package_summary
// state of the chunks table. Mirrors Run()'s Pass 5 and Pass 6, but
// reads its inputs from the store instead of from in-flight pkgFiles.
func (ix *Indexer) cascadePackageAndRepo(ctx context.Context, startTime time.Time) (int, error) {
	if ctx.Err() != nil {
		return 0, ctx.Err()
	}
	allSHAs, err := ix.Store.FileSummarySHAs(ctx)
	if err != nil {
		return 0, fmt.Errorf("file summary SHAs: %w", err)
	}
	if len(allSHAs) == 0 {
		return 0, nil
	}

	type pkgFileEntry struct {
		path string
		sha  string
	}
	pkgFiles := make(map[string][]pkgFileEntry)
	for path, sha := range allSHAs {
		dir := filepath.Dir(path)
		pkgFiles[dir] = append(pkgFiles[dir], pkgFileEntry{path, sha})
	}

	dirs := make([]string, 0, len(pkgFiles)+1)
	for d := range pkgFiles {
		dirs = append(dirs, d)
	}
	dirs = append(dirs, ".")
	existingBatch, err := ix.Store.ExistingSHAsBatch(ctx, dirs)
	if err != nil {
		return 0, fmt.Errorf("existing SHAs: %w", err)
	}

	type pkgJob struct {
		dir       string
		filePaths []string
		pkgSHA    string
	}
	var jobs []pkgJob
	for dir, entries := range pkgFiles {
		if ctx.Err() != nil {
			break
		}
		shas := make([]string, len(entries))
		filePaths := make([]string, len(entries))
		for i, e := range entries {
			shas[i] = e.sha
			filePaths[i] = e.path
		}
		sort.Strings(shas)
		pkgSHA := chunkSHA(strings.Join(shas, ":"))
		if existingBatch[dir][pkgSHA] {
			if err := ix.Store.TouchSeen(ctx, dir, pkgSHA, "", 0, 0, startTime); err != nil {
				return 0, err
			}
			continue
		}
		jobs = append(jobs, pkgJob{dir: dir, filePaths: filePaths, pkgSHA: pkgSHA})
	}

	generated := 0
	if len(jobs) > 0 {
		conc := ix.Options.SummaryConcurrency
		if conc < 1 {
			conc = 1
		}
		results := make([]*pending, len(jobs))
		eg, egctx := errgroup.WithContext(ctx)
		eg.SetLimit(conc)
		for i := range jobs {
			j := jobs[i]
			eg.Go(func() error {
				fileSummaries, err := ix.Store.FileSummariesForPaths(egctx, j.filePaths)
				if err != nil || len(fileSummaries) == 0 {
					return nil
				}
				summary, err := summarizePackage(egctx, ix.Options.Chat, ix.Options.SummaryModels.Package, j.dir, fileSummaries)
				if err != nil {
					ix.Options.Logger.Warn("package summarize failed", "dir", j.dir, "err", err)
					return nil
				}
				if strings.TrimSpace(summary) == "" {
					return nil
				}
				results[i] = &pending{
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
			return generated, err
		}
		var pkgEmbed []pending
		for _, p := range results {
			if p != nil {
				pkgEmbed = append(pkgEmbed, *p)
			}
		}
		if len(pkgEmbed) > 0 {
			texts := make([]string, len(pkgEmbed))
			for i, p := range pkgEmbed {
				texts[i] = p.chunk.EmbedText()
			}
			vecs, err := ix.Embed.Embed(ctx, texts)
			if err != nil {
				return generated, fmt.Errorf("package embed: %w", err)
			}
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
				return generated, err
			}
			generated += len(pkgEmbed)
		}
	}

	// Repo summary: one per project, derived from current package_summary
	// chunks. Stored under path=".".
	if ctx.Err() != nil {
		return generated, nil
	}
	pkgSummaries, err := ix.Store.AllSummariesByKind(ctx, chunk.KindPackageSummary)
	if err != nil || len(pkgSummaries) == 0 {
		return generated, nil
	}
	repoSHA := chunkSHA(strings.Join(pkgSummaries, "\x00"))
	if existingBatch["."][repoSHA] {
		if err := ix.Store.TouchSeen(ctx, ".", repoSHA, "", 0, 0, startTime); err != nil {
			return generated, err
		}
		return generated, nil
	}
	summary, err := summarizeRepo(ctx, ix.Options.Chat, ix.Options.SummaryModels.Repo, pkgSummaries)
	if err != nil {
		ix.Options.Logger.Warn("repo summarize failed", "err", err)
		return generated, nil
	}
	if strings.TrimSpace(summary) == "" {
		return generated, nil
	}
	vecs, err := ix.Embed.Embed(ctx, []string{chunk.KindRepoSummary + "\n" + summary})
	if err != nil {
		ix.Options.Logger.Warn("repo summary embed failed", "err", err)
		return generated, nil
	}
	rows := []store.PendingChunk{{
		Path:       ".",
		Kind:       chunk.KindRepoSummary,
		ContentSHA: repoSHA,
		Content:    summary,
		Vec:        vecs[0],
	}}
	if err := ix.Store.UpsertMany(ctx, rows, startTime); err != nil {
		return generated, err
	}
	generated++
	return generated, nil
}
