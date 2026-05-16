// Package index orchestrates the walk → chunk → embed → upsert pipeline.
package index

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/alehatsman/mcsearch/internal/chunk"
	"github.com/alehatsman/mcsearch/internal/embed"
	"github.com/alehatsman/mcsearch/internal/ignore"
	"github.com/alehatsman/mcsearch/internal/proj"
	"github.com/alehatsman/mcsearch/internal/store"
)

// Options controls one index run.
type Options struct {
	MaxFileSize int64 // skip files larger than this (bytes); 0 = 1 MB default
	Verbose     bool
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
				log.Printf("walk: %s: %v", path, err)
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
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			return nil
		}
		if !ignore.IndexableExt(path) {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		if info.Size() > ix.Options.MaxFileSize {
			skipped++
			if ix.Options.Verbose {
				log.Printf("skip %s (too large: %d bytes)", rel, info.Size())
			}
			return nil
		}
		// Mtime fast-path: file hasn't changed since the last successful
		// index → just bump last_seen_at on its existing chunks.
		if !lastIndexed.IsZero() && !info.ModTime().After(lastIndexed) {
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
		if ignore.LooksLikeSecret(data) {
			log.Printf("skip %s (matches secret pattern)", rel)
			skipped++
			return nil
		}

		chunks, err := chunk.Chunks(ctx, rel, data)
		if err != nil {
			if ix.Options.Verbose {
				log.Printf("chunk %s: %v", rel, err)
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
			log.Printf("embedding %d new/changed chunks", len(toEmbed))
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
			for i, p := range batch {
				if err := ix.Store.Upsert(ctx,
					p.rel, p.chunk.Kind, p.chunk.StartLine, p.chunk.EndLine,
					p.sha, p.chunk.Content, vecs[i], startTime); err != nil {
					return err
				}
			}
		}
	}

	pruned, err := ix.Store.PruneUnseen(ctx, startTime)
	if err != nil {
		return err
	}
	if ix.Options.Verbose && pruned > 0 {
		log.Printf("pruned %d stale chunks (files removed since last index)", pruned)
	}
	if err := ix.Store.SetLastIndexedAt(ctx, startTime); err != nil {
		return err
	}
	if ix.Options.Verbose {
		log.Printf("indexed: %d chunks seen (%d files fast-path), %d new/changed embedded, %d pruned, %d files skipped",
			seen, mtimeSkips, len(toEmbed), pruned, skipped)
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
