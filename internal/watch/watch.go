// Package watch keeps a project's index fresh by watching the filesystem
// and re-running the indexer on a debounced burst of save events.
//
// The implementation deliberately punts on per-file incremental updates:
// after a quiet window (Debounce, default 500 ms), it runs a full
// `index.Indexer.Run` pass against the project. The indexer is already
// content-hash incremental, so unchanged files don't re-embed — the only
// per-save overhead is one filesystem walk, which is cheap for the
// project sizes mcsearch is built for.
package watch

import (
	"context"
	"errors"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"

	"github.com/alehatsman/mcsearch/internal/ignore"
	"github.com/alehatsman/mcsearch/internal/index"
)

type Options struct {
	Debounce time.Duration // quiet window before re-indexing; default 500ms
	Verbose  bool
}

type Watcher struct {
	indexer *index.Indexer
	ig      *ignore.Matcher
	root    string
	opts    Options
	ctx     context.Context // set by Run, used by flush

	mu       sync.Mutex
	dirty    bool   // events have arrived since the last successful flush
	running  bool   // a flush goroutine is currently running
	timer    *time.Timer
}

func New(idx *index.Indexer, ig *ignore.Matcher, root string, opt Options) *Watcher {
	if opt.Debounce <= 0 {
		opt.Debounce = 500 * time.Millisecond
	}
	return &Watcher{indexer: idx, ig: ig, root: root, opts: opt}
}

// Run starts the watch loop and blocks until ctx is cancelled or an
// unrecoverable error occurs.
func (w *Watcher) Run(ctx context.Context) error {
	w.ctx = ctx
	fw, err := fsnotify.NewWatcher()
	if err != nil {
		return err
	}
	defer fw.Close()

	if err := w.addWatches(fw, w.root); err != nil {
		return err
	}
	if w.opts.Verbose {
		log.Printf("watch: ready on %s (debounce=%s)", w.root, w.opts.Debounce)
	}

	// Initial re-index (covers anything that changed while the daemon was
	// stopped).
	w.markDirty()

	for {
		select {
		case <-ctx.Done():
			return nil
		case ev, ok := <-fw.Events:
			if !ok {
				return errors.New("fsnotify events channel closed")
			}
			w.handle(fw, ev)
		case err, ok := <-fw.Errors:
			if !ok {
				return errors.New("fsnotify errors channel closed")
			}
			log.Printf("watch: fsnotify error: %v", err)
		}
	}
}

func (w *Watcher) handle(fw *fsnotify.Watcher, ev fsnotify.Event) {
	rel, err := filepath.Rel(w.root, ev.Name)
	if err != nil {
		return
	}
	// Skip events on ignored paths.
	info, statErr := os.Stat(ev.Name)
	isDir := statErr == nil && info.IsDir()
	if w.ig.Match(rel, isDir) {
		return
	}
	// New directory → add a watch to it (recursively). fsnotify is
	// non-recursive; we maintain coverage by walking each new subtree.
	if ev.Has(fsnotify.Create) && isDir {
		if err := w.addWatches(fw, ev.Name); err != nil && w.opts.Verbose {
			log.Printf("watch: addWatches(%s): %v", ev.Name, err)
		}
	}
	// File-level events that affect indexed content.
	if !ev.Has(fsnotify.Create) && !ev.Has(fsnotify.Write) && !ev.Has(fsnotify.Remove) && !ev.Has(fsnotify.Rename) {
		return
	}
	if !isDir && !ignore.IndexableExt(ev.Name) && !ev.Has(fsnotify.Remove) && !ev.Has(fsnotify.Rename) {
		return
	}
	if w.opts.Verbose {
		log.Printf("watch: %s %s", ev.Op, rel)
	}
	w.markDirty()
}

// markDirty resets the debounce timer; on expiry it runs an index pass.
func (w *Watcher) markDirty() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.dirty = true
	if w.timer != nil {
		w.timer.Stop()
	}
	w.timer = time.AfterFunc(w.opts.Debounce, w.flush)
}

func (w *Watcher) flush() {
	w.mu.Lock()
	if w.running {
		// Another goroutine is already re-indexing. Leave dirty=true so
		// that the in-flight flush re-runs after it finishes. Without
		// this, two flushes would race on the SQLite writer lock.
		w.mu.Unlock()
		return
	}
	if !w.dirty {
		w.mu.Unlock()
		return
	}
	w.dirty = false
	w.running = true
	w.mu.Unlock()

	for {
		start := time.Now()
		err := w.indexer.Run(w.ctx)
		if err != nil {
			if errors.Is(err, context.Canceled) {
				// Shutdown initiated. Stop quietly.
			} else {
				log.Printf("watch: re-index failed: %v", err)
			}
		} else if w.opts.Verbose {
			log.Printf("watch: re-indexed in %s", time.Since(start).Round(time.Millisecond))
		}

		// If events landed during the run, re-flush in the same goroutine
		// rather than spawning another one. This serializes work and
		// guarantees no concurrent indexers.
		w.mu.Lock()
		if !w.dirty || w.ctx.Err() != nil {
			w.running = false
			w.mu.Unlock()
			return
		}
		w.dirty = false
		w.mu.Unlock()
	}
}

// addWatches registers fw on dir and all of its non-ignored subdirectories.
func (w *Watcher) addWatches(fw *fsnotify.Watcher, dir string) error {
	return filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if !d.IsDir() {
			return nil
		}
		rel, _ := filepath.Rel(w.root, path)
		if rel == "." {
			return fw.Add(path)
		}
		if w.ig.Match(rel, true) {
			return filepath.SkipDir
		}
		if err := fw.Add(path); err != nil {
			if w.opts.Verbose {
				log.Printf("watch: fw.Add(%s): %v", path, err)
			}
		}
		return nil
	})
}
