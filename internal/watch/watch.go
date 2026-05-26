// Package watch keeps a project's index fresh by watching the filesystem
// and re-running the indexer on a debounced burst of save events.
//
// The implementation deliberately punts on per-file incremental updates:
// after a quiet window (Debounce, default 500 ms), it runs a full
// `index.Indexer.Run` pass against the project. The indexer is already
// content-hash incremental, so unchanged files don't re-embed — the only
// per-save overhead is one filesystem walk, which is cheap for the
// project sizes dex is built for.
package watch

import (
	"context"
	"errors"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/fsnotify/fsnotify"

	"github.com/alehatsman/dex/internal/ignore"
	"github.com/alehatsman/dex/internal/index"
)

type Options struct {
	Debounce time.Duration // quiet window before re-indexing; default 500ms
	Verbose  bool
	Logger   *slog.Logger // destination for log output; nil = io.Discard
	// AfterIndex, if non-nil, is invoked after each successful indexer
	// flush. Used by `dex watch` to refresh the Go static graph in
	// lockstep with the chunk index. Returning an error is logged but
	// does not stop the watch loop — the chunk side is already committed
	// and the next event can retry.
	AfterIndex func(context.Context) error
	// OnIdle, if non-nil, fires OnIdleAfter has elapsed since the most
	// recent successful flush left the dirty flag false. Used to drive
	// background summary draining: the hook gets a context that the
	// watcher cancels as soon as the next fs event arrives, so a long
	// drain yields immediately when the user starts editing again.
	//
	// Return done=true to stop the idle cycle until the next flush;
	// done=false re-arms for another OnIdleAfter window (the hook
	// signals "more work waiting"). Errors are logged and treated as
	// done=true.
	OnIdle func(ctx context.Context) (done bool, err error)
	// OnIdleAfter is the quiet window before OnIdle fires; default 5s.
	// Ignored when OnIdle is nil.
	OnIdleAfter time.Duration
}

type Watcher struct {
	indexer *index.Indexer
	ig      *ignore.Matcher
	root    string
	opts    Options

	mu         sync.Mutex
	dirty      bool // events have arrived since the last successful flush
	running    bool // a flush goroutine is currently running
	timer      *time.Timer
	idleTimer  *time.Timer        // armed after a clean flush; nil otherwise
	idleCancel context.CancelFunc // cancels the in-flight OnIdle, if any
}

func New(idx *index.Indexer, ig *ignore.Matcher, root string, opt Options) *Watcher {
	if opt.Debounce <= 0 {
		opt.Debounce = 500 * time.Millisecond
	}
	if opt.Logger == nil {
		opt.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	return &Watcher{indexer: idx, ig: ig, root: root, opts: opt}
}

// Run starts the watch loop and blocks until ctx is cancelled or an
// unrecoverable error occurs.
func (w *Watcher) Run(ctx context.Context) error {
	fw, err := fsnotify.NewWatcher()
	if err != nil {
		return err
	}
	defer fw.Close()

	if err := w.addWatches(fw, w.root); err != nil {
		return err
	}
	if w.opts.Verbose {
		w.opts.Logger.Info("watch ready", "root", w.root, "debounce", w.opts.Debounce)
	}

	// Initial re-index (covers anything that changed while the daemon was
	// stopped).
	w.markDirty(ctx)

	for {
		select {
		case <-ctx.Done():
			return nil
		case ev, ok := <-fw.Events:
			if !ok {
				return errors.New("fsnotify events channel closed")
			}
			w.handle(ctx, fw, ev)
		case err, ok := <-fw.Errors:
			if !ok {
				return errors.New("fsnotify errors channel closed")
			}
			w.opts.Logger.Warn("fsnotify error", "err", err)
		}
	}
}

func (w *Watcher) handle(ctx context.Context, fw *fsnotify.Watcher, ev fsnotify.Event) {
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
			w.opts.Logger.Warn("addWatches failed", "path", ev.Name, "err", err)
		}
	}
	// File-level events that affect indexed content.
	if !ev.Has(fsnotify.Create) && !ev.Has(fsnotify.Write) && !ev.Has(fsnotify.Remove) && !ev.Has(fsnotify.Rename) {
		return
	}
	if !isDir && !ignore.IndexableExt(ev.Name) && !ignore.IndexableBasename(ev.Name) && !ev.Has(fsnotify.Remove) && !ev.Has(fsnotify.Rename) {
		return
	}
	if w.opts.Verbose {
		w.opts.Logger.Info("fs event", "op", ev.Op.String(), "path", rel)
	}
	w.markDirty(ctx)
}

// markDirty resets the debounce timer; on expiry it runs an index pass.
// Also preempts any pending or in-flight idle hook — fresh events mean
// the indexer is about to run again, so background work should yield.
func (w *Watcher) markDirty(ctx context.Context) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.dirty = true
	if w.idleTimer != nil {
		w.idleTimer.Stop()
		w.idleTimer = nil
	}
	if w.idleCancel != nil {
		w.idleCancel()
		w.idleCancel = nil
	}
	if w.timer != nil {
		w.timer.Stop()
	}
	w.timer = time.AfterFunc(w.opts.Debounce, func() { w.flush(ctx) })
}

func (w *Watcher) flush(ctx context.Context) {
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
		err := w.indexer.Run(ctx)
		if err != nil {
			if !errors.Is(err, context.Canceled) {
				w.opts.Logger.Error("re-index failed", "err", err)
			}
		} else {
			if w.opts.Verbose {
				w.opts.Logger.Info("re-indexed", "elapsed", time.Since(start).Round(time.Millisecond))
			}
			if w.opts.AfterIndex != nil {
				if hookErr := w.opts.AfterIndex(ctx); hookErr != nil && !errors.Is(hookErr, context.Canceled) {
					w.opts.Logger.Warn("post-index hook failed", "err", hookErr)
				}
			}
		}

		// If events landed during the run, re-flush in the same goroutine
		// rather than spawning another one. This serializes work and
		// guarantees no concurrent indexers.
		w.mu.Lock()
		if !w.dirty || ctx.Err() != nil {
			w.running = false
			// Successful drain → arm idle work; on ctx-cancel skip it.
			if ctx.Err() == nil {
				w.armIdleLocked(ctx)
			}
			w.mu.Unlock()
			return
		}
		w.dirty = false
		w.mu.Unlock()
	}
}

// armIdleLocked schedules an OnIdle tick OnIdleAfter from now. Caller
// must hold w.mu. No-op when OnIdle is unconfigured or ctx is dead.
func (w *Watcher) armIdleLocked(ctx context.Context) {
	if w.opts.OnIdle == nil || ctx.Err() != nil {
		return
	}
	after := w.opts.OnIdleAfter
	if after <= 0 {
		after = 5 * time.Second
	}
	if w.idleTimer != nil {
		w.idleTimer.Stop()
	}
	w.idleTimer = time.AfterFunc(after, func() { w.runIdle(ctx) })
}

// runIdle invokes the OnIdle callback with a cancellable child of
// ctx. If the next fs event arrives during the callback, markDirty
// cancels that child so a long drain yields immediately. done=false
// re-arms the idle timer for another window; done=true (or an error)
// stops the cycle until the next flush.
func (w *Watcher) runIdle(ctx context.Context) {
	w.mu.Lock()
	w.idleTimer = nil
	if w.dirty || w.running || ctx.Err() != nil || w.opts.OnIdle == nil {
		w.mu.Unlock()
		return
	}
	ic, cancel := context.WithCancel(ctx)
	w.idleCancel = cancel
	w.mu.Unlock()

	done, err := w.opts.OnIdle(ic)
	cancel()
	if err != nil && !errors.Is(err, context.Canceled) {
		w.opts.Logger.Warn("idle hook failed", "err", err)
	}

	w.mu.Lock()
	defer w.mu.Unlock()
	w.idleCancel = nil
	if w.dirty || ctx.Err() != nil {
		return
	}
	if !done && err == nil {
		w.armIdleLocked(ctx)
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
				w.opts.Logger.Warn("fw.Add failed", "path", path, "err", err)
			}
		}
		return nil
	})
}
