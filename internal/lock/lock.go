// Package lock provides a cross-process advisory file lock for the
// per-project dex cache directory. It exists so two `dex index` (or
// `dex index summarize`, `dex reindex`, `dex watch`) invocations on
// the same project root don't both walk the tree, hammer the embed
// endpoint, and step on each other's prune passes.
//
// The lock is a POSIX flock(2) acquired non-blocking on a regular file
// at <CacheDir>/index.lock. The same file carries a small JSON payload
// describing the current holder so the loser of the race (and
// `dex index status`) can report who is in the way.
//
// flock semantics: the kernel releases the lock when the holding
// process exits, even on SIGKILL. That gives us automatic stale-lock
// recovery without a separate liveness check — but Steal is provided
// for the rare case where a holder leaks the fd to a child that
// outlives it.
package lock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"syscall"
	"time"
)

// ErrLocked is returned by Acquire when another process holds the
// lock. The caller can read holder details via ReadHolder(path).
var ErrLocked = errors.New("lock already held")

// Holder describes the process currently holding (or attempting to
// hold) the lock. It is serialized into the lock file as JSON so other
// processes can report it to the user.
type Holder struct {
	PID     int       `json:"pid"`
	Host    string    `json:"host,omitempty"`
	Command string    `json:"cmd,omitempty"`   // "index" | "summarize" | "reindex" | "watch"
	Phase   string    `json:"phase,omitempty"` // "chunk" | "graph" | "summarize" | "idle"
	Started time.Time `json:"started"`
}

// Lock is an acquired flock on a lock file. Release exactly once.
type Lock struct {
	path string
	f    *os.File
}

// Acquire takes the lock at path non-blocking. Returns ErrLocked if
// another process holds it. The holder JSON is written into the file
// while the lock is held; readers (ReadHolder) get a best-effort view.
//
// The lock file is created if missing. Parent directory must exist.
func Acquire(path string, h Holder) (*Lock, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open lock %s: %w", path, err)
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		_ = f.Close()
		if errors.Is(err, syscall.EWOULDBLOCK) {
			return nil, ErrLocked
		}
		return nil, fmt.Errorf("flock %s: %w", path, err)
	}
	l := &Lock{path: path, f: f}
	if err := l.writeHolder(h); err != nil {
		_ = l.Release()
		return nil, err
	}
	return l, nil
}

// AcquireWait blocks until the lock is taken or ctx is cancelled. It
// polls Acquire with exponential backoff (capped). Returns the same
// errors as Acquire, plus ctx.Err() if ctx ends.
func AcquireWait(ctx context.Context, path string, h Holder) (*Lock, error) {
	delay := 50 * time.Millisecond
	const maxDelay = 2 * time.Second
	for {
		l, err := Acquire(path, h)
		if err == nil {
			return l, nil
		}
		if !errors.Is(err, ErrLocked) {
			return nil, err
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
		}
		if delay < maxDelay {
			delay *= 2
			if delay > maxDelay {
				delay = maxDelay
			}
		}
	}
}

// Steal forcibly takes the lock by removing the existing lock file
// and acquiring fresh. Use only when the operator confirms the
// previous holder is gone (a leaked fd in a daemon, for example). A
// live flock cannot be broken by another process — if the holder is
// still running, Steal returns ErrLocked just like Acquire.
//
// Removing the file is safe under flock: the kernel keeps the inode
// alive for the holder; the next Acquire creates a fresh inode.
func Steal(path string, h Holder) (*Lock, error) {
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("remove stale lock: %w", err)
	}
	return Acquire(path, h)
}

// ReadHolder returns the JSON payload of the lock file without
// acquiring the lock. Returns (nil, nil) if the file does not exist.
// A malformed payload (e.g. mid-write race) is reported as an error
// so callers can fall back to "in flight, no detail".
func ReadHolder(path string) (*Holder, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	if len(data) == 0 {
		return nil, nil
	}
	var h Holder
	if err := json.Unmarshal(data, &h); err != nil {
		return nil, fmt.Errorf("decode holder: %w", err)
	}
	return &h, nil
}

// SetPhase updates the Phase field of the holder JSON without
// releasing the lock. Called as the indexer transitions between
// pipeline stages so `dex index status` can report progress.
func (l *Lock) SetPhase(phase string) error {
	h, err := readFromFile(l.f)
	if err != nil {
		return err
	}
	h.Phase = phase
	return l.writeHolder(*h)
}

// Release drops the flock and closes the underlying file. Idempotent.
func (l *Lock) Release() error {
	if l == nil || l.f == nil {
		return nil
	}
	// Truncate so a stale ReadHolder can tell the lock is unheld.
	_ = l.f.Truncate(0)
	err := syscall.Flock(int(l.f.Fd()), syscall.LOCK_UN)
	if cerr := l.f.Close(); err == nil {
		err = cerr
	}
	l.f = nil
	return err
}

// Path returns the lock file path. Mainly useful for diagnostics.
func (l *Lock) Path() string { return l.path }

// writeHolder serializes h into the lock file, truncating any prior
// content. Called while the flock is held, so the only concurrent
// reader is ReadHolder; the JSON payload fits comfortably in one
// page so write(2) is atomic w.r.t. read(2) on Linux.
func (l *Lock) writeHolder(h Holder) error {
	if h.Started.IsZero() {
		h.Started = time.Now()
	}
	data, err := json.Marshal(h)
	if err != nil {
		return fmt.Errorf("encode holder: %w", err)
	}
	if err := l.f.Truncate(0); err != nil {
		return fmt.Errorf("truncate lock: %w", err)
	}
	if _, err := l.f.WriteAt(data, 0); err != nil {
		return fmt.Errorf("write holder: %w", err)
	}
	return nil
}

// readFromFile reads the JSON holder from an already-open file
// descriptor. Used by SetPhase to round-trip the existing payload
// before mutating one field.
func readFromFile(f *os.File) (*Holder, error) {
	if _, err := f.Seek(0, 0); err != nil {
		return nil, err
	}
	var h Holder
	dec := json.NewDecoder(f)
	if err := dec.Decode(&h); err != nil {
		return nil, fmt.Errorf("decode holder: %w", err)
	}
	return &h, nil
}
