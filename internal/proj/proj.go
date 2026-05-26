// Package proj resolves a user-supplied project path to a canonical
// project root and a deterministic per-project cache directory.
package proj

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// Project identifies an indexed project on disk.
type Project struct {
	Root     string // canonical absolute path
	ID       string // sha256(Root) hex — primary key for the cache dir
	CacheDir string // $DEX_INDEX_DIR/<ID>
	DBPath   string // CacheDir/index.db
	LockPath string // CacheDir/index.lock — cross-process indexer mutex
}

// Resolve canonicalizes path and returns the project identity. The path
// must exist and be a directory. Errors are phrased for direct display
// to the user (no Go-internals like "eval symlinks: lstat …").
func Resolve(path, baseCacheDir string) (*Project, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("invalid path %q: %w", path, err)
	}
	real, err := filepath.EvalSymlinks(abs)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("path does not exist: %s", abs)
		}
		return nil, fmt.Errorf("resolve %s: %w", abs, err)
	}
	st, err := os.Stat(real)
	if err != nil {
		return nil, fmt.Errorf("stat %s: %w", real, err)
	}
	if !st.IsDir() {
		return nil, fmt.Errorf("%s is not a directory", real)
	}
	sum := sha256.Sum256([]byte(real))
	id := hex.EncodeToString(sum[:])
	cache := filepath.Join(baseCacheDir, id)
	return &Project{
		Root:     real,
		ID:       id,
		CacheDir: cache,
		DBPath:   filepath.Join(cache, "index.db"),
		LockPath: filepath.Join(cache, "index.lock"),
	}, nil
}

// EnsureCacheDir creates the per-project cache directory.
func (p *Project) EnsureCacheDir() error {
	return os.MkdirAll(p.CacheDir, 0o755)
}
