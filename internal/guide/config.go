// Package guide renders LLM_GUIDE.txt from existing summary chunks
// (repo_summary + package_summary) produced by the dex indexer.
//
// The guide is a markdown overview the user can drop into a pre-commit
// hook so the file in their repo stays in sync with the index. Render
// is read-only against the SQLite store and does no LLM calls — the
// summaries already exist as chunks; the renderer just formats them.
package guide

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Config controls guide output. Loaded from .dex/guide.toml at the
// project root. All fields have defaults; the file is optional.
type Config struct {
	Output string // markdown destination, default "LLM_GUIDE.txt"
}

// DefaultConfig returns the values used when .dex/guide.toml is absent.
func DefaultConfig() Config {
	return Config{Output: "LLM_GUIDE.txt"}
}

// LoadConfig reads .dex/guide.toml under root. Missing file is not an
// error — defaults are returned. The parser handles a deliberately tiny
// TOML subset (sections + bare key = "value" pairs) so dex doesn't
// pull in a TOML dependency for one file with three keys.
func LoadConfig(root string) (Config, error) {
	cfg := DefaultConfig()
	path := filepath.Join(root, ".dex", "guide.toml")
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil
		}
		return cfg, err
	}
	defer func() { _ = f.Close() }()

	var section string
	sc := bufio.NewScanner(f)
	for ln := 1; sc.Scan(); ln++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			section = strings.TrimSpace(line[1 : len(line)-1])
			continue
		}
		eq := strings.IndexByte(line, '=')
		if eq < 0 {
			return cfg, fmt.Errorf("%s:%d: expected key = value", path, ln)
		}
		key := strings.TrimSpace(line[:eq])
		val := strings.Trim(strings.TrimSpace(line[eq+1:]), `"`)
		switch section + "." + key {
		case "guide.output":
			cfg.Output = val
		}
	}
	return cfg, sc.Err()
}
