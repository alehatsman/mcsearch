package proj

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const EnvAllowPaths = "MCSEARCH_ALLOW_PATHS"

var systemHardDeny = map[string]struct{}{
	"/":            {},
	"/home":        {},
	"/Users":       {},
	"/usr":         {},
	"/var":         {},
	"/etc":         {},
	"/proc":        {},
	"/sys":         {},
	"/tmp":         {},
	"/mnt":         {},
	"/run":         {},
	"/dev":         {},
	"/boot":        {},
	"/opt":         {},
	"/srv":         {},
	"/lost+found":  {},
	"/System":      {},
	"/Library":     {},
	"/Applications": {},
	"/private":     {},
	"/cores":       {},
	"/Volumes":     {},
	"/Network":     {},
}

func CheckIndexable(p *Project, force bool) error {
	if force {
		return nil
	}
	root := filepath.Clean(p.Root)
	if _, hard := systemHardDeny[root]; hard {
		return fmt.Errorf("refusing to index %s: protected system path (re-run with --force if you really mean it)", root)
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" && root == filepath.Clean(home) {
		return fmt.Errorf("refusing to index %s: that's your home directory (re-run with --force if you really mean it)", root)
	}
	if isInGitWorkTree(root) {
		return nil
	}
	for _, prefix := range allowlistPrefixes() {
		if pathHasPrefix(root, prefix) {
			return nil
		}
	}
	return fmt.Errorf("refusing to index %s: not inside a git work tree and not under any %s prefix (re-run with --force, or add a prefix to %s)", root, EnvAllowPaths, EnvAllowPaths)
}

// .git is a file (not a dir) inside git worktrees and submodules.
func isInGitWorkTree(path string) bool {
	for {
		if _, err := os.Stat(filepath.Join(path, ".git")); err == nil {
			return true
		}
		parent := filepath.Dir(path)
		if parent == path {
			return false
		}
		path = parent
	}
}

func allowlistPrefixes() []string {
	raw := os.Getenv(EnvAllowPaths)
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, string(filepath.ListSeparator))
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = expandHome(strings.TrimSpace(p))
		if p == "" || !filepath.IsAbs(p) {
			continue
		}
		out = append(out, filepath.Clean(p))
	}
	return out
}

func expandHome(p string) string {
	if p == "~" || strings.HasPrefix(p, "~"+string(filepath.Separator)) {
		if home, err := os.UserHomeDir(); err == nil && home != "" {
			p = filepath.Join(home, strings.TrimPrefix(p, "~"))
		}
	}
	return os.ExpandEnv(p)
}

// strings.HasPrefix alone would falsely match "/foo" against "/foobar".
func pathHasPrefix(path, prefix string) bool {
	if path == prefix {
		return true
	}
	sep := string(filepath.Separator)
	if !strings.HasSuffix(prefix, sep) {
		prefix += sep
	}
	return strings.HasPrefix(path, prefix)
}
