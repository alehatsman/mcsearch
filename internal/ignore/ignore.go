// Package ignore decides which files to skip during indexing.
//
// Three layers, evaluated in order:
//  1. Built-in defaults (always-skip: vendor dirs, build outputs, secret
//     filenames). Hard-coded; not overridable.
//  2. .gitignore at the project root (best-effort — we read the root
//     file only, not nested .gitignore. Enough in practice for the
//     ignore-most-of-vendored-stuff job).
//  3. .mcsearch-ignore at the project root (same syntax as .gitignore).
//
// A separate secret pre-scan checks file contents for AWS keys / private
// keys / GitHub tokens before embedding.
package ignore

import (
	"bufio"
	"bytes"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// DefaultPatterns are always applied, on top of whatever .gitignore or
// .mcsearch-ignore say. Kept conservative to avoid surprising omissions.
var DefaultPatterns = []string{
	".env",
	".env.*",
	"*.pem",
	"*.key",
	"*.key.gpg",
	"*.kdbx",
	"id_rsa",
	"id_rsa.pub",
	"id_ed25519",
	"id_ed25519.pub",
	"id_ecdsa",
	"id_ecdsa.pub",
	"authorized_keys",
	"known_hosts",
	"secrets.yml",
	"secrets.yaml",
	"*.tfvars",
	".terraform/",
	"node_modules/",
	"vendor/",
	".venv/",
	"venv/",
	".tox/",
	"__pycache__/",
	"target/",
	"dist/",
	"build/",
	".next/",
	".cache/",
	".git/",
	".hg/",
	".svn/",
	"*.log",
	"*.lock",
	"*.min.js",
	"*.min.css",
}

// IndexableExtensions are the file extensions mcsearch will attempt to
// chunk. Everything else is skipped. Add to this list when extending the
// chunker to a new language.
var IndexableExtensions = map[string]bool{
	".go":       true,
	".py":       true,
	".js":       true,
	".jsx":      true,
	".ts":       true,
	".tsx":      true,
	".rs":       true,
	".java":     true,
	".kt":       true,
	".c":        true,
	".h":        true,
	".cc":       true,
	".cpp":      true,
	".hpp":      true,
	".rb":       true,
	".lua":      true,
	".sh":       true,
	".bash":     true,
	".zsh":      true,
	".md":       true,
	".rst":      true,
	".txt":      true,
	".yml":      true,
	".yaml":     true,
	".toml":     true,
	".json":     true,
	".sql":      true,
	".clj":      true,
	".cljs":     true,
	".cljc":     true,
}

// Matcher decides whether a path (relative to project root) should be
// indexed.
type Matcher struct {
	patterns []pattern
}

type pattern struct {
	raw      string
	negate   bool // leading `!`
	anchored bool // leading `/`
	dirOnly  bool // trailing `/`
	body     string
}

// New loads default + .gitignore + .mcsearch-ignore from root.
func New(root string) (*Matcher, error) {
	m := &Matcher{}
	m.addPatterns(DefaultPatterns)
	for _, name := range []string{".gitignore", ".mcsearch-ignore"} {
		f, err := os.Open(filepath.Join(root, name))
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		s := bufio.NewScanner(f)
		var lines []string
		for s.Scan() {
			lines = append(lines, s.Text())
		}
		f.Close()
		if err := s.Err(); err != nil {
			return nil, err
		}
		m.addPatterns(lines)
	}
	return m, nil
}

func (m *Matcher) addPatterns(lines []string) {
	for _, raw := range lines {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		p := pattern{raw: line}
		if strings.HasPrefix(line, "!") {
			p.negate = true
			line = line[1:]
		}
		if strings.HasPrefix(line, "/") {
			p.anchored = true
			line = line[1:]
		}
		if strings.HasSuffix(line, "/") {
			p.dirOnly = true
			line = strings.TrimSuffix(line, "/")
		}
		p.body = line
		m.patterns = append(m.patterns, p)
	}
}

// Match returns true if the relative path is ignored. relPath uses
// forward slashes (filepath.ToSlash). isDir hints the pattern matcher
// about trailing-slash patterns.
func (m *Matcher) Match(relPath string, isDir bool) bool {
	relPath = filepath.ToSlash(relPath)
	matched := false
	for _, p := range m.patterns {
		if !p.matches(relPath, isDir) {
			continue
		}
		if p.negate {
			matched = false
		} else {
			matched = true
		}
	}
	return matched
}

func (p pattern) matches(relPath string, isDir bool) bool {
	if p.dirOnly && !isDir {
		// `name/` only matches when the path itself names a directory.
		// We approximate by also matching descendants.
		// Fall through to the body matcher; it'll handle prefixes.
	}
	if p.anchored {
		return globMatch(p.body, relPath) || (p.dirOnly && strings.HasPrefix(relPath, p.body+"/"))
	}
	// Unanchored: match the basename, or any path segment.
	if globMatch(p.body, filepath.Base(relPath)) {
		return true
	}
	// Match any directory segment.
	segs := strings.Split(relPath, "/")
	for i := 0; i < len(segs); i++ {
		if globMatch(p.body, segs[i]) {
			return true
		}
	}
	// Match `dir/` as a prefix.
	if p.dirOnly && strings.Contains("/"+relPath+"/", "/"+p.body+"/") {
		return true
	}
	return false
}

// globMatch is filepath.Match with `**` support.
func globMatch(pat, name string) bool {
	if strings.Contains(pat, "**") {
		// Convert `**` to a regex `.*` and `*` to `[^/]*`. Cheap; we only
		// hit this for the few patterns that use `**`.
		var b strings.Builder
		b.WriteByte('^')
		i := 0
		for i < len(pat) {
			switch {
			case i+1 < len(pat) && pat[i] == '*' && pat[i+1] == '*':
				b.WriteString(".*")
				i += 2
			case pat[i] == '*':
				b.WriteString("[^/]*")
				i++
			case pat[i] == '?':
				b.WriteString("[^/]")
				i++
			case pat[i] == '.':
				b.WriteString(`\.`)
				i++
			default:
				b.WriteByte(pat[i])
				i++
			}
		}
		b.WriteByte('$')
		re, err := regexp.Compile(b.String())
		if err != nil {
			return false
		}
		return re.MatchString(name)
	}
	ok, _ := filepath.Match(pat, name)
	return ok
}

// IndexableExt returns true if the file extension is one mcsearch will
// attempt to chunk.
func IndexableExt(path string) bool {
	return IndexableExtensions[strings.ToLower(filepath.Ext(path))]
}

// secretPatterns are checked against the first 4 KB of any candidate file.
// A match causes the file to be skipped with a logged warning.
var secretPatterns = []*regexp.Regexp{
	regexp.MustCompile(`AKIA[0-9A-Z]{16}`),                   // AWS access key
	regexp.MustCompile(`-----BEGIN [A-Z ]*PRIVATE KEY-----`), // PEM private key
	regexp.MustCompile(`ghp_[A-Za-z0-9]{36}`),                // GitHub PAT (classic)
	regexp.MustCompile(`github_pat_[A-Za-z0-9_]{82}`),        // GitHub fine-grained PAT
	regexp.MustCompile(`xox[abps]-[A-Za-z0-9-]{10,}`),        // Slack token
	regexp.MustCompile(`sk-(?:proj-)?[A-Za-z0-9]{20,}`),      // OpenAI/Anthropic-style API key
}

// LooksLikeSecret returns true if the first 4 KB of data matches a
// well-known secret pattern.
func LooksLikeSecret(data []byte) bool {
	head := data
	if len(head) > 4096 {
		head = head[:4096]
	}
	for _, re := range secretPatterns {
		if re.Match(head) {
			return true
		}
	}
	return false
}

// LooksBinary returns true if data contains a NUL byte in the first 8 KB.
// Cheap heuristic to skip binary files that slipped through the extension
// filter (e.g. a `.yml` that's actually a packed binary).
func LooksBinary(data []byte) bool {
	head := data
	if len(head) > 8192 {
		head = head[:8192]
	}
	return bytes.IndexByte(head, 0) >= 0
}
