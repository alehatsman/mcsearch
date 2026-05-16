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

	gitignore "github.com/sabhiram/go-gitignore"
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
// indexed. The gitignore grammar (patterns, anchoring, negation, `**`
// semantics) is delegated to github.com/sabhiram/go-gitignore so we
// don't reinvent — and subtly miss — the corner cases of a
// 20-year-old spec. We only contribute the DefaultPatterns,
// .mcsearch-ignore composition, and the wider always-skip rules.
type Matcher struct {
	g *gitignore.GitIgnore
}

// New loads DefaultPatterns + project-root .gitignore + .mcsearch-ignore
// (in that order — later wins per gitignore semantics).
func New(root string) (*Matcher, error) {
	var lines []string
	lines = append(lines, DefaultPatterns...)
	for _, name := range []string{".gitignore", ".mcsearch-ignore"} {
		extra, err := readLines(filepath.Join(root, name))
		if err != nil {
			return nil, err
		}
		lines = append(lines, extra...)
	}
	return &Matcher{g: gitignore.CompileIgnoreLines(lines...)}, nil
}

// readLines returns the lines of path, or nil if the file doesn't exist.
func readLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	var out []string
	s := bufio.NewScanner(f)
	for s.Scan() {
		out = append(out, s.Text())
	}
	return out, s.Err()
}

// Match returns true if the relative path is ignored. relPath uses
// forward slashes (filepath.ToSlash). gitignore's dir-only patterns
// (`name/`) only match when the input path is itself a directory, so
// we append a trailing slash for directories — that's how the spec
// distinguishes them.
func (m *Matcher) Match(relPath string, isDir bool) bool {
	p := filepath.ToSlash(relPath)
	if isDir && !strings.HasSuffix(p, "/") {
		p += "/"
	}
	return m.g.MatchesPath(p)
}

// IndexableExt returns true if the file extension is one mcsearch will
// attempt to chunk.
func IndexableExt(path string) bool {
	return IndexableExtensions[strings.ToLower(filepath.Ext(path))]
}

// IndexableBasenames are well-known filenames that have no extension
// (or a misleading one) but whose content is still useful to index.
var IndexableBasenames = map[string]bool{
	"Makefile":       true,
	"makefile":       true,
	"GNUmakefile":    true,
	"Dockerfile":     true,
	"dockerfile":     true,
	"Containerfile":  true,
	"Justfile":       true,
	"justfile":       true,
	"CMakeLists.txt": true,
	"BUILD":          true,
	"BUILD.bazel":    true,
	"WORKSPACE":      true,
	"Rakefile":       true,
	"Gemfile":        true,
	"Procfile":       true,
}

// IndexableBasename returns true for known basenames that lack an
// indexable extension but contain code-like content worth chunking.
func IndexableBasename(path string) bool {
	return IndexableBasenames[filepath.Base(path)]
}

// secretPatterns are checked against the first 4 KB of any candidate file.
// A match causes the file to be skipped with a logged warning.
var secretPatterns = []*regexp.Regexp{
	regexp.MustCompile(`AKIA[0-9A-Z]{16}`),                       // AWS access key
	regexp.MustCompile(`ASIA[0-9A-Z]{16}`),                       // AWS STS temporary access key
	regexp.MustCompile(`-----BEGIN [A-Z ]*PRIVATE KEY-----`),     // PEM private key
	regexp.MustCompile(`ghp_[A-Za-z0-9]{36}`),                    // GitHub PAT (classic)
	regexp.MustCompile(`github_pat_[A-Za-z0-9_]{82}`),            // GitHub fine-grained PAT
	regexp.MustCompile(`xox[abps]-[A-Za-z0-9-]{10,}`),            // Slack token
	regexp.MustCompile(`sk-(?:proj-)?[A-Za-z0-9_-]{20,}`),        // OpenAI/Anthropic-style API key
	regexp.MustCompile(`AIza[0-9A-Za-z_-]{35}`),                  // Google API key
	regexp.MustCompile(`sk_live_[0-9a-zA-Z]{24,}`),               // Stripe live secret key
	regexp.MustCompile(`rk_live_[0-9a-zA-Z]{24,}`),               // Stripe restricted key
	regexp.MustCompile(`glpat-[A-Za-z0-9_-]{20,}`),               // GitLab personal access token
	regexp.MustCompile(`SG\.[A-Za-z0-9_-]{22,}\.[A-Za-z0-9_-]+`), // SendGrid
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
