// Package ignore decides which files to skip during indexing.
//
// The full filter chain a file passes through:
//
//  1. Allow-list gate (IndexableExt / IndexableBasename): the file's
//     extension must appear in IndexableExtensions, OR its basename in
//     IndexableBasenames. Anything else is dropped before any further
//     check runs. This is the first gate, not a fallback.
//
//  2. Matcher (gitignore-style exclusion): three sub-layers composed
//     and evaluated together by github.com/sabhiram/go-gitignore — so
//     full gitignore semantics apply (anchoring, negation, `**`,
//     dir-only patterns, later-pattern-wins). The sub-layers, in
//     declaration order:
//     a. DefaultPatterns — hard-coded: vendor dirs, build outputs,
//     secret-shaped filenames, license-family files.
//     b. .gitignore at the project root (root file only; nested
//     .gitignore files are intentionally not read).
//     c. .dex-ignore at the project root (same syntax).
//
//  3. LooksBinary — NUL-byte heuristic for binaries that slipped
//     through the allow-list (e.g. a `.yml` that's actually a packed
//     binary).
//
//  4. LooksLikeSecret — scans the first 4 KB of content against a
//     panel of well-known secret regexes (AWS, GitHub PAT, Slack,
//     OpenAI, Stripe, GitLab, SendGrid, …). Suppressed when
//     IsTestPath(path) is true so test files holding fake credentials
//     aren't dropped.
//
// MaxFileSize and other indexer-orchestration limits live in
// internal/index/index.go, not here.
package ignore

import (
	"bufio"
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	gitignore "github.com/sabhiram/go-gitignore"
)

// DefaultPatterns are always applied, on top of whatever .gitignore or
// .dex-ignore say. Kept conservative to avoid surprising omissions.
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
	".dex/",
	"*.log",
	"*.lock",
	"*.min.js",
	"*.min.css",
	// License / legal-text files. They index successfully but their
	// uniform legalese gives the embedder something to latch onto for
	// almost any query, polluting RAG context with chunks that have
	// nothing to do with the question. Cover the all-caps GNU/Apache
	// conventions plus common case variants and extension forms.
	"LICENSE",
	"LICENSE.*",
	"LICENCE",
	"LICENCE.*",
	"License",
	"License.*",
	"license",
	"license.*",
	"COPYING",
	"COPYING.*",
	"COPYRIGHT",
	"COPYRIGHT.*",
	"NOTICE",
	"NOTICE.*",
	"AUTHORS",
	"AUTHORS.*",
	"PATENTS",
	"PATENTS.*",
	"LEGAL",
	"LEGAL.*",
}

// IndexableExtensions are the file extensions dex will attempt to
// chunk. Everything else is skipped. Add to this list when extending the
// chunker to a new language.
var IndexableExtensions = map[string]bool{
	".go":      true,
	".py":      true,
	".js":      true,
	".jsx":     true,
	".ts":      true,
	".tsx":     true,
	".rs":      true,
	".java":    true,
	".kt":      true,
	".c":       true,
	".h":       true,
	".cc":      true,
	".cpp":     true,
	".hpp":     true,
	".rb":      true,
	".lua":     true,
	".sh":      true,
	".bash":    true,
	".zsh":     true,
	".md":      true,
	".rst":     true,
	".txt":     true,
	".yml":     true,
	".yaml":    true,
	".toml":    true,
	".json":    true,
	".sql":     true,
	".clj":     true,
	".cljs":    true,
	".cljc":    true,
	".scala":   true,
	".cs":      true,
	".swift":   true,
	".dart":    true,
	".php":     true,
	".hs":      true,
	".ex":      true,
	".exs":     true,
	".erl":     true,
	".tf":      true,
	".proto":   true,
	".graphql": true,
	".gql":     true,
	".r":       true,
	".jl":      true,
	".zig":     true,
	".fish":    true,
	".ps1":     true,
	".html":    true,
	".css":     true,
	".scss":    true,
	".vue":     true,
	".svelte":  true,
}

// Matcher decides whether a path (relative to project root) should be
// indexed. The gitignore grammar (patterns, anchoring, negation, `**`
// semantics) is delegated to github.com/sabhiram/go-gitignore so we
// don't reinvent — and subtly miss — the corner cases of a
// 20-year-old spec. We only contribute the DefaultPatterns,
// .dex-ignore composition, and the wider always-skip rules.
type Matcher struct {
	g *gitignore.GitIgnore
}

// New loads DefaultPatterns + project-root .gitignore + .dex-ignore
// (in that order — later wins per gitignore semantics).
func New(root string) (*Matcher, error) {
	var lines []string
	lines = append(lines, DefaultPatterns...)
	for _, name := range []string{".gitignore", ".dex-ignore"} {
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
		if errors.Is(err, os.ErrNotExist) {
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

// IndexableExt returns true if the file extension is one dex will
// attempt to chunk.
func IndexableExt(path string) bool {
	return IndexableExtensions[strings.ToLower(filepath.Ext(path))]
}

// IsTestPath returns true for file paths that conventionally hold
// test code or fixtures. The indexer uses this to suppress the
// secret-pattern skip: test files routinely embed fake credentials
// (`AKIA0123456789ABCDEF`, dummy PEM blocks) as inputs to their own
// detection logic, and refusing to index them was hiding real test
// code from search. The pattern check still runs against
// non-test files where a literal secret almost always is one.
func IsTestPath(relPath string) bool {
	p := filepath.ToSlash(relPath)
	base := filepath.Base(p)
	for _, dir := range []string{
		"tests/", "test/", "__tests__/", "spec/", "specs/", "testdata/", "fixtures/",
	} {
		if strings.Contains("/"+p+"/", "/"+dir) {
			return true
		}
	}
	// Go: foo_test.go
	if strings.HasSuffix(base, "_test.go") {
		return true
	}
	// Python: test_foo.py, foo_test.py
	if strings.HasSuffix(base, ".py") &&
		(strings.HasPrefix(base, "test_") || strings.HasSuffix(base, "_test.py")) {
		return true
	}
	// JS/TS/JSX/TSX: foo.test.* / foo.spec.*
	for _, ext := range []string{".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"} {
		if strings.HasSuffix(base, ".test"+ext) || strings.HasSuffix(base, ".spec"+ext) {
			return true
		}
	}
	// Rust: tests/*.rs is covered by the dir loop; integration tests
	// inside src often end with _test.rs.
	if strings.HasSuffix(base, "_test.rs") {
		return true
	}
	// Ruby: foo_spec.rb
	if strings.HasSuffix(base, "_spec.rb") || strings.HasSuffix(base, "_test.rb") {
		return true
	}
	return false
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
	"go.mod":         true,
	"go.work":        true,
	"Brewfile":       true,
	"Vagrantfile":    true,
	"Tiltfile":       true,
	"Caddyfile":      true,
	"Pipfile":        true,
	".editorconfig":  true,
	// LICENSE / COPYING / AUTHORS / NOTICE / PATENTS / LEGAL are
	// deliberately not whitelisted — DefaultPatterns above filters
	// them and their .md/.txt variants. Keep CHANGELOG and README
	// (substantive prose, not legal boilerplate).
	"CHANGELOG": true,
	"README":    true,
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
