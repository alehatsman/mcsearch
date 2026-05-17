package ignore

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultsIgnoreVendorDirs(t *testing.T) {
	root := t.TempDir()
	m, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		path    string
		isDir   bool
		ignored bool
	}{
		{"node_modules", true, true},
		{"node_modules/foo.js", false, true},
		{"vendor/bar/baz.go", false, true},
		{".git", true, true},
		{".git/HEAD", false, true},
		{"dist", true, true},
		{"build/out.txt", false, true},
		{".env", false, true},
		{".env.local", false, true},
		{"id_rsa", false, true},
		{"id_ed25519.pub", false, true},
		{"secrets.yml", false, true},
		{"foo.min.js", false, true},
		// negatives
		{"src/main.go", false, false},
		{"README.md", false, false},
		{".github/workflows/ci.yml", false, false},
	}
	for _, c := range cases {
		got := m.Match(c.path, c.isDir)
		if got != c.ignored {
			t.Errorf("Match(%q, isDir=%v) = %v, want %v", c.path, c.isDir, got, c.ignored)
		}
	}
}

func TestGitignoreAndMcsearchIgnore(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, ".gitignore"),
		[]byte("# project\n*.tmp\n/build\ndocs/private/\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, ".mcsearch-ignore"),
		[]byte("scratch/\n!scratch/keep.md\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		path    string
		isDir   bool
		ignored bool
	}{
		{"foo.tmp", false, true},
		{"a/b/c.tmp", false, true},
		{"build", true, true},
		{"build/x", false, true},
		{"docs/private", true, true},
		{"docs/private/secret.md", false, true},
		{"docs/public/howto.md", false, false},
		{"scratch", true, true},
		{"scratch/x.md", false, true},
		// negation re-includes a specific file
		{"scratch/keep.md", false, false},
		{"src/main.go", false, false},
	}
	for _, c := range cases {
		got := m.Match(c.path, c.isDir)
		if got != c.ignored {
			t.Errorf("Match(%q, isDir=%v) = %v, want %v", c.path, c.isDir, got, c.ignored)
		}
	}
}

func TestDoubleStarPatterns(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, ".mcsearch-ignore"),
		[]byte("**/__pycache__/\n**/*.bak\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := New(root)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		path    string
		isDir   bool
		ignored bool
	}{
		{"__pycache__", true, true},
		{"src/__pycache__", true, true},
		{"deep/a/b/__pycache__/foo.pyc", false, true},
		{"x.bak", false, true},
		{"deep/x.bak", false, true},
		{"src/main.py", false, false},
	}
	for _, c := range cases {
		got := m.Match(c.path, c.isDir)
		if got != c.ignored {
			t.Errorf("Match(%q, isDir=%v) = %v, want %v", c.path, c.isDir, got, c.ignored)
		}
	}
}

func TestIndexableExt(t *testing.T) {
	cases := map[string]bool{
		"main.go":     true,
		"main.GO":     true, // case-insensitive
		"app.py":      true,
		"x.rs":        true,
		"README.md":   true,
		"y.unknown":   false,
		"binary":      false,
		"image.png":   false,
		"sub/dir/x.c": true,
		// extended language coverage
		"App.swift":      true,
		"Program.cs":     true,
		"build.scala":    true,
		"main.dart":      true,
		"index.php":      true,
		"Lib.hs":         true,
		"app.ex":         true,
		"app.exs":        true,
		"node.erl":       true,
		"main.tf":        true,
		"schema.proto":   true,
		"schema.graphql": true,
		"schema.gql":     true,
		"analysis.r":     true,
		"workflow.jl":    true,
		"main.zig":       true,
		"config.fish":    true,
		"deploy.ps1":     true,
		// markup / web
		"index.html": true,
		"style.css":  true,
		"theme.scss": true,
		"App.vue":    true,
		"App.svelte": true,
	}
	for path, want := range cases {
		if got := IndexableExt(path); got != want {
			t.Errorf("IndexableExt(%q) = %v, want %v", path, got, want)
		}
	}
}

func TestIndexableBasename(t *testing.T) {
	cases := map[string]bool{
		"Makefile":             true,
		"GNUmakefile":          true,
		"Dockerfile":           true,
		"Containerfile":        true,
		"sub/Dockerfile":       true,
		"x.go":                 false,
		"makefile":             true,
		"random":               false,
		"CMakeLists.txt":       true,
		"build/CMakeLists.txt": true,
		// Go modules
		"go.mod":  true,
		"go.work": true,
		// Ruby DSL-style / dev environment
		"Brewfile":    true,
		"Vagrantfile": true,
		"Tiltfile":    true,
		"Caddyfile":   true,
		"Pipfile":     true,
		// Editor
		".editorconfig": true,
		// License / docs without extension
		"LICENSE":   true,
		"COPYING":   true,
		"AUTHORS":   true,
		"NOTICE":    true,
		"CHANGELOG": true,
		"README":    true,
		// negatives
		"go.sum":  false,
		"license": false, // case-sensitive on purpose
	}
	for path, want := range cases {
		if got := IndexableBasename(path); got != want {
			t.Errorf("IndexableBasename(%q) = %v, want %v", path, got, want)
		}
	}
}

func TestLooksLikeSecret(t *testing.T) {
	cases := []struct {
		blob string
		hit  bool
	}{
		{"// regular code\nfunc foo() {}", false},
		{"AWS_KEY=AKIA0123456789ABCDEF", true},
		{"-----BEGIN RSA PRIVATE KEY-----\nMIIB...\n", true},
		{"token=ghp_" + repeat("A", 36), true},
		{"token=AIza" + repeat("a", 35), true},
		{"key=sk_live_" + repeat("x", 30), true},
		{"key=glpat-" + repeat("x", 24), true},
		// Should NOT trigger: prefix lookalikes
		{"sk-but-short", false},
		{"BEGINPRIVATE KEY but not a real header", false},
	}
	for _, c := range cases {
		got := LooksLikeSecret([]byte(c.blob))
		if got != c.hit {
			t.Errorf("LooksLikeSecret(%q) = %v, want %v", trim(c.blob), got, c.hit)
		}
	}
}

func TestIsTestPath(t *testing.T) {
	cases := map[string]bool{
		// Go
		"internal/ignore/ignore_test.go": true,
		"main.go":                        false,
		// Python
		"tests/test_auth.py":  true,
		"src/test_helpers.py": true,
		"src/auth_test.py":    true,
		"src/auth.py":         false,
		// JS/TS
		"src/foo.test.js":   true,
		"src/foo.spec.ts":   true,
		"src/foo.ts":        false,
		"__tests__/util.ts": true,
		// Rust
		"tests/integration.rs": true,
		"src/util_test.rs":     true,
		"src/util.rs":          false,
		// Ruby
		"spec/models/user_spec.rb": true,
		"app/models/user.rb":       false,
		// Generic fixture dirs
		"testdata/sample.json": true,
		"fixtures/keys.txt":    true,
	}
	for path, want := range cases {
		if got := IsTestPath(path); got != want {
			t.Errorf("IsTestPath(%q) = %v, want %v", path, got, want)
		}
	}
}

func TestLooksBinary(t *testing.T) {
	if LooksBinary([]byte("hello world")) {
		t.Error("plain text flagged as binary")
	}
	if !LooksBinary([]byte("hello\x00world")) {
		t.Error("NUL byte not detected")
	}
	// 8 KB scanning window — content after should not affect detection.
	big := make([]byte, 16384)
	for i := range big {
		big[i] = 'a'
	}
	big[9000] = 0
	if LooksBinary(big) {
		t.Error("NUL past 8 KB window should be ignored (false positive)")
	}
}

func repeat(s string, n int) string {
	out := make([]byte, 0, len(s)*n)
	for range n {
		out = append(out, s...)
	}
	return string(out)
}

func trim(s string) string {
	if len(s) > 40 {
		return s[:40] + "…"
	}
	return s
}
