package mcp

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

// ─── pairSiblingTests ────────────────────────────────────────────────────

func TestPairSiblingTests(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "pkg", "foo.go"), "package pkg\n")
	writeFile(t, filepath.Join(root, "pkg", "foo_test.go"), "package pkg\n")
	writeFile(t, filepath.Join(root, "pkg", "lone.go"), "package pkg\n")
	writeFile(t, filepath.Join(root, "py", "bar.py"), "x = 1\n")
	writeFile(t, filepath.Join(root, "py", "test_bar.py"), "x = 1\n")
	writeFile(t, filepath.Join(root, "ts", "qux.ts"), "export {}\n")
	writeFile(t, filepath.Join(root, "ts", "qux.test.ts"), "test\n")

	cases := []struct {
		path string
		want []string
	}{
		{"pkg/foo.go", []string{"pkg/foo_test.go"}},
		{"pkg/lone.go", nil}, // no sibling test exists
		{"pkg/foo_test.go", nil}, // already a test, don't recurse
		{"py/bar.py", []string{"py/test_bar.py"}},
		{"ts/qux.ts", []string{"ts/qux.test.ts"}},
		{"unknown.cpp", nil}, // unsupported extension
	}
	for _, tc := range cases {
		got := pairSiblingTests(root, tc.path)
		if len(got) != len(tc.want) {
			t.Errorf("pairSiblingTests(%q) = %v, want %v", tc.path, got, tc.want)
			continue
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("pairSiblingTests(%q)[%d] = %q, want %q", tc.path, i, got[i], tc.want[i])
			}
		}
	}
}

// ─── findNearestDoc ──────────────────────────────────────────────────────

func TestFindNearestDoc(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "README.md"), "# root\n")
	writeFile(t, filepath.Join(root, "internal", "deep", "pkg", "code.go"), "package pkg\n")
	writeFile(t, filepath.Join(root, "internal", "CLAUDE.md"), "# claude\n")

	// CLAUDE.md at internal/ wins over README.md at root.
	got := findNearestDoc(root, "internal/deep/pkg/code.go")
	if got != "internal/CLAUDE.md" {
		t.Errorf("got %q, want internal/CLAUDE.md", got)
	}

	// A file at the top level falls back to root README.md.
	writeFile(t, filepath.Join(root, "top.go"), "package x\n")
	got = findNearestDoc(root, "top.go")
	if got != "README.md" {
		t.Errorf("got %q, want README.md", got)
	}

	// The README itself shouldn't be returned as its own nearest doc.
	got = findNearestDoc(root, "README.md")
	if got == "README.md" {
		t.Error("findNearestDoc should not return the file itself")
	}
}

// ─── readSignatureAndDoc ─────────────────────────────────────────────────

func TestReadSignatureAndDoc(t *testing.T) {
	root := t.TempDir()
	src := strings.Join([]string{
		"package x",
		"",
		"// Greet returns a greeting for name.",
		"// Empty name returns a fallback.",
		"func Greet(name string) string {",
		"  return \"hi \" + name",
		"}",
		"",
	}, "\n") + "\n"
	path := filepath.Join(root, "g.go")
	writeFile(t, path, src)

	sig, doc := readSignatureAndDoc(path, 5)
	if !strings.HasPrefix(sig, "func Greet") {
		t.Errorf("sig=%q, want to start with `func Greet`", sig)
	}
	if !strings.Contains(doc, "Greet returns a greeting") {
		t.Errorf("doc=%q, want it to contain the docstring", doc)
	}
	if !strings.Contains(doc, "Empty name returns a fallback") {
		t.Errorf("doc=%q, want both comment lines", doc)
	}

	// Symbol with no leading comment gets empty doc.
	src2 := "package x\n\nfunc Bare() {}\n"
	path2 := filepath.Join(root, "b.go")
	writeFile(t, path2, src2)
	_, doc2 := readSignatureAndDoc(path2, 3)
	if doc2 != "" {
		t.Errorf("doc=%q, want empty for func with no comment", doc2)
	}
}

func TestReadSignatureAndDocStalenessGuard(t *testing.T) {
	// Simulate a stale index: the recorded StartLine points at a
	// line that no longer holds a declaration. Both fields must come
	// back empty rather than emitting whatever junk lives at that
	// offset.
	root := t.TempDir()
	src := strings.Join([]string{
		"package x",
		"",
		"// what was once a doc comment for a moved func.",
		"BuildTags string `json:\"build_tags,omitempty\"`", // not a declaration
		"",
	}, "\n") + "\n"
	path := filepath.Join(root, "stale.go")
	writeFile(t, path, src)

	sig, doc := readSignatureAndDoc(path, 4)
	if sig != "" || doc != "" {
		t.Errorf("stale-line should suppress fields; got sig=%q doc=%q", sig, doc)
	}
}

func TestLooksLikeDeclaration(t *testing.T) {
	cases := map[string]bool{
		"func Greet(name string) string {":  true,
		"  func (s *Server) Search() {}":    true,
		"type Foo struct {":                  true,
		"def hello(name):":                   true,
		"class Greeter:":                     true,
		"export function hi() {":             true,
		"const x = 1":                        true,
		"pub fn run() {}":                    true,
		"":                                   false,
		"// just a comment":                  false,
		"BuildTags string `json:\"...\"`":    false,
		"return nil":                         false,
		"x := 1":                             false,
	}
	for line, want := range cases {
		if got := looksLikeDeclaration(line); got != want {
			t.Errorf("looksLikeDeclaration(%q) = %v, want %v", line, got, want)
		}
	}
}

// ─── readBuildTagsAndPackage ─────────────────────────────────────────────

func TestReadBuildTagsAndPackage(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "f.go")
	writeFile(t, path, "//go:build linux && cgo\n\npackage mcp\n\nfunc X() {}\n")

	tags, pkg := readBuildTagsAndPackage(path)
	if !strings.Contains(tags, "//go:build linux") {
		t.Errorf("tags=%q, want it to contain //go:build linux", tags)
	}
	if pkg != "mcp" {
		t.Errorf("pkg=%q, want mcp", pkg)
	}

	// Legacy `// +build` line.
	path2 := filepath.Join(root, "legacy.go")
	writeFile(t, path2, "// +build !windows\n\npackage mcp\n")
	tags2, pkg2 := readBuildTagsAndPackage(path2)
	if !strings.Contains(tags2, "// +build !windows") {
		t.Errorf("tags=%q, want legacy +build line", tags2)
	}
	if pkg2 != "mcp" {
		t.Errorf("pkg=%q, want mcp", pkg2)
	}

	// No build tag, just package.
	path3 := filepath.Join(root, "plain.go")
	writeFile(t, path3, "package mcp\n")
	tags3, pkg3 := readBuildTagsAndPackage(path3)
	if tags3 != "" {
		t.Errorf("tags=%q, want empty when no build constraint", tags3)
	}
	if pkg3 != "mcp" {
		t.Errorf("pkg=%q, want mcp", pkg3)
	}
}

// ─── matchOwners ─────────────────────────────────────────────────────────

func TestMatchOwners(t *testing.T) {
	rules := []codeownersRule{
		{pattern: "*", owners: []string{"@everyone"}},
		{pattern: "internal/", owners: []string{"@backend"}},
		{pattern: "*.md", owners: []string{"@docs"}},
	}
	cases := []struct {
		path string
		want []string
	}{
		{"README.md", []string{"@docs"}},        // *.md wins (later)
		{"internal/store.go", []string{"@backend"}}, // internal/ matches prefix
		{"cmd/main.go", []string{"@everyone"}}, // only the * wildcard matches
	}
	for _, tc := range cases {
		got := matchOwners(rules, tc.path)
		if len(got) != len(tc.want) || (len(got) > 0 && got[0] != tc.want[0]) {
			t.Errorf("matchOwners(%q) = %v, want %v", tc.path, got, tc.want)
		}
	}
}

// ─── parseRipgrepLine ────────────────────────────────────────────────────

func TestParseRipgrepLine(t *testing.T) {
	cases := []struct {
		in       string
		wantPath string
		wantLine int
		wantSnip string
		wantOk   bool
	}{
		{"foo.go:42:    Search()", "foo.go", 42, "    Search()", true},
		{"a/b/c.go:1:package c", "a/b/c.go", 1, "package c", true},
		{"malformed", "", 0, "", false},
		{"foo.go:notanumber:x", "", 0, "", false},
	}
	for _, tc := range cases {
		p, l, s, ok := parseRipgrepLine(tc.in)
		if ok != tc.wantOk || p != tc.wantPath || l != tc.wantLine || s != tc.wantSnip {
			t.Errorf("parseRipgrepLine(%q) = (%q, %d, %q, %v); want (%q, %d, %q, %v)",
				tc.in, p, l, s, ok, tc.wantPath, tc.wantLine, tc.wantSnip, tc.wantOk)
		}
	}
}

// ─── enrich() integration ────────────────────────────────────────────────

func TestEnrichEditingContext(t *testing.T) {
	root := t.TempDir()
	// Write a source file with sibling test, doc, and build tag.
	writeFile(t, filepath.Join(root, "pkg", "core.go"),
		"//go:build !cgo\n\npackage pkg\n\n// Run does the thing.\nfunc Run() {}\n")
	writeFile(t, filepath.Join(root, "pkg", "core_test.go"), "package pkg\n")
	writeFile(t, filepath.Join(root, "pkg", "CLAUDE.md"), "# claude\n")
	writeFile(t, filepath.Join(root, "CODEOWNERS"), "pkg/ @backend\n")

	out := &ContextOutput{
		SuggestedReads: []SuggestedRead{{Path: "pkg/core.go", StartLine: 6, EndLine: 6}},
		Symbols:        []SymbolHit{{QualifiedName: "Run", Path: "pkg/core.go", StartLine: 6, EndLine: 6}},
	}
	enrich(context.Background(), root, IntentEditingContext, out)

	if out.Symbols[0].Signature == "" {
		t.Error("symbol signature should be populated")
	}
	if !strings.Contains(out.Symbols[0].Doc, "Run does the thing") {
		t.Errorf("symbol doc should mention `Run does the thing`; got %q", out.Symbols[0].Doc)
	}

	meta, ok := out.Annotations["pkg/core.go"]
	if !ok {
		t.Fatal("annotations should include pkg/core.go")
	}
	if len(meta.Tests) == 0 || meta.Tests[0] != "pkg/core_test.go" {
		t.Errorf("tests should pair core_test.go; got %v", meta.Tests)
	}
	if meta.NearestDoc != "pkg/CLAUDE.md" {
		t.Errorf("nearest doc should be pkg/CLAUDE.md; got %q", meta.NearestDoc)
	}
	if !strings.Contains(meta.BuildTags, "//go:build !cgo") {
		t.Errorf("build tags should be picked up; got %q", meta.BuildTags)
	}
	if meta.Package != "pkg" {
		t.Errorf("package=%q, want pkg", meta.Package)
	}
	if len(meta.Owners) == 0 || meta.Owners[0] != "@backend" {
		t.Errorf("owners should be matched from CODEOWNERS; got %v", meta.Owners)
	}
	// LastCommit may be empty if we're not inside a git checkout in the
	// temp dir — don't assert on it.
}

func TestEnrichBehaviorSearchOmitsHeavyLegs(t *testing.T) {
	// behavior_search should populate tests + nearest_doc (always-on)
	// but skip blame, owners, build tags, references.
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "x.go"), "package x\n\nfunc F() {}\n")
	writeFile(t, filepath.Join(root, "x_test.go"), "package x\n")
	writeFile(t, filepath.Join(root, "README.md"), "# x\n")
	writeFile(t, filepath.Join(root, "CODEOWNERS"), "* @owner\n")

	out := &ContextOutput{
		SuggestedReads: []SuggestedRead{{Path: "x.go", StartLine: 3, EndLine: 3}},
		Symbols:        []SymbolHit{{QualifiedName: "F", Path: "x.go", StartLine: 3, EndLine: 3}},
	}
	enrich(context.Background(), root, IntentBehaviorSearch, out)

	meta, ok := out.Annotations["x.go"]
	if !ok {
		t.Fatal("expected annotations for x.go")
	}
	if len(meta.Tests) == 0 {
		t.Error("tests pairing is always-on; should be populated")
	}
	if meta.NearestDoc == "" {
		t.Error("nearest_doc is always-on; should be populated")
	}
	if len(meta.Owners) != 0 {
		t.Errorf("owners should be skipped for behavior_search; got %v", meta.Owners)
	}
	if meta.BuildTags != "" || meta.Package != "" {
		t.Errorf("build tags / package should be skipped for behavior_search; got %q / %q", meta.BuildTags, meta.Package)
	}
	if len(out.References) != 0 {
		t.Errorf("references should be skipped for behavior_search; got %d", len(out.References))
	}
}

