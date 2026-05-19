package mcp

import (
	"context"
	"fmt"
	"os"
	"os/exec"
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

	sig, doc := readSignatureAndDoc(path, 5, "")
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
	_, doc2 := readSignatureAndDoc(path2, 3, "")
	if doc2 != "" {
		t.Errorf("doc=%q, want empty for func with no comment", doc2)
	}
}

func TestReadSignatureAndDocCommentLeadingStartLine(t *testing.T) {
	// The Go chunker stores StartLine at the first line of the chunk,
	// which for funcs/methods with a doc comment is the comment line —
	// not the `func` line. The reader must scan forward past blanks
	// and contiguous comments to find the declaration, and treat those
	// skipped comments as the doc.
	root := t.TempDir()
	src := strings.Join([]string{
		"package x",
		"",
		"// Search returns the top-k chunks ranked by hybrid scoring with optional",
		"// per-file diversity via Options.MaxHitsPerFile.",
		"func (s *Store) Search(ctx context.Context) ([]Hit, error) {",
		"  return nil, nil",
		"}",
		"",
	}, "\n") + "\n"
	path := filepath.Join(root, "s.go")
	writeFile(t, path, src)

	// StartLine = 3 (the first doc-comment line, mimicking the chunker).
	sig, doc := readSignatureAndDoc(path, 3, "")
	if !strings.HasPrefix(sig, "func (s *Store) Search") {
		t.Errorf("sig=%q, want to start with `func (s *Store) Search`", sig)
	}
	if !strings.Contains(doc, "Search returns the top-k chunks") {
		t.Errorf("doc=%q, want it to contain the docstring", doc)
	}
	if !strings.Contains(doc, "per-file diversity") {
		t.Errorf("doc=%q, want both comment lines", doc)
	}
}

func TestReadSignatureAndDocAdjacentFunctionDisambiguation(t *testing.T) {
	// Observed in production: when two functions are adjacent, the
	// chunker can record the SECOND function's chunk with start_line
	// pointing at the FIRST function's doc block (likely because the
	// chunk's content range extends back to include the section header
	// or trailing context). The forward scan would then return the
	// first function's signature. Passing wantName must redirect the
	// scan to the actual target function.
	root := t.TempDir()
	src := strings.Join([]string{
		"package x",
		"",
		"// First explains the first function.",
		"func First() bool {",
		"  return true",
		"}",
		"",
		"// Second explains the second function.",
		"func Second(n int) error {",
		"  return nil",
		"}",
		"",
	}, "\n") + "\n"
	path := filepath.Join(root, "adj.go")
	writeFile(t, path, src)

	// With wantName="Second" and start_line=3 (pointing at First's doc),
	// the scan must skip past `func First` and return Second.
	sig, doc := readSignatureAndDoc(path, 3, "Second")
	if !strings.HasPrefix(sig, "func Second") {
		t.Errorf("sig=%q, want `func Second` (disambiguation failed)", sig)
	}
	if !strings.Contains(doc, "Second explains") {
		t.Errorf("doc=%q, want Second's doc comment, not First's", doc)
	}
	if strings.Contains(doc, "First explains") {
		t.Errorf("doc=%q, leaked First's comment", doc)
	}

	// Without wantName, legacy behavior — accept the first declaration.
	sigLegacy, _ := readSignatureAndDoc(path, 3, "")
	if !strings.HasPrefix(sigLegacy, "func First") {
		t.Errorf("empty wantName should accept first decl; got %q", sigLegacy)
	}

	// declarationMentions is identifier-boundary aware — "Search" must
	// not match inside "searchRaw" / "SearchSummaries".
	if declarationMentions("func searchRaw() {}", "Search") {
		t.Error("`Search` should not match inside searchRaw")
	}
	if declarationMentions("func SearchSummaries() {}", "Search") {
		t.Error("`Search` should not match inside SearchSummaries")
	}
	if !declarationMentions("func (s *Store) Search() {}", "Search") {
		t.Error("`Search` should match the actual Search method")
	}
}

func TestReadSignatureAndDocFieldShape(t *testing.T) {
	// Struct fields and Python attrs don't start with a declaration
	// keyword — they start with the field name itself. When wantName
	// is supplied AND the line's leading token equals wantName, accept
	// the line as the signature without requiring `func`/`type`/etc.
	root := t.TempDir()
	src := strings.Join([]string{
		"package x",
		"",
		"// Options controls one index run.",
		"type Options struct {",
		"\t// MaxFileSize caps file size in bytes.",
		"\tMaxFileSize int64",
		"\tVerbose     bool",
		"}",
		"",
	}, "\n") + "\n"
	path := filepath.Join(root, "opts.go")
	writeFile(t, path, src)

	// Field MaxFileSize lives at line 6 (1-indexed). With wantName
	// set, the scan must accept the field line as the signature and
	// pull the "// MaxFileSize caps file size in bytes." doc comment
	// from the line above.
	sig, doc := readSignatureAndDoc(path, 6, "MaxFileSize")
	if !strings.HasPrefix(sig, "MaxFileSize") {
		t.Errorf("sig=%q, want it to start with MaxFileSize", sig)
	}
	if !strings.Contains(doc, "MaxFileSize caps file size") {
		t.Errorf("doc=%q, want the field's leading comment", doc)
	}

	// Without wantName, the field line is not declaration-shaped and
	// must NOT be accepted (legacy staleness-guard semantics).
	sig2, _ := readSignatureAndDoc(path, 6, "")
	if sig2 != "" {
		t.Errorf("without wantName, field line should be rejected; got sig=%q", sig2)
	}

	// startsWithName boundary check: token must end at a non-ident byte.
	if !startsWithName("MaxFileSize int64", "MaxFileSize") {
		t.Error("startsWithName should match `MaxFileSize int64`")
	}
	if !startsWithName("\tMaxFileSize int64", "MaxFileSize") {
		t.Error("startsWithName should ignore leading tabs")
	}
	if startsWithName("MaxFileSizeOther int64", "MaxFileSize") {
		t.Error("startsWithName must respect identifier boundary")
	}
	if startsWithName("Other MaxFileSize int64", "MaxFileSize") {
		t.Error("startsWithName must require leading position")
	}
	// `name(` is a call — must reject.
	if startsWithName("extractFile(p, file)", "extractFile") {
		t.Error("startsWithName must reject call sites (name followed by paren)")
	}
	if startsWithName("\textractFile(ctx)", "extractFile") {
		t.Error("startsWithName must reject calls even with leading whitespace")
	}
	// Sanity: a field with parens-in-type (Go func type, etc.) — common
	// shape `Callback func(...)`. The leading token is `Callback`, next
	// non-ident char is space (then `func`), so it's accepted.
	if !startsWithName("Callback func(int) error", "Callback") {
		t.Error("startsWithName should accept field whose type is a function type")
	}
}

func TestReadSignatureAndDocMultiLineSignature(t *testing.T) {
	// Multi-line function signatures (Go with one param per line, etc.)
	// must be assembled into a single signature string — the agent
	// needs the params to understand the API. Previously the reader
	// returned only the first line (`func extractFile(`), giving
	// nothing actionable.
	root := t.TempDir()
	src := strings.Join([]string{
		"package x",
		"",
		"// extractFile processes one parsed file.",
		"func extractFile(",
		"\tp *Pkg,",
		"\tfile *File,",
		"\tfset *FileSet,",
		"\tnodes *NodeSet,",
		") {",
		"\treturn",
		"}",
	}, "\n") + "\n"
	path := filepath.Join(root, "f.go")
	writeFile(t, path, src)

	sig, doc := readSignatureAndDoc(path, 3, "extractFile")
	if !strings.HasPrefix(sig, "func extractFile(") {
		t.Errorf("sig=%q, want it to start with `func extractFile(`", sig)
	}
	if !strings.Contains(sig, "p *Pkg") {
		t.Errorf("sig=%q, must include first param", sig)
	}
	if !strings.Contains(sig, "nodes *NodeSet") {
		t.Errorf("sig=%q, must include last param", sig)
	}
	if !strings.HasSuffix(strings.TrimSpace(sig), "{") {
		t.Errorf("sig=%q, should end at body opener `{`", sig)
	}
	if !strings.Contains(doc, "extractFile processes one parsed file") {
		t.Errorf("doc=%q should still contain leading comment", doc)
	}
}

func TestAssembleSignatureSinglePass(t *testing.T) {
	cases := []struct {
		lines []string
		want  string
	}{
		// Single-line signature passes through untouched.
		{
			lines: []string{"func Greet(name string) string {"},
			want:  "func Greet(name string) string {",
		},
		// Multi-line Go params.
		{
			lines: []string{
				"func Big(",
				"\ta int,",
				"\tb int,",
				") int {",
			},
			want: "func Big( a int, b int, ) int {",
		},
		// Python multi-line def with `:` terminator.
		{
			lines: []string{
				"def big(",
				"    a: int,",
				"    b: int,",
				") -> int:",
			},
			want: "def big( a: int, b: int, ) -> int:",
		},
		// Interface method signature (no body opener, balanced).
		{
			lines: []string{"M(int) error"},
			want:  "M(int) error",
		},
	}
	for _, tc := range cases {
		got := assembleSignature(tc.lines, 0)
		if got != tc.want {
			t.Errorf("assembleSignature(%v) = %q, want %q", tc.lines, got, tc.want)
		}
	}

	// signatureParenDelta ignores strings.
	if d := signatureParenDelta(`s := "(\""`); d != 0 {
		t.Errorf("paren delta should ignore strings; got %d for sig", d)
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

	sig, doc := readSignatureAndDoc(path, 4, "")
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
	enrich(context.Background(), root, IntentEditingContext, 8, out)

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
	enrich(context.Background(), root, IntentBehaviorSearch, 8, out)

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


// ─── runReferencesLane: k drives the cap ─────────────────────────────────

func TestRunReferencesLaneCapsScaleWithK(t *testing.T) {
	if _, err := exec.LookPath("rg"); err != nil {
		t.Skip("ripgrep not on PATH")
	}
	root := t.TempDir()
	// Write 50 files each with a single usage of `Target` — that's
	// well past the default 30/20 caps so we can observe k's effect.
	for i := 0; i < 50; i++ {
		path := filepath.Join(root, fmt.Sprintf("u%d.go", i))
		body := "package x\n\nfunc _" + fmt.Sprint(i) + "() { Target() }\n"
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// Definition lives in a separate file so it's filtered out, not counted.
	if err := os.WriteFile(filepath.Join(root, "def.go"),
		[]byte("package x\n\nfunc Target() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	def := SymbolHit{
		QualifiedName: "Target",
		Path:          "def.go",
		StartLine:     3,
		EndLine:       3,
	}

	// With a single symbol the per-symbol cap dominates (the total cap
	// never kicks in). At k=8: perSymCap=clamp(8*3,20,60)=24.
	got8 := runReferencesLane(context.Background(), root, 8, []SymbolHit{def})
	wantPerSym8, _ := refCapsFor(8)
	if len(got8) != wantPerSym8 {
		t.Errorf("k=8 produced %d refs, want %d (per-symbol cap)", len(got8), wantPerSym8)
	}

	// At k=30: perSymCap=60, total=100; the fixture has only 50 usages
	// so all 50 surface. The point of the test is k=30 > k=8.
	got30 := runReferencesLane(context.Background(), root, 30, []SymbolHit{def})
	if len(got30) <= len(got8) {
		t.Errorf("k=30 yielded %d refs, k=8 yielded %d; expected wider cap to surface more usages",
			len(got30), len(got8))
	}
	if len(got30) != 50 {
		t.Errorf("k=30 with 50 usages should surface all 50; got %d", len(got30))
	}
}

func TestRefCapsFor(t *testing.T) {
	cases := []struct {
		k                          int
		wantPerSym, wantTotal int
	}{
		{0, defaultRefsPerSymbol, defaultRefHits},
		{8, defaultRefsPerSymbol, defaultRefHits}, // 8*3=24 > 20; 8*4=32 > 30; both should be > floor
		{1, defaultRefsPerSymbol, defaultRefHits},
		{30, maxRefsPerSymbol, maxRefHits},
		{100, maxRefsPerSymbol, maxRefHits}, // clamp to ceiling
	}
	for _, tc := range cases {
		perSym, total := refCapsFor(tc.k)
		// For mid-range k, just sanity-check monotonicity and bounds.
		if perSym < defaultRefsPerSymbol || perSym > maxRefsPerSymbol {
			t.Errorf("k=%d perSym=%d out of [%d,%d]", tc.k, perSym, defaultRefsPerSymbol, maxRefsPerSymbol)
		}
		if total < defaultRefHits || total > maxRefHits {
			t.Errorf("k=%d total=%d out of [%d,%d]", tc.k, total, defaultRefHits, maxRefHits)
		}
		// Endpoint asserts.
		if tc.k == 0 && (perSym != tc.wantPerSym || total != tc.wantTotal) {
			t.Errorf("k=0 want floors; got perSym=%d total=%d", perSym, total)
		}
		if tc.k >= 30 && (perSym != tc.wantPerSym || total != tc.wantTotal) {
			t.Errorf("k=%d want ceilings; got perSym=%d total=%d", tc.k, perSym, total)
		}
	}
}
