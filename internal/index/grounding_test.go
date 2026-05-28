package index

import (
	"strings"
	"testing"
)

// TestBuildPackageUserPrompt_Empty asserts the ungrounded path emits a
// byte-shape compatible with the pre-grounding prompt: PACKAGE header,
// no SYMBOLS/IMPORTS sections, then FILE SUMMARIES.
func TestBuildPackageUserPrompt_Empty(t *testing.T) {
	got := buildPackageUserPrompt("internal/foo", []string{"file a summary", "file b summary"}, pkgGrounding{})
	if !strings.HasPrefix(got, "PACKAGE: internal/foo\n\n") {
		t.Errorf("missing PACKAGE header at start; got:\n%s", got)
	}
	if strings.Contains(got, "EXPORTED SYMBOLS") {
		t.Errorf("EXPORTED SYMBOLS section should be absent for empty grounding; got:\n%s", got)
	}
	if strings.Contains(got, "PROJECT IMPORTS") {
		t.Errorf("PROJECT IMPORTS section should be absent for empty grounding; got:\n%s", got)
	}
	if !strings.Contains(got, "FILE SUMMARIES:\nfile a summary\n\n---\n\nfile b summary") {
		t.Errorf("file summaries section malformed; got:\n%s", got)
	}
}

// TestBuildPackageUserPrompt_Grounded asserts SYMBOLS / IMPORTS appear
// when grounding is supplied, and that the names land verbatim so the
// model can constrain its output to them.
func TestBuildPackageUserPrompt_Grounded(t *testing.T) {
	g := pkgGrounding{
		Symbols:        []string{"Store", "Open", "UpsertMany"},
		ProjectImports: []string{"internal/chunk", "internal/proj"},
	}
	got := buildPackageUserPrompt("internal/store", []string{"only file summary"}, g)

	if !strings.Contains(got, "EXPORTED SYMBOLS:\nStore, Open, UpsertMany") {
		t.Errorf("symbols section missing or wrong shape; got:\n%s", got)
	}
	if !strings.Contains(got, "PROJECT IMPORTS:\ninternal/chunk, internal/proj") {
		t.Errorf("imports section missing or wrong shape; got:\n%s", got)
	}
	if strings.Contains(got, "Open\nUpsertMany") {
		t.Errorf("symbols should be comma-joined on one line, not newline-separated; got:\n%s", got)
	}
}

// TestBuildRepoUserPrompt_Empty asserts the ungrounded repo prompt is
// byte-compatible with the pre-grounding shape (just PACKAGE SUMMARIES).
func TestBuildRepoUserPrompt_Empty(t *testing.T) {
	got := buildRepoUserPrompt([]string{"pkg a", "pkg b"}, repoGrounding{})
	if strings.Contains(got, "PACKAGES") {
		t.Errorf("PACKAGES section should be absent for empty grounding; got:\n%s", got)
	}
	if !strings.HasPrefix(got, "PACKAGE SUMMARIES:\npkg a") {
		t.Errorf("must start with PACKAGE SUMMARIES when ungrounded; got:\n%s", got)
	}
}

// TestBuildRepoUserPrompt_Grounded asserts the PACKAGES section lists
// each dir + its top symbols on its own line, and dirs with no symbols
// still appear so the model knows the package exists.
func TestBuildRepoUserPrompt_Grounded(t *testing.T) {
	g := repoGrounding{
		Packages: []repoPkgGrounding{
			{Dir: "internal/store", TopSymbols: []string{"Open", "Search", "UpsertMany"}},
			{Dir: "internal/proj"}, // no symbols
		},
	}
	got := buildRepoUserPrompt([]string{"store summary", "proj summary"}, g)

	if !strings.Contains(got, "PACKAGES (dir → top exported symbols):\n") {
		t.Errorf("PACKAGES header missing; got:\n%s", got)
	}
	if !strings.Contains(got, "- internal/store → Open, Search, UpsertMany") {
		t.Errorf("store entry missing or wrong; got:\n%s", got)
	}
	if !strings.Contains(got, "- internal/proj\n") {
		t.Errorf("no-symbols package should still appear as a bare entry; got:\n%s", got)
	}
	if !strings.Contains(got, "PACKAGE SUMMARIES:\nstore summary") {
		t.Errorf("package summaries section malformed; got:\n%s", got)
	}
}

// TestProjectImports_Trims asserts external imports are dropped and the
// module prefix is stripped from project ones.
func TestProjectImports_Trims(t *testing.T) {
	imps := []string{
		"github.com/alehatsman/dex/internal/store",
		"github.com/alehatsman/dex/internal/chunk",
		"github.com/mattn/go-sqlite3",
		"strings",
	}
	got := projectImports(imps, "github.com/alehatsman/dex")
	want := []string{"internal/store", "internal/chunk"}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("[%d] got %q, want %q", i, got[i], want[i])
		}
	}
}

// TestProjectImports_EmptyModPath asserts a non-Go project (no go.mod)
// gets nil — caller falls back to the ungrounded prompt cleanly.
func TestProjectImports_EmptyModPath(t *testing.T) {
	got := projectImports([]string{"a", "b"}, "")
	if got != nil {
		t.Errorf("got %v, want nil for empty modPath", got)
	}
}
