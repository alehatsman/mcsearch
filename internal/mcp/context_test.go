package mcp

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/alehatsman/mcsearch/internal/graph"
	"github.com/alehatsman/mcsearch/internal/proj"
	"github.com/alehatsman/mcsearch/internal/store"
)

// ─── resolveIntent ────────────────────────────────────────────────────────

func TestResolveIntent(t *testing.T) {
	cases := []struct {
		name string
		in   ContextInput
		want string
	}{
		// Explicit Intent wins.
		{"explicit callers", ContextInput{Intent: IntentCallers, Question: "fix the bug"}, IntentCallers},
		{"explicit upper", ContextInput{Intent: "ARCHITECTURE", Question: "fix the bug"}, IntentArchitecture},
		{"explicit auto falls through", ContextInput{Intent: "auto", Question: "fix the rerank pool"}, IntentEditingContext},
		{"invalid intent falls through", ContextInput{Intent: "frobnicate", Question: "callers of Foo"}, IntentCallers},

		// Keyword regex.
		{"callers", ContextInput{Question: "callers of (*Store).Search"}, IntentCallers},
		{"who calls", ContextInput{Question: "who calls Search"}, IntentCallers},
		{"callees", ContextInput{Question: "what does Search call"}, IntentCallees},
		{"architecture", ContextInput{Question: "how does indexing work"}, IntentArchitecture},
		{"overview", ContextInput{Question: "give me an overview of the indexer"}, IntentArchitecture},
		{"packages", ContextInput{Question: "show the package topology"}, IntentPackageTopology},
		{"editing", ContextInput{Question: "fix the rerank pool overflow"}, IntentEditingContext},

		// Bare identifier query → symbol_lookup.
		{"bare qualified", ContextInput{Question: "(*Store).Search"}, IntentSymbolLookup},
		{"bare pascal", ContextInput{Question: "OpenWith"}, IntentSymbolLookup},
		{"bare camel", ContextInput{Question: "inlineContent"}, IntentSymbolLookup},

		// Default: behavior_search.
		{"plain question", ContextInput{Question: "where do we open the SQLite store"}, IntentBehaviorSearch},

		// Priority: callers beats editing when both present.
		{"callers beats editing", ContextInput{Question: "fix the callers of Search"}, IntentCallers},

		// change/update no longer trigger editing_context — they're too
		// noisy on questions like "when X changes" / "update the timestamp".
		{"change is behavior_search", ContextInput{Question: "where does the cache invalidate when chunks change"}, IntentBehaviorSearch},
		{"update is behavior_search", ContextInput{Question: "what triggers an update to last_indexed"}, IntentBehaviorSearch},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, _ := resolveIntent(tc.in)
			if got != tc.want {
				t.Errorf("resolveIntent(%+v) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// ─── extractIdentifiers ───────────────────────────────────────────────────

func TestExtractIdentifiers(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"callers of (*Store).Search", []string{"(*Store).Search"}},
		{"where is OpenWith defined", []string{"OpenWith"}},
		{"the user_role table and old_users column", []string{"user_role", "old_users"}},
		{"plain english only", nil},
		{"a Foo Bar duplicate Foo", []string{"Foo", "Bar"}},
		// camelCase — Go unexported names should be picked up.
		{"inlineContent", []string{"inlineContent"}},
		{"where is markDirty called", []string{"markDirty"}},
		// A camelCase token inside a qualified form must not double-add
		// (the qualified span masks sub-token matches).
		{"(*Store).searchRaw", []string{"(*Store).searchRaw"}},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			got := extractIdentifiers(tc.in)
			if len(got) != len(tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("idx %d: got %q, want %q (full: %v)", i, got[i], tc.want[i], got)
				}
			}
		})
	}
}

// ─── pickSuggestedReads ───────────────────────────────────────────────────

func TestPickSuggestedReadsSymbolIntent(t *testing.T) {
	syms := []SymbolHit{
		{QualifiedName: "Foo", Path: "a.go", StartLine: 10, EndLine: 20},
		{QualifiedName: "Bar", Path: "b.go", StartLine: 5, EndLine: 15},
	}
	got := pickSuggestedReads(IntentSymbolLookup, nil, syms, nil)
	if len(got) != 2 || got[0].Path != "a.go" || got[1].Path != "b.go" {
		t.Fatalf("got %+v", got)
	}
	if got[0].Reason != "definition of Foo" {
		t.Errorf("reason=%q", got[0].Reason)
	}
}

func TestPickSuggestedReadsCrossLaneBias(t *testing.T) {
	// b.go appears in both lanes; a.go is semantic-only with higher score.
	// Cross-lane agreement should bump b.go to the top regardless of score.
	sem := []SemHit{
		{Path: "a.go", StartLine: 1, EndLine: 10, Score: 0.9},
		{Path: "b.go", StartLine: 1, EndLine: 10, Score: 0.7},
	}
	symbolPaths := map[string]struct{}{"b.go": {}}

	got := pickSuggestedReads(IntentBehaviorSearch, sem, nil, symbolPaths)
	if len(got) != 2 {
		t.Fatalf("got %+v", got)
	}
	if got[0].Path != "b.go" {
		t.Errorf("cross-lane should win: got %s first, want b.go", got[0].Path)
	}
	if !strings.Contains(got[0].Reason, "symbol agreement") {
		t.Errorf("reason should mention symbol agreement: %q", got[0].Reason)
	}
}

func TestPickSuggestedReadsCodePreferredOverDocs(t *testing.T) {
	// A README at 0.66 shouldn't beat the .go file at 0.51 for a
	// behavior_search question — the rerank stage occasionally lets
	// docs outscore the code they describe, so the router applies a
	// tiebreaker.
	sem := []SemHit{
		{Path: "README.md", Score: 0.66},
		{Path: "internal/store/store.go", Score: 0.51},
	}
	got := pickSuggestedReads(IntentBehaviorSearch, sem, nil, nil)
	if len(got) == 0 || got[0].Path != "internal/store/store.go" {
		t.Errorf("behavior_search should prefer .go over .md tiebreaker; got %+v", got)
	}

	// Architecture explicitly welcomes README — no demotion.
	gotArch := pickSuggestedReads(IntentArchitecture, sem, nil, nil)
	if len(gotArch) == 0 || gotArch[0].Path != "README.md" {
		t.Errorf("architecture should keep README on top; got %+v", gotArch)
	}
}

func TestPickSuggestedReadsCodePreferredOverBuildFiles(t *testing.T) {
	// Taskfile.yml shouldn't beat the .go file it wraps when the
	// intent is implementation-oriented. Architecture intentionally
	// keeps the build file since it can reveal structure.
	sem := []SemHit{
		{Path: "Taskfile.yml", Score: 0.66},
		{Path: "internal/mcp/server.go", Score: 0.40},
	}
	got := pickSuggestedReads(IntentEditingContext, sem, nil, nil)
	if len(got) == 0 || got[0].Path != "internal/mcp/server.go" {
		t.Errorf("editing_context should prefer .go over Taskfile.yml; got %+v", got)
	}
	gotArch := pickSuggestedReads(IntentArchitecture, sem, nil, nil)
	if len(gotArch) == 0 || gotArch[0].Path != "Taskfile.yml" {
		t.Errorf("architecture should leave score order intact; got %+v", gotArch)
	}
}

func TestIsBuildOrConfigPath(t *testing.T) {
	tests := map[string]bool{
		"Taskfile.yml":    true,
		"Taskfile.yaml":   true,
		"Dockerfile":      true,
		"Makefile":        true,
		".github/ci.yml":  true,
		"config.toml":     true,
		"internal/x.go":   false,
		"README.md":       false,
		"go.mod":          false, // intentionally not demoted
		"package.json":    false,
	}
	for p, want := range tests {
		if got := isBuildOrConfigPath(p); got != want {
			t.Errorf("isBuildOrConfigPath(%q) = %v, want %v", p, got, want)
		}
	}
}

func TestIsDocPath(t *testing.T) {
	tests := map[string]bool{
		"README.md":              true,
		"docs/spec.rst":          true,
		"NOTES.txt":              true,
		"docs/page.adoc":         true,
		"site/post.mdx":          true,
		"internal/store/store.go": false,
		"cmd/main.py":             false,
	}
	for p, want := range tests {
		if got := isDocPath(p); got != want {
			t.Errorf("isDocPath(%q) = %v, want %v", p, got, want)
		}
	}
}

func TestIsTestPath(t *testing.T) {
	tests := map[string]bool{
		"internal/mcp/context_test.go":   true,
		"pkg/foo/bar_test.go":            true,
		"src/Foo.test.ts":                true,
		"src/Foo.test.tsx":               true,
		"src/foo.spec.js":                true,
		"src/foo.spec.jsx":               true,
		"tests/test_foo.py":              true,
		"tests/foo_test.py":              true,
		"spec/foo_spec.rb":               true,
		"src/foo_test.rs":                true,
		"internal/store/store.go":        false,
		"README.md":                      false,
		"cmd/main.py":                    false,
		"src/foo.ts":                     false,
	}
	for p, want := range tests {
		if got := isTestPath(p); got != want {
			t.Errorf("isTestPath(%q) = %v, want %v", p, got, want)
		}
	}
}

// ─── enrichGraph caps ─────────────────────────────────────────────────────

// TestEnrichGraphCaps guards against unbounded graph payloads — the
// regression that motivated maxGraphNodes/maxGraphEdges. A big
// package's rollup or a god-struct's sibling fan-out should not blow
// the response budget.
func TestEnrichGraphCaps(t *testing.T) {
	t.Run("node cap via package rollup", func(t *testing.T) {
		view := &graphView{
			nodesByID:        map[string]graphNode{},
			nodesByName:      map[string][]graphNode{},
			nodesByQualified: map[string][]graphNode{},
			nodesByPackage:   map[string][]graphNode{},
			nodesByPath:      map[string][]graphNode{},
			edgesBySrc:       map[string][]graphEdge{},
			edgesByDst:       map[string][]graphEdge{},
			edgesByKind:      map[graph.EdgeKind][]graphEdge{},
		}
		const pkg = "example.com/bigpkg"
		for i := range 100 {
			n := graphNode{
				ID:            fmt.Sprintf("n%d", i),
				Kind:          graph.NodeFunction,
				Name:          fmt.Sprintf("Fn%d", i),
				QualifiedName: fmt.Sprintf("Fn%d", i),
				PackagePath:   pkg,
				FilePath:      "bigpkg/bigpkg.go",
			}
			view.nodesByID[n.ID] = n
			view.nodesByPackage[pkg] = append(view.nodesByPackage[pkg], n)
			view.nodesByPath[n.FilePath] = append(view.nodesByPath[n.FilePath], n)
		}
		out := &ContextOutput{}
		enrichGraph(out, IntentArchitecture, view, []SemHit{{Path: "bigpkg/bigpkg.go"}}, nil)
		if got := len(out.Graph.Nodes); got > maxGraphNodes {
			t.Errorf("got %d nodes, want ≤ %d", got, maxGraphNodes)
		}
		if len(out.Graph.Nodes) == 0 {
			t.Error("expected some nodes from package rollup")
		}
	})

	t.Run("edge cap via package_topology imports", func(t *testing.T) {
		view := &graphView{
			nodesByID:        map[string]graphNode{},
			nodesByName:      map[string][]graphNode{},
			nodesByQualified: map[string][]graphNode{},
			nodesByPackage:   map[string][]graphNode{},
			nodesByPath:      map[string][]graphNode{},
			edgesBySrc:       map[string][]graphEdge{},
			edgesByDst:       map[string][]graphEdge{},
			edgesByKind:      map[graph.EdgeKind][]graphEdge{},
		}
		const src = "example.com/src"
		srcPkg := graphNode{ID: "src", Kind: graph.NodePackage, Name: "src", PackagePath: src, FilePath: "src/src.go"}
		view.nodesByID[srcPkg.ID] = srcPkg
		view.nodesByPackage[src] = append(view.nodesByPackage[src], srcPkg)
		view.nodesByPath[srcPkg.FilePath] = append(view.nodesByPath[srcPkg.FilePath], srcPkg)
		for i := range 100 {
			dstID := fmt.Sprintf("dst%d", i)
			dst := graphNode{ID: dstID, Kind: graph.NodePackage, Name: dstID, PackagePath: "example.com/" + dstID}
			view.nodesByID[dstID] = dst
			e := graphEdge{Kind: graph.EdgeImports, SrcID: srcPkg.ID, DstID: dstID}
			view.edgesByKind[graph.EdgeImports] = append(view.edgesByKind[graph.EdgeImports], e)
			view.edgesBySrc[srcPkg.ID] = append(view.edgesBySrc[srcPkg.ID], e)
			view.edgesByDst[dstID] = append(view.edgesByDst[dstID], e)
		}
		out := &ContextOutput{}
		enrichGraph(out, IntentPackageTopology, view, []SemHit{{Path: "src/src.go"}}, nil)
		if got := len(out.Graph.Edges); got > maxGraphEdges {
			t.Errorf("got %d edges, want ≤ %d", got, maxGraphEdges)
		}
		if len(out.Graph.Edges) == 0 {
			t.Error("expected some edges from imports rollup")
		}
	})
}

// ─── inlineSuggestedReads ─────────────────────────────────────────────────

func TestInlineSuggestedReadsBasic(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "f.go"),
		"line 1\nline 2\nline 3\nline 4\nline 5\n")

	reads := []SuggestedRead{{Path: "f.go", StartLine: 2, EndLine: 4, Reason: "x"}}
	inlineContent(root, IntentBehaviorSearch, reads, nil)

	want := "line 2\nline 3\nline 4\n"
	if reads[0].Content != want {
		t.Errorf("content=%q want %q", reads[0].Content, want)
	}
	if reads[0].Truncated {
		t.Error("should not be truncated")
	}
}

func TestInlineSuggestedReadsPerReadLineCap(t *testing.T) {
	// Generate a 200-line file and ask for the whole thing. The
	// per-read cap (60 lines) should clip the content and flag it.
	root := t.TempDir()
	var b strings.Builder
	for i := 1; i <= 200; i++ {
		fmt.Fprintf(&b, "line %d\n", i)
	}
	writeFile(t, filepath.Join(root, "big.go"), b.String())

	reads := []SuggestedRead{{Path: "big.go", StartLine: 1, EndLine: 200}}
	inlineContent(root, IntentBehaviorSearch, reads, nil)

	if !reads[0].Truncated {
		t.Error("want truncated=true when range exceeds per-read cap")
	}
	got := strings.Count(reads[0].Content, "\n")
	if got > 60 {
		t.Errorf("got %d lines, want ≤60", got)
	}
	// EndLine on the wire stays as the original request so the
	// caller can issue a follow-up Read for the rest.
	if reads[0].EndLine != 200 {
		t.Errorf("EndLine=%d, want 200 (unchanged)", reads[0].EndLine)
	}
}

func TestInlineSuggestedReadsTotalByteBudget(t *testing.T) {
	// Three reads each at the per-read byte cap should exhaust the
	// total budget before all are filled.
	root := t.TempDir()
	// Write a file with ~30 long lines so any 60-line slice hits the
	// per-read byte cap (4 KB) first.
	var b strings.Builder
	for range 30 {
		b.WriteString(strings.Repeat("x", 500))
		b.WriteByte('\n')
	}
	for _, n := range []string{"a.go", "b.go", "c.go", "d.go"} {
		writeFile(t, filepath.Join(root, n), b.String())
	}
	reads := []SuggestedRead{
		{Path: "a.go", StartLine: 1, EndLine: 30},
		{Path: "b.go", StartLine: 1, EndLine: 30},
		{Path: "c.go", StartLine: 1, EndLine: 30},
		{Path: "d.go", StartLine: 1, EndLine: 30},
	}
	inlineContent(root, IntentBehaviorSearch, reads, nil)

	total := 0
	for _, r := range reads {
		total += len(r.Content)
	}
	if total > 12*1024 {
		t.Errorf("total inlined bytes %d > 12 KB cap", total)
	}
	// Last read should be empty — budget exhausted.
	if reads[len(reads)-1].Content != "" {
		t.Errorf("last read should be empty once budget is exhausted; got %d bytes", len(reads[len(reads)-1].Content))
	}
}

func TestInlineContentSemanticHitsAlsoFilled(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "a.go"), "line 1\nline 2\nline 3\n")
	writeFile(t, filepath.Join(root, "b.go"), "line A\nline B\nline C\n")

	reads := []SuggestedRead{{Path: "a.go", StartLine: 1, EndLine: 3}}
	sem := []SemHit{
		{Path: "a.go", StartLine: 1, EndLine: 3},
		{Path: "b.go", StartLine: 1, EndLine: 3},
	}
	inlineContent(root, IntentBehaviorSearch, reads, sem)

	if sem[0].Content == "" {
		t.Error("semantic_hits[0] should be filled (cache-hit from suggested_reads)")
	}
	if sem[1].Content == "" {
		t.Error("semantic_hits[1] should be filled (separate file, within budget)")
	}
	if reads[0].Content == "" {
		t.Error("suggested_reads[0] should still be filled")
	}
}

func TestInlineContentSharedBudgetDoesNotDoubleCharge(t *testing.T) {
	// Same range appears in both lanes; the read cache should serve
	// the second request without re-charging the budget, so plenty
	// of headroom remains for other hits.
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "shared.go"), "line 1\nline 2\nline 3\n")
	writeFile(t, filepath.Join(root, "other.go"), "x\ny\nz\n")

	reads := []SuggestedRead{{Path: "shared.go", StartLine: 1, EndLine: 3}}
	sem := []SemHit{
		{Path: "shared.go", StartLine: 1, EndLine: 3},
		{Path: "other.go", StartLine: 1, EndLine: 3},
	}
	inlineContent(root, IntentBehaviorSearch, reads, sem)

	if reads[0].Content == "" || sem[0].Content == "" || sem[1].Content == "" {
		t.Errorf("expected all three to be filled; got reads=%q sem0=%q sem1=%q",
			reads[0].Content, sem[0].Content, sem[1].Content)
	}
	if reads[0].Content != sem[0].Content {
		t.Errorf("dedup cache should return identical content")
	}
}

func TestInlineSuggestedReadsMissingFileGraceful(t *testing.T) {
	root := t.TempDir()
	reads := []SuggestedRead{{Path: "does-not-exist.go", StartLine: 1, EndLine: 5}}
	inlineContent(root, IntentBehaviorSearch, reads, nil) // must not panic
	if reads[0].Content != "" {
		t.Errorf("missing file should leave content empty, got %q", reads[0].Content)
	}
}

func TestContextRouterInlinesByDefault(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	writeFile(t, filepath.Join(projDir, "main.go"),
		"package main\n\nfunc Greet(name string) string { return \"hi \" + name }\nfunc Bye() {}\n")
	root := indexProject(t, projDir, cacheDir, srv.URL)
	s := newServer(srv.URL, cacheDir)

	_, out, err := s.ContextRouter(context.Background(), ContextInput{
		Question: "where do we greet users",
		Project:  root,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.SuggestedReads) == 0 {
		t.Fatal("want suggested_reads")
	}
	if out.SuggestedReads[0].Content == "" {
		t.Errorf("suggested_reads[0].Content should be inlined by default; got empty")
	}
}

func TestContextRouterNoInline(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	writeFile(t, filepath.Join(projDir, "main.go"),
		"package main\n\nfunc Greet(name string) string { return \"hi \" + name }\nfunc Bye() {}\n")
	root := indexProject(t, projDir, cacheDir, srv.URL)
	s := newServer(srv.URL, cacheDir)

	_, out, err := s.ContextRouter(context.Background(), ContextInput{
		Question: "where do we greet users",
		Project:  root,
		NoInline: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out.SuggestedReads) == 0 {
		t.Fatal("want suggested_reads")
	}
	for i, r := range out.SuggestedReads {
		if r.Content != "" {
			t.Errorf("suggested_reads[%d].Content should be empty with NoInline=true; got %d bytes", i, len(r.Content))
		}
	}
}

func TestPickSuggestedReadsArchitectureCap(t *testing.T) {
	sem := []SemHit{
		{Path: "a.go", Score: 0.9}, {Path: "b.go", Score: 0.8},
		{Path: "c.go", Score: 0.7}, {Path: "d.go", Score: 0.6},
		{Path: "e.go", Score: 0.5}, {Path: "f.go", Score: 0.4},
	}
	got := pickSuggestedReads(IntentArchitecture, sem, nil, nil)
	// Exploration intents widen to 5 reads so the initial bundle gives
	// the caller a real cross-file picture.
	if len(got) != 5 {
		t.Errorf("architecture should return 5 reads, got %d", len(got))
	}
}

func TestInlineCapsFor(t *testing.T) {
	exploration := []string{IntentArchitecture, IntentPackageTopology}
	targeted := []string{
		IntentBehaviorSearch, IntentSymbolLookup, IntentCallers,
		IntentCallees, IntentEditingContext,
	}
	for _, intent := range exploration {
		c := inlineCapsFor(intent)
		if c.totalBytesCap < 32*1024 {
			t.Errorf("%s totalBytesCap=%d, want ≥32 KB for exploration", intent, c.totalBytesCap)
		}
		if c.maxLinesPerRead < 100 {
			t.Errorf("%s maxLinesPerRead=%d, want ≥100 for exploration", intent, c.maxLinesPerRead)
		}
	}
	for _, intent := range targeted {
		c := inlineCapsFor(intent)
		if c.totalBytesCap > 16*1024 {
			t.Errorf("%s totalBytesCap=%d, want ≤16 KB for targeted intents", intent, c.totalBytesCap)
		}
	}
}

func TestInlineSuggestedReadsExplorationDenser(t *testing.T) {
	// 200-line file requested in full. targeted caps clip at 60 lines;
	// exploration caps should clip at 120.
	root := t.TempDir()
	var b strings.Builder
	for i := 1; i <= 200; i++ {
		fmt.Fprintf(&b, "line %d\n", i)
	}
	writeFile(t, filepath.Join(root, "big.go"), b.String())

	targeted := []SuggestedRead{{Path: "big.go", StartLine: 1, EndLine: 200}}
	inlineContent(root, IntentBehaviorSearch, targeted, nil)
	targetedLines := strings.Count(targeted[0].Content, "\n")

	exploration := []SuggestedRead{{Path: "big.go", StartLine: 1, EndLine: 200}}
	inlineContent(root, IntentArchitecture, exploration, nil)
	explorationLines := strings.Count(exploration[0].Content, "\n")

	if !(explorationLines > targetedLines) {
		t.Errorf("exploration should include more lines than targeted; got %d vs %d", explorationLines, targetedLines)
	}
	if explorationLines > 120 {
		t.Errorf("exploration line count %d exceeds expected 120 cap", explorationLines)
	}
}

// ─── buildNextAction / buildAvoid (prose) ─────────────────────────────────

func TestBuildNextAction(t *testing.T) {
	reads := []SuggestedRead{{Path: "x.go", StartLine: 10, EndLine: 30}}
	syms := []SymbolHit{{QualifiedName: "Foo", Path: "x.go"}}

	cases := []struct {
		intent string
		reads  []SuggestedRead
		syms   []SymbolHit
		topSem float32
		want   string // substring match
	}{
		{IntentSymbolLookup, reads, syms, 0.8, "Read x.go lines 10-30"},
		{IntentEditingContext, reads, syms, 0.8, "before editing"},
		{IntentBehaviorSearch, reads, syms, 0.8, "ground your answer"},
		{IntentArchitecture, reads, syms, 0.8, "structural overview"},
		{IntentCallers, nil, syms, 0.8, "Call-graph edges are not yet extracted"},
		{IntentSymbolLookup, nil, nil, 0, "Rephrase"},
		// Low-confidence: no symbols and top semantic score below the
		// threshold should route to the "weak match" branch instead of
		// confidently pointing at a noise hit.
		{IntentBehaviorSearch, reads, nil, 0.30, "Top semantic match is weak"},
		// Symbols present — confidence comes from the structural lane,
		// so the low-score branch must NOT trigger.
		{IntentBehaviorSearch, reads, syms, 0.30, "ground your answer"},
	}
	for _, tc := range cases {
		t.Run(tc.intent+" "+tc.want, func(t *testing.T) {
			got := buildNextAction(tc.intent, tc.reads, tc.syms, tc.topSem)
			if !strings.Contains(got, tc.want) {
				t.Errorf("got %q, want substring %q", got, tc.want)
			}
		})
	}
}

func TestBuildAvoid(t *testing.T) {
	sem := []SemHit{{Path: "a.go"}}
	syms := []SymbolHit{{QualifiedName: "Foo", Path: "a.go"}}

	cases := []struct {
		name         string
		intent       string
		sem          []SemHit
		syms         []SymbolHit
		graphIndexed bool
		want         string
	}{
		{"callers always warns", IntentCallers, sem, syms, true, "`calls` edges are not yet extracted"},
		{"callers without graph still warns", IntentCallers, sem, syms, false, "`calls` edges are not yet extracted"},
		{"symbol_lookup without graph nudges to index", IntentSymbolLookup, sem, syms, false, "Run `mcsearch graph index"},
		{"symbol_lookup with graph: don't grep", IntentSymbolLookup, sem, syms, true, "Do not grep"},
		{"behavior + both lanes", IntentBehaviorSearch, sem, syms, true, "Do not grep for the identifier"},
		{"behavior + symbols only", IntentBehaviorSearch, nil, syms, true, "Do not grep for the identifier"},
		{"behavior + semantic only", IntentBehaviorSearch, sem, nil, true, "Do not read entire files"},
		{"behavior + nothing", IntentBehaviorSearch, nil, nil, true, ""},
		// behavior_search without graph now also gets the index nag —
		// graph enrichment runs on every intent.
		{"behavior without graph nudges to index", IntentBehaviorSearch, sem, syms, false, "Run `mcsearch graph index"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := buildAvoid(tc.intent, tc.sem, tc.syms, tc.graphIndexed, false)
			if tc.want == "" {
				if got != "" {
					t.Errorf("want empty, got %q", got)
				}
				return
			}
			if !strings.Contains(got, tc.want) {
				t.Errorf("got %q, want substring %q", got, tc.want)
			}
		})
	}
}

// ─── integration: contextRouter end-to-end ────────────────────────────────

func TestContextRouterBehaviorSearch(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	writeFile(t, filepath.Join(projDir, "main.go"),
		"package main\n\nfunc Greet(name string) string { return \"hi \" + name }\nfunc Bye() {}\n")

	root := indexProject(t, projDir, cacheDir, srv.URL)
	s := newServer(srv.URL, cacheDir)

	_, out, err := s.ContextRouter(context.Background(), ContextInput{
		Question: "where do we greet users",
		Project:  root,
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.Status != "ok" {
		t.Fatalf("status=%s hint=%s", out.Status, out.Hint)
	}
	if out.Intent != IntentBehaviorSearch {
		t.Errorf("intent=%s, want behavior_search", out.Intent)
	}
	if len(out.SemanticHits) == 0 {
		t.Fatal("want semantic_hits, got 0")
	}
	if len(out.SuggestedReads) == 0 {
		t.Error("want suggested_reads")
	}
	if out.NextAction == "" {
		t.Error("want non-empty next_action prose")
	}
	// out.Graph is omitted when enrichGraph produced nothing — the
	// JSON tag is `omitempty`. We don't assert anything about it here;
	// the dedicated TestContextRouter*GraphPopulated tests cover the
	// populated path.
}

func TestContextRouterSymbolLookup(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	writeFile(t, filepath.Join(projDir, "main.go"),
		"package main\n\nfunc Greet(name string) string { return \"hi \" + name }\nfunc Bye() {}\n")

	root := indexProject(t, projDir, cacheDir, srv.URL)
	s := newServer(srv.URL, cacheDir)

	_, out, err := s.ContextRouter(context.Background(), ContextInput{
		Question: "Greet",
		Project:  root,
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.Status != "ok" {
		t.Fatalf("status=%s hint=%s", out.Status, out.Hint)
	}
	if out.Intent != IntentSymbolLookup {
		t.Errorf("intent=%s, want symbol_lookup", out.Intent)
	}
	if len(out.Symbols) == 0 {
		t.Fatal("want symbols")
	}
	if out.Symbols[0].QualifiedName != "Greet" {
		t.Errorf("symbol[0]=%s, want Greet", out.Symbols[0].QualifiedName)
	}
	if !strings.Contains(out.NextAction, "Read") {
		t.Errorf("next_action should be a Read directive: %q", out.NextAction)
	}
	// Without a graph indexed, avoid nudges toward `mcsearch graph index`.
	// With graph indexed it would say "Do not grep". Either is acceptable
	// here; the symbol_lookup path is exercised either way.
	if !strings.Contains(out.Avoid, "Do not grep") && !strings.Contains(out.Avoid, "mcsearch graph index") {
		t.Errorf("avoid should mention either don't-grep or graph-index nudge: %q", out.Avoid)
	}
}

func TestContextRouterCallersGraphDeferred(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	writeFile(t, filepath.Join(projDir, "main.go"),
		"package main\n\nfunc Search() {}\nfunc UsesSearch() { Search() }\n")
	root := indexProject(t, projDir, cacheDir, srv.URL)
	s := newServer(srv.URL, cacheDir)

	_, out, err := s.ContextRouter(context.Background(), ContextInput{
		Question: "callers of Search",
		Project:  root,
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.Intent != IntentCallers {
		t.Errorf("intent=%s, want callers", out.Intent)
	}
	// avoid message branches on whether the references lane (rg) found
	// usages. With rg available + a real usage in UsesSearch we expect
	// the "references already lists usages" variant; if rg isn't on
	// PATH or returned empty we fall back to the original "calls edges
	// not extracted" line. Both mention `calls` graph caveat.
	if !strings.Contains(out.Avoid, "calls") {
		t.Errorf("avoid should flag the calls-graph caveat: %q", out.Avoid)
	}
	// If rg is available, references should be populated for this fixture.
	if hasRg() && len(out.References) == 0 {
		t.Errorf("expected references for `Search` usages when rg is available; got 0")
	}
}

func hasRg() bool {
	_, err := exec.LookPath("rg")
	return err == nil
}

func TestContextRouterNoIndex(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	writeFile(t, filepath.Join(projDir, "x.go"), "package x\n")

	s := newServer(srv.URL, cacheDir)
	_, out, err := s.ContextRouter(context.Background(), ContextInput{
		Question: "anything",
		Project:  projDir,
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.Status != "no-index" {
		t.Errorf("status=%s, want no-index", out.Status)
	}
}

func TestContextRouterEmptyQuestion(t *testing.T) {
	s := newServer("http://127.0.0.1:0", t.TempDir())
	_, out, err := s.ContextRouter(context.Background(), ContextInput{Project: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if out.Status != "error" {
		t.Errorf("status=%s, want error", out.Status)
	}
}

func TestCompactID(t *testing.T) {
	cases := []struct {
		name string
		n    graphNode
		want string
	}{
		{"method", graphNode{Kind: graph.NodeMethod, Name: "ContextRouter", QualifiedName: "(*Server).ContextRouter", PackagePath: "github.com/foo/bar/internal/mcp"}, "mcp.(*Server).ContextRouter"},
		{"type", graphNode{Kind: graph.NodeStruct, Name: "Server", QualifiedName: "Server", PackagePath: "github.com/foo/bar/internal/mcp"}, "mcp.Server"},
		{"package", graphNode{Kind: graph.NodePackage, QualifiedName: "github.com/foo/bar/internal/mcp", PackagePath: "github.com/foo/bar/internal/mcp"}, "mcp"},
		{"import", graphNode{Kind: graph.NodeImport, Name: "sync", QualifiedName: "sync", PackagePath: "github.com/foo/bar/internal/mcp"}, "sync"},
		{"field", graphNode{Kind: graph.NodeField, Name: "ChatClient", QualifiedName: "Server.ChatClient", PackagePath: "github.com/foo/bar/internal/mcp"}, "mcp.Server.ChatClient"},
		{"stdlib pkg path", graphNode{Kind: graph.NodeFunction, Name: "Println", QualifiedName: "Println", PackagePath: "fmt"}, "fmt.Println"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := compactID(tc.n); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// seedGraph writes a synthetic graph for `root` directly via the
// store's upsert methods. Avoids invoking ExtractGo (which needs a
// real go.mod + GOPATH-resolvable imports) so we can test the router's
// graph integration on a one-file fixture.
func seedGraph(t *testing.T, ctx context.Context, root, cacheDir string) {
	t.Helper()
	p, err := proj.Resolve(root, cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	st, err := store.Open(ctx, p.DBPath)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()

	now := time.Now()
	typeID := "m::p::type::Store"
	methodID := "m::p::method::(*Store).Search"
	siblingID := "m::p::method::(*Store).Open"
	nodes := []store.GraphNodeRow{
		{ID: typeID, Kind: string(graph.NodeType), Name: "Store", QualifiedName: "Store",
			PackagePath: "p", FilePath: "main.go", StartLine: 1, EndLine: 5,
			MetadataJSON: []byte("{}"), ContentHash: "n1"},
		{ID: methodID, Kind: string(graph.NodeMethod), Name: "Search", QualifiedName: "(*Store).Search",
			PackagePath: "p", FilePath: "main.go", StartLine: 10, EndLine: 20,
			MetadataJSON: []byte("{}"), ContentHash: "n2"},
		{ID: siblingID, Kind: string(graph.NodeMethod), Name: "Open", QualifiedName: "(*Store).Open",
			PackagePath: "p", FilePath: "main.go", StartLine: 30, EndLine: 40,
			MetadataJSON: []byte("{}"), ContentHash: "n3"},
	}
	if err := st.GraphUpsertNodes(ctx, nodes, now); err != nil {
		t.Fatal(err)
	}
	edges := []store.GraphEdgeRow{
		{ID: "e1", Kind: string(graph.EdgeHasMethod), SrcID: typeID, DstID: methodID,
			FilePath: "main.go", StartLine: 10, EndLine: 20,
			MetadataJSON: []byte("{}"), ContentHash: "h1"},
		{ID: "e2", Kind: string(graph.EdgeHasMethod), SrcID: typeID, DstID: siblingID,
			FilePath: "main.go", StartLine: 30, EndLine: 40,
			MetadataJSON: []byte("{}"), ContentHash: "h2"},
	}
	if err := st.GraphUpsertEdges(ctx, edges, now); err != nil {
		t.Fatal(err)
	}
}

func TestContextRouterGraphSymbolLookup(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	writeFile(t, filepath.Join(projDir, "main.go"),
		"package main\n\ntype Store struct{}\nfunc (s *Store) Search() {}\nfunc (s *Store) Open() {}\n")
	root := indexProject(t, projDir, cacheDir, srv.URL)
	ctx := context.Background()
	seedGraph(t, ctx, root, cacheDir)

	s := newServer(srv.URL, cacheDir)
	_, out, err := s.ContextRouter(ctx, ContextInput{
		Question: "Search",
		Project:  root,
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.Status != "ok" {
		t.Fatalf("status=%s hint=%s", out.Status, out.Hint)
	}
	if out.Graph == nil || len(out.Graph.Nodes) == 0 {
		t.Fatalf("graph.nodes should be populated; got %+v", out.Graph)
	}

	var names, ids []string
	for _, n := range out.Graph.Nodes {
		names = append(names, n.QualifiedName)
		ids = append(ids, n.ID)
	}
	joinedNames := strings.Join(names, ",")
	for _, want := range []string{"Store", "(*Store).Search", "(*Store).Open"} {
		if !strings.Contains(joinedNames, want) {
			t.Errorf("graph.nodes should include %q; got %s", want, joinedNames)
		}
	}
	// IDs must be the compact form (`<pkg-tail>.<qualified-name>`),
	// never the legacy `<module>::<pkg>::<kind>::<qname>`.
	joinedIDs := strings.Join(ids, ",")
	if strings.Contains(joinedIDs, "::") {
		t.Errorf("graph.nodes[].id should be compact, not module-qualified; got %s", joinedIDs)
	}
	for _, want := range []string{"p.(*Store).Search", "p.(*Store).Open", "p.Store"} {
		if !strings.Contains(joinedIDs, want) {
			t.Errorf("graph.nodes[].id should include %q; got %s", want, joinedIDs)
		}
	}
	// Edges must reference the same compact IDs.
	for _, e := range out.Graph.Edges {
		if strings.Contains(e.From, "::") || strings.Contains(e.To, "::") {
			t.Errorf("edge id should be compact; got from=%q to=%q", e.From, e.To)
		}
	}
	if !strings.Contains(out.Avoid, "Do not grep") {
		t.Errorf("avoid should say don't grep when graph is indexed: %q", out.Avoid)
	}
}

func TestContextRouterKBudget(t *testing.T) {
	srv := fakeEmbed(t, 16)
	defer srv.Close()
	cacheDir := t.TempDir()
	projDir := t.TempDir()
	writeFile(t, filepath.Join(projDir, "a.go"), "package x\n\nfunc A() {}\n")
	writeFile(t, filepath.Join(projDir, "b.go"), "package x\n\nfunc B() {}\n")
	writeFile(t, filepath.Join(projDir, "c.go"), "package x\n\nfunc C() {}\n")
	root := indexProject(t, projDir, cacheDir, srv.URL)
	s := newServer(srv.URL, cacheDir)

	_, out, err := s.ContextRouter(context.Background(), ContextInput{
		Question: "function",
		Project:  root,
		K:        2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if out.Status != "ok" {
		t.Fatalf("status=%s hint=%s", out.Status, out.Hint)
	}
	if len(out.SemanticHits) > 2 {
		t.Errorf("k=2 should cap semantic_hits; got %d", len(out.SemanticHits))
	}
}
