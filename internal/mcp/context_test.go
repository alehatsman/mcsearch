package mcp

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
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

		// Default: behavior_search.
		{"plain question", ContextInput{Question: "where do we open the SQLite store"}, IntentBehaviorSearch},

		// Priority: callers beats editing when both present.
		{"callers beats editing", ContextInput{Question: "fix the callers of Search"}, IntentCallers},
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

func TestPickSuggestedReadsArchitectureCap(t *testing.T) {
	sem := []SemHit{
		{Path: "a.go", Score: 0.9}, {Path: "b.go", Score: 0.8},
		{Path: "c.go", Score: 0.7}, {Path: "d.go", Score: 0.6},
	}
	got := pickSuggestedReads(IntentArchitecture, sem, nil, nil)
	if len(got) != 3 {
		t.Errorf("architecture should return 3 reads, got %d", len(got))
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
		want   string // substring match
	}{
		{IntentSymbolLookup, reads, syms, "Read x.go lines 10-30"},
		{IntentEditingContext, reads, syms, "before editing"},
		{IntentBehaviorSearch, reads, syms, "ground your answer"},
		{IntentArchitecture, reads, syms, "structural overview"},
		{IntentCallers, nil, syms, "Graph layer not available"},
		{IntentSymbolLookup, nil, nil, "Rephrase"},
	}
	for _, tc := range cases {
		t.Run(tc.intent+" "+tc.want, func(t *testing.T) {
			got := buildNextAction(tc.intent, tc.reads, tc.syms)
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
		intent string
		sem    []SemHit
		syms   []SymbolHit
		want   string
	}{
		{IntentCallers, sem, syms, "graph extraction is not yet wired"},
		{IntentBehaviorSearch, sem, syms, "Do not grep for the identifier"},
		{IntentBehaviorSearch, nil, syms, "Do not grep for the identifier"},
		{IntentBehaviorSearch, sem, nil, "Do not read entire files"},
		{IntentBehaviorSearch, nil, nil, ""},
	}
	for _, tc := range cases {
		t.Run(tc.intent, func(t *testing.T) {
			got := buildAvoid(tc.intent, tc.sem, tc.syms)
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
	if out.Graph == nil {
		t.Error("graph field should always be present (even if empty)")
	}
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
	if !strings.Contains(out.Avoid, "Do not grep") {
		t.Errorf("avoid should mention not grepping: %q", out.Avoid)
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
	if !strings.Contains(out.Avoid, "graph extraction is not yet wired") {
		t.Errorf("avoid should flag graph-deferred status: %q", out.Avoid)
	}
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
