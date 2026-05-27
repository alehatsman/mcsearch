package guide

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/alehatsman/dex/internal/chunk"
	"github.com/alehatsman/dex/internal/store"
)

// ─── pure helpers ─────────────────────────────────────────────────────

func TestSplitImports(t *testing.T) {
	cases := []struct {
		name           string
		imports        []string
		modPath        string
		wantProject    []string
		wantExternal   []string
	}{
		{
			name:         "no modpath puts everything external",
			imports:      []string{"context", "github.com/foo/bar"},
			modPath:      "",
			wantProject:  nil,
			wantExternal: []string{"context", "github.com/foo/bar"},
		},
		{
			name:         "modpath splits project vs external",
			imports:      []string{"context", "github.com/a/b/internal/x", "github.com/a/b/internal/y", "github.com/other/lib"},
			modPath:      "github.com/a/b",
			wantProject:  []string{"internal/x", "internal/y"},
			wantExternal: []string{"context", "github.com/other/lib"},
		},
		{
			name:         "exact module match is not a project import (no slash after prefix)",
			imports:      []string{"github.com/a/b"},
			modPath:      "github.com/a/b",
			wantProject:  nil,
			wantExternal: []string{"github.com/a/b"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotP, gotE := splitImports(tc.imports, tc.modPath)
			if !equalSlice(gotP, tc.wantProject) {
				t.Errorf("project: got %v, want %v", gotP, tc.wantProject)
			}
			if !equalSlice(gotE, tc.wantExternal) {
				t.Errorf("external: got %v, want %v", gotE, tc.wantExternal)
			}
		})
	}
}

func TestReadModulePath(t *testing.T) {
	dir := t.TempDir()

	// Missing go.mod → empty string.
	if got := readModulePath(dir); got != "" {
		t.Errorf("missing go.mod: got %q, want \"\"", got)
	}

	// Valid go.mod.
	mustWrite(t, filepath.Join(dir, "go.mod"), "module github.com/example/proj\n\ngo 1.26\n")
	if got := readModulePath(dir); got != "github.com/example/proj" {
		t.Errorf("valid: got %q, want module path", got)
	}

	// Leading whitespace before `module` line.
	mustWrite(t, filepath.Join(dir, "go.mod"), "// header\n   module   github.com/x/y   \n")
	if got := readModulePath(dir); got != "github.com/x/y" {
		t.Errorf("with whitespace: got %q", got)
	}
}

func TestTrimModulePrefix(t *testing.T) {
	in := []string{"github.com/a/b/internal/x", "github.com/other/lib"}

	if got := trimModulePrefix(in, ""); !equalSlice(got, in) {
		t.Errorf("empty modPath should pass through, got %v", got)
	}
	got := trimModulePrefix(in, "github.com/a/b")
	want := []string{"internal/x", "github.com/other/lib"}
	if !equalSlice(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestIsFixtureDir(t *testing.T) {
	cases := []struct {
		path string
		want bool
	}{
		{"testdata", true},
		{"testdata/foo", true},
		{"internal/graph/testdata", true},
		{"internal/graph/testdata/mooncake", true},
		{"internal/graph/testdata/mooncake/shared", true},
		{"internal/graph", false},
		{"internal/mytestdata", false}, // not a segment boundary
		{"testdataish/x", false},
		{".", false},
		{"", false},
	}
	for _, tc := range cases {
		if got := isFixtureDir(tc.path); got != tc.want {
			t.Errorf("isFixtureDir(%q) = %v, want %v", tc.path, got, tc.want)
		}
	}
}

func TestFilterFixtureDirs(t *testing.T) {
	in := []store.SummaryRow{
		{Path: "cmd/dex"},
		{Path: "internal/graph"},
		{Path: "internal/graph/testdata/simple"},
		{Path: "internal/graph/testdata/simple/store"},
		{Path: "internal/store"},
		{Path: "testdata"},
	}
	got := filterFixtureDirs(in)
	want := []string{"cmd/dex", "internal/graph", "internal/store"}
	if len(got) != len(want) {
		t.Fatalf("got %d rows, want %d: %+v", len(got), len(want), got)
	}
	for i, p := range want {
		if got[i].Path != p {
			t.Errorf("[%d] got %q, want %q", i, got[i].Path, p)
		}
	}
}

func TestDisplayName(t *testing.T) {
	cases := []struct {
		name string
		sym  store.GraphSymbol
		want string
	}{
		{
			name: "function uses Name",
			sym:  store.GraphSymbol{Name: "Open", Kind: "function", QualifiedName: "github.com/x/y.Open"},
			want: "Open",
		},
		{
			name: "method uses QualifiedName receiver+method",
			sym:  store.GraphSymbol{Name: "Search", Kind: "method", QualifiedName: "github.com/x/y/store.Store.Search"},
			want: "Store.Search",
		},
		{
			name: "method without QualifiedName falls back to Name",
			sym:  store.GraphSymbol{Name: "Close", Kind: "method", QualifiedName: ""},
			want: "Close",
		},
		{
			name: "struct uses Name (kind != method)",
			sym:  store.GraphSymbol{Name: "Hit", Kind: "struct", QualifiedName: "github.com/x/y.Hit"},
			want: "Hit",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := displayName(tc.sym); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// ─── end-to-end render ────────────────────────────────────────────────

func TestRender_NoSummaries(t *testing.T) {
	st, ctx, root := newGuideTestStore(t)

	_, err := Render(ctx, st, root, DefaultConfig(), Options{})
	if err == nil || !strings.Contains(err.Error(), "no summaries") {
		t.Fatalf("want \"no summaries\" error, got %v", err)
	}
}

func TestRender_FirstRenderWrites(t *testing.T) {
	st, ctx, root := newGuideTestStore(t)
	seedSummaries(t, st, map[string]string{
		".":                "Top-level overview of the project.",
		"internal/example": "Package example does things.",
	})

	res, err := Render(ctx, st, root, DefaultConfig(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Wrote || !res.Dirty {
		t.Fatalf("first render: Wrote=%v Dirty=%v", res.Wrote, res.Dirty)
	}
	body := mustReadFile(t, res.OutputPath)
	if !strings.Contains(body, "## Overview") {
		t.Errorf("missing Overview section")
	}
	if !strings.Contains(body, "## Module: internal/example") {
		t.Errorf("missing module section")
	}
	if !strings.Contains(body, "Top-level overview") {
		t.Errorf("repo summary content missing")
	}
}

func TestRender_IncrementalNoOp(t *testing.T) {
	st, ctx, root := newGuideTestStore(t)
	seedSummaries(t, st, map[string]string{
		".":                "Overview.",
		"internal/example": "Pkg example.",
	})

	_, err := Render(ctx, st, root, DefaultConfig(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	res, err := Render(ctx, st, root, DefaultConfig(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Wrote || res.Dirty {
		t.Errorf("second run should be no-op: Wrote=%v Dirty=%v", res.Wrote, res.Dirty)
	}
}

func TestRender_DryRunDoesNotWrite(t *testing.T) {
	st, ctx, root := newGuideTestStore(t)
	seedSummaries(t, st, map[string]string{".": "Overview."})

	res, err := Render(ctx, st, root, DefaultConfig(), Options{DryRun: true})
	if err != nil {
		t.Fatal(err)
	}
	if res.Wrote {
		t.Errorf("dry run wrote the file")
	}
	if !res.Dirty {
		t.Errorf("dry run should still report Dirty=true")
	}
	if _, err := os.Stat(res.OutputPath); !os.IsNotExist(err) {
		t.Errorf("dry run created %s", res.OutputPath)
	}
}

func TestRender_ForceRerendersClean(t *testing.T) {
	st, ctx, root := newGuideTestStore(t)
	seedSummaries(t, st, map[string]string{".": "Overview."})

	if _, err := Render(ctx, st, root, DefaultConfig(), Options{}); err != nil {
		t.Fatal(err)
	}
	res, err := Render(ctx, st, root, DefaultConfig(), Options{Force: true})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Wrote || !res.Dirty {
		t.Errorf("--force should re-render: Wrote=%v Dirty=%v", res.Wrote, res.Dirty)
	}
}

func TestRender_NonGoModuleHasNoGraphSections(t *testing.T) {
	st, ctx, root := newGuideTestStore(t)
	// scripts/ is non-Go: no graph_nodes seeded, so render should
	// emit only the LLM prose for it — no Exported API / Used by / etc.
	seedSummaries(t, st, map[string]string{
		".":       "Overview.",
		"scripts": "Bash automation scripts.",
	})

	res, err := Render(ctx, st, root, DefaultConfig(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	body := mustReadFile(t, res.OutputPath)
	if !strings.Contains(body, "## Module: scripts") {
		t.Errorf("scripts module section missing")
	}
	if strings.Contains(body, "### Exported API") {
		t.Errorf("scripts should not have Exported API section without graph data")
	}
}

func TestRender_GraphSectionsAppearWhenSeeded(t *testing.T) {
	st, ctx, root := newGuideTestStore(t)
	mustWrite(t, filepath.Join(root, "go.mod"), "module example.com/proj\n")
	seedSummaries(t, st, map[string]string{
		".":                "Overview.",
		"internal/example": "Pkg example.",
	})
	seedGraphPackage(t, st, "example.com/proj/internal/example",
		[]graphNode{
			{name: "DoThing", kind: "function", file: "internal/example/x.go", line: 10, pagerank: 0.5, inDeg: 3},
			{name: "internalThing", kind: "function", file: "internal/example/x.go", line: 30, pagerank: 0.2, inDeg: 1},
			{name: "Helper", kind: "struct", file: "internal/example/x.go", line: 5, pagerank: 0.0, inDeg: 0},
		},
		[]string{"context", "fmt"},
	)

	res, err := Render(ctx, st, root, DefaultConfig(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	body := mustReadFile(t, res.OutputPath)
	if !strings.Contains(body, "### Exported API") {
		t.Errorf("Exported API section missing\n%s", body)
	}
	if !strings.Contains(body, "DoThing") || !strings.Contains(body, "Helper") {
		t.Errorf("exported symbols missing\n%s", body)
	}
	if strings.Contains(body, "internalThing") {
		t.Errorf("unexported symbol should not appear in Exported API")
	}
	if !strings.Contains(body, "### Key entry points") {
		t.Errorf("Key entry points section missing (exported function exists)\n%s", body)
	}
	if !strings.Contains(body, "### Depends on") {
		t.Errorf("Depends on section missing\n%s", body)
	}
	if !strings.Contains(body, "external: context, fmt") {
		t.Errorf("external imports list malformed\n%s", body)
	}
}

func TestRender_InternalFallbackHeading(t *testing.T) {
	// Package with no exported nodes — central section should switch
	// to the "internal hot spots" heading rather than silently empty.
	st, ctx, root := newGuideTestStore(t)
	mustWrite(t, filepath.Join(root, "go.mod"), "module example.com/proj\n")
	seedSummaries(t, st, map[string]string{
		".":                "Overview.",
		"internal/helpers": "All-private helpers.",
	})
	seedGraphPackage(t, st, "example.com/proj/internal/helpers",
		[]graphNode{
			{name: "doIt", kind: "function", file: "internal/helpers/h.go", line: 1, pagerank: 0.4, inDeg: 5},
		},
		nil,
	)

	res, err := Render(ctx, st, root, DefaultConfig(), Options{})
	if err != nil {
		t.Fatal(err)
	}
	body := mustReadFile(t, res.OutputPath)
	if !strings.Contains(body, "### Key internal hot spots") {
		t.Errorf("expected 'Key internal hot spots' fallback heading, body:\n%s", body)
	}
}

// ─── test helpers ─────────────────────────────────────────────────────

func newGuideTestStore(t *testing.T) (*store.Store, context.Context, string) {
	t.Helper()
	root := t.TempDir()
	dbPath := filepath.Join(t.TempDir(), "index.db")
	ctx := context.Background()
	st, err := store.Open(ctx, dbPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = st.Close() })
	return st, ctx, root
}

// seedSummaries inserts repo_summary (path=".") and package_summary
// rows. content for path "." is treated as repo_summary; every other
// path is a package_summary.
func seedSummaries(t *testing.T, st *store.Store, summaries map[string]string) {
	t.Helper()
	now := time.Now()
	rows := make([]store.PendingChunk, 0, len(summaries))
	for path, content := range summaries {
		kind := chunk.KindPackageSummary
		if path == "." {
			kind = chunk.KindRepoSummary
		}
		rows = append(rows, store.PendingChunk{
			Path:       path,
			Kind:       kind,
			ContentSHA: "sha-" + path,
			Content:    content,
			Vec:        []float32{1, 0, 0, 0},
		})
	}
	if err := st.UpsertMany(context.Background(), rows, now); err != nil {
		t.Fatal(err)
	}
}

type graphNode struct {
	name     string
	kind     string
	file     string
	line     int
	pagerank float64
	inDeg    int
}

// seedGraphPackage inserts declaration nodes (functions/methods/etc.)
// for a package and import nodes for each entry in imports.
// declarations get file_path set; imports get only package_path set
// (mirrors how the real graph extractor stores them).
func seedGraphPackage(t *testing.T, st *store.Store, pkgPath string, decls []graphNode, imports []string) {
	t.Helper()
	now := time.Now()
	rows := make([]store.GraphNodeRow, 0, len(decls)+len(imports))
	for i, d := range decls {
		rows = append(rows, store.GraphNodeRow{
			ID:            pkgPath + "::decl::" + d.name,
			Kind:          d.kind,
			Name:          d.name,
			QualifiedName: pkgPath + "." + d.name,
			PackagePath:   pkgPath,
			FilePath:      d.file,
			StartLine:     d.line,
			EndLine:       d.line + 1,
			ContentHash:   "h" + itoa(i),
			InDegree:      d.inDeg,
			PageRank:      d.pagerank,
		})
	}
	for i, imp := range imports {
		rows = append(rows, store.GraphNodeRow{
			ID:          pkgPath + "::import::" + imp,
			Kind:        "import",
			Name:        imp,
			PackagePath: pkgPath,
			ContentHash: "i" + itoa(i),
		})
	}
	if err := st.GraphUpsertNodes(context.Background(), rows, now); err != nil {
		t.Fatal(err)
	}
}

func mustWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func mustReadFile(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func equalSlice(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// itoa avoids strconv import bloat in this test file.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	neg := n < 0
	if neg {
		n = -n
	}
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
