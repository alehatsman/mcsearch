// Package mcp — enrichment legs for mcsearch_context.
//
// enrich.go holds the secondary lanes that the router calls *after*
// the semantic + symbol lanes have produced the primary bundle. They
// are intentionally static: filesystem walks, bounded subprocess calls
// to `git` and `rg`, and a few regex scans. No embeddings, no LLM.
//
// Gating matrix (driven by intent):
//
//	leg                | always | callers/callees | editing_context | architecture / package_topology
//	───────────────────┼────────┼─────────────────┼─────────────────┼─────────────────────────────────
//	signatures + docs  |   ✓    |                 |                 |
//	tests pairing      |   ✓    |                 |                 |
//	nearest doc        |   ✓    |                 |                 |
//	references         |        |        ✓        |                 |
//	git blame          |        |                 |        ✓        |
//	CODEOWNERS         |        |                 |        ✓        |
//	build tags / pkg   |        |                 |        ✓        |              ✓
//
// All legs are best-effort: any failure (missing git binary, no
// CODEOWNERS file, unreadable source) leaves the relevant field empty
// and does not propagate an error to the caller.
package mcp

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// ─── budgets ─────────────────────────────────────────────────────────────

const (
	maxDocLines = 10
	maxDocBytes = 600
	// References caps scale with the request's `k` so widening the
	// per-lane hit count widens references too. defaultRefHits /
	// defaultRefsPerSymbol act as floors (k=0 baseline), maxRefHits /
	// maxRefsPerSymbol act as ceilings to bound rg work.
	defaultRefHits       = 30
	defaultRefsPerSymbol = 20
	maxRefHits           = 100
	maxRefsPerSymbol     = 60
	blameTimeout         = 600 * time.Millisecond
	rgTimeout            = 2 * time.Second
)

// refCapsFor returns (perSymbol, total) reference caps for the given
// request k. Floors at the original defaults so a caller passing the
// default k=8 sees no behavior change; ceilings keep rg work bounded
// for the maximum k=30.
func refCapsFor(k int) (perSym, total int) {
	perSym = clampInt(k*3, defaultRefsPerSymbol, maxRefsPerSymbol)
	total = clampInt(k*4, defaultRefHits, maxRefHits)
	return
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// ─── leg 1: signature + doc extraction on SymbolHit ──────────────────────

// enrichSymbolsSigDoc fills Signature and Doc on each SymbolHit in
// place. Each symbol costs one bounded file read. Symbols with empty
// Path or StartLine=0 are skipped silently.
func enrichSymbolsSigDoc(projectRoot string, syms []SymbolHit) {
	for i := range syms {
		if syms[i].Path == "" || syms[i].StartLine <= 0 {
			continue
		}
		abs := syms[i].Path
		if !filepath.IsAbs(abs) {
			abs = filepath.Join(projectRoot, abs)
		}
		sig, doc := readSignatureAndDoc(abs, syms[i].StartLine)
		syms[i].Signature = sig
		syms[i].Doc = doc
	}
}

// readSignatureAndDoc reads the file once and returns:
//   - signature: the declaration line at-or-just-after startLine, trimmed
//   - doc: contiguous //-prefix (Go) or #-prefix (Python/shell) lines
//     attached to the declaration, joined with newlines, capped
//
// The chunker stores startLine pointing at the first line of the chunk,
// which for Go funcs+methods is the first doc-comment line, NOT the
// `func` line. So we scan forward from startLine through contiguous
// blanks/comments and treat the first non-comment line as the
// declaration anchor. The forward-walked comments become the doc; if
// there are none (chunk start lands on the decl itself), we fall back
// to comments above startLine.
//
// Both fields come back empty when no declaration is found within a
// small forward window. That's the staleness guard — when the index
// drifts, the recorded offset rarely lands near a real decl, so
// suppressing the fields is better than emitting nonsense.
func readSignatureAndDoc(path string, startLine int) (string, string) {
	f, err := os.Open(path)
	if err != nil {
		return "", ""
	}
	defer func() { _ = f.Close() }()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)

	// `above` buffers up to maxDocLines lines preceding startLine for
	// the doc-fallback when the chunk starts on the decl itself.
	// `forward` buffers up to maxDocLines+1 lines starting at startLine
	// so we can walk past leading comments to find the decl.
	above := make([]string, 0, maxDocLines)
	forward := make([]string, 0, maxDocLines+1)
	lineNum := 0
	for sc.Scan() {
		lineNum++
		if lineNum < startLine {
			line := sc.Text()
			if len(above) == maxDocLines {
				above = above[1:]
			}
			above = append(above, line)
			continue
		}
		forward = append(forward, sc.Text())
		if len(forward) > maxDocLines {
			break
		}
	}
	if err := sc.Err(); err != nil {
		return "", ""
	}
	if len(forward) == 0 {
		return "", ""
	}

	// Walk forward through blanks/comments to find the declaration.
	declIdx := -1
	for i, line := range forward {
		t := strings.TrimSpace(line)
		if t == "" || isCommentLine(t) {
			continue
		}
		if looksLikeDeclaration(t) {
			declIdx = i
		}
		break
	}
	if declIdx < 0 {
		// Staleness guard: nothing decl-shaped in the forward window.
		return "", ""
	}

	sig := strings.TrimSpace(forward[declIdx])

	// Doc precedence: if the forward walk passed through comment lines,
	// those ARE the doc (the chunk starts on the doc comment). Otherwise
	// fall back to comments immediately above startLine.
	var docLines []string
	if declIdx > 0 {
		for _, line := range forward[:declIdx] {
			t := strings.TrimSpace(line)
			if isCommentLine(t) {
				docLines = append(docLines, t)
			}
		}
	} else {
		for i := len(above) - 1; i >= 0; i-- {
			t := strings.TrimSpace(above[i])
			if !isCommentLine(t) {
				break
			}
			docLines = append([]string{t}, docLines...)
		}
	}
	doc := strings.Join(docLines, "\n")
	if len(doc) > maxDocBytes {
		doc = doc[:maxDocBytes] + "…"
	}
	return sig, doc
}

func isCommentLine(s string) bool {
	return strings.HasPrefix(s, "//") || strings.HasPrefix(s, "#")
}

// declarationKeywords are the first tokens that mark a declaration
// line in the languages mcsearch's chunker indexes (Go, Python, JS/TS,
// Rust, plus a few common visibility modifiers). The list is
// deliberately conservative — false negatives are cheap (empty
// signature/doc) but false positives let stale-index noise leak
// through.
var declarationKeywords = map[string]struct{}{
	"func": {}, "type": {}, "const": {}, "var": {}, "package": {},
	"def": {}, "class": {}, "async": {},
	"function": {}, "interface": {}, "let": {}, "export": {},
	"fn": {}, "struct": {}, "enum": {}, "trait": {}, "impl": {}, "pub": {},
}

// looksLikeDeclaration returns true when the first whitespace-
// delimited token of `line` is a known declaration keyword. The
// chunker stores StartLine pointing at the declaration line itself,
// so this is the right anchor — once the index drifts, the line at
// that offset rarely starts with a declaration keyword and we want to
// drop the field rather than emit a misleading signature.
func looksLikeDeclaration(line string) bool {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		return false
	}
	// First whitespace-separated token.
	first := trimmed
	if i := strings.IndexAny(trimmed, " \t("); i > 0 {
		first = trimmed[:i]
	}
	_, ok := declarationKeywords[first]
	return ok
}

// ─── leg 2: tests pairing (path heuristic, always-on) ────────────────────

// pairSiblingTests returns relative paths of test files that look like
// siblings of the input path, using language-conventional naming. It
// never recurses and never opens files beyond os.Stat to confirm
// existence. Returns paths relative to projectRoot when possible so
// they match the format used elsewhere in the bundle.
func pairSiblingTests(projectRoot, relPath string) []string {
	if relPath == "" {
		return nil
	}
	dir := filepath.Dir(relPath)
	base := filepath.Base(relPath)
	ext := filepath.Ext(base)
	stem := strings.TrimSuffix(base, ext)

	var candidates []string
	switch ext {
	case ".go":
		// foo.go → foo_test.go. Skip if input already _test.go.
		if strings.HasSuffix(stem, "_test") {
			return nil
		}
		candidates = []string{stem + "_test.go"}
	case ".py":
		// foo.py → test_foo.py or foo_test.py. Skip if already a test.
		if strings.HasPrefix(stem, "test_") || strings.HasSuffix(stem, "_test") {
			return nil
		}
		candidates = []string{"test_" + stem + ".py", stem + "_test.py"}
	case ".ts", ".tsx", ".js", ".jsx":
		// foo.ts → foo.test.ts, foo.spec.ts. Skip if already a test.
		if strings.Contains(stem, ".test") || strings.Contains(stem, ".spec") {
			return nil
		}
		candidates = []string{
			stem + ".test" + ext,
			stem + ".spec" + ext,
		}
	default:
		return nil
	}

	var out []string
	for _, c := range candidates {
		rel := filepath.Join(dir, c)
		abs := filepath.Join(projectRoot, rel)
		if _, err := os.Stat(abs); err == nil {
			out = append(out, rel)
		}
	}
	return out
}

// ─── leg 3: nearest doc walk (always-on) ─────────────────────────────────

// nearestDocFiles lists the docs we look for, in priority order. The
// first one found while walking up wins.
var nearestDocFiles = []string{"CLAUDE.md", "doc.go", "README.md"}

// findNearestDoc walks up from filepath.Dir(relPath) toward
// projectRoot, returning the first doc file it finds. Returns "" if
// none. Cap on traversal: stops at projectRoot or at depth 10 (defends
// against pathological project layouts).
func findNearestDoc(projectRoot, relPath string) string {
	if relPath == "" {
		return ""
	}
	dir := filepath.Dir(relPath)
	for range 10 {
		for _, name := range nearestDocFiles {
			candidate := filepath.Join(dir, name)
			abs := filepath.Join(projectRoot, candidate)
			if _, err := os.Stat(abs); err == nil {
				// Don't return relPath itself — if the suggested read IS
				// the README, skipping it as a sibling doc is correct.
				if filepath.Clean(candidate) == filepath.Clean(relPath) {
					continue
				}
				return candidate
			}
		}
		if dir == "." || dir == "" || dir == "/" {
			return ""
		}
		dir = filepath.Dir(dir)
	}
	return ""
}

// ─── leg 4: ripgrep references (callers/callees only) ────────────────────

// runReferencesLane shells out to ripgrep for each symbol's bare name
// and returns deduplicated RefHits. The definition line (filtered by
// matching the SymbolHit's StartLine) is excluded so the list is
// genuinely "uses of" rather than "appearances of". Caps scale with
// `k` via refCapsFor — defaults match the legacy 30/20 budget.
//
// If `rg` isn't on PATH or all invocations fail, returns nil — the
// caller still has the deferred-graph `avoid` line to fall back on.
func runReferencesLane(ctx context.Context, projectRoot string, k int, symbols []SymbolHit) []RefHit {
	if _, err := exec.LookPath("rg"); err != nil {
		return nil
	}

	perSymCap, totalCap := refCapsFor(k)
	seen := map[string]struct{}{} // path:line dedupe
	var out []RefHit

	for _, sym := range symbols {
		if len(out) >= totalCap {
			break
		}
		bare := bareSymbolName(sym.QualifiedName)
		if bare == "" {
			continue
		}
		hits := ripgrepSymbol(ctx, projectRoot, bare, perSymCap, sym, seen)
		// Per-symbol cap before moving to the next, so a hot symbol
		// can't starve the others.
		if len(hits) > perSymCap {
			hits = hits[:perSymCap]
		}
		for _, h := range hits {
			if len(out) >= totalCap {
				break
			}
			out = append(out, h)
		}
	}
	return out
}

// bareSymbolName strips a "(*T)." or "T." prefix and returns the
// rightmost identifier — the form rg will actually find in usages.
func bareSymbolName(qualified string) string {
	if i := strings.LastIndex(qualified, "."); i >= 0 {
		return qualified[i+1:]
	}
	return qualified
}

// ripgrepSymbol runs `rg -nw --max-count=<...> -e <symbol> <root>` and
// parses its `path:line:text` output into RefHits, skipping the
// definition line. Single subprocess, bounded by rgTimeout. perSymCap
// caps per-file matches via rg's --max-count.
func ripgrepSymbol(ctx context.Context, projectRoot, symbol string, perSymCap int, defSym SymbolHit, seen map[string]struct{}) []RefHit {
	cctx, cancel := context.WithTimeout(ctx, rgTimeout)
	defer cancel()

	// --word-regexp: don't match SearchTerms when looking for Search.
	// --max-count: cap per-file matches; rg's default is unlimited.
	// --no-heading, --color=never: parseable single-line output.
	cmd := exec.CommandContext(cctx, "rg",
		"--word-regexp",
		"--max-count="+fmt.Sprint(perSymCap),
		"--no-heading",
		"--color=never",
		"--line-number",
		"-e", symbol,
		projectRoot,
	)
	stdout, err := cmd.Output()
	if err != nil {
		// rg exits 1 when nothing matches — treat as empty, not error.
		return nil
	}

	var out []RefHit
	defAbs := filepath.Join(projectRoot, defSym.Path)
	sc := bufio.NewScanner(strings.NewReader(string(stdout)))
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	for sc.Scan() {
		path, line, snippet, ok := parseRipgrepLine(sc.Text())
		if !ok {
			continue
		}
		// Skip the definition itself: same file + within def line range.
		if defSym.Path != "" && (path == defAbs || path == defSym.Path) {
			if line >= defSym.StartLine && line <= defSym.EndLine {
				continue
			}
		}
		// Normalize to project-relative path for consistency with the
		// rest of the bundle.
		rel, err := filepath.Rel(projectRoot, path)
		if err == nil && !strings.HasPrefix(rel, "..") {
			path = rel
		}
		key := path + ":" + fmt.Sprint(line)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, RefHit{
			Path:    path,
			Line:    line,
			Snippet: strings.TrimSpace(snippet),
			Symbol:  defSym.QualifiedName,
		})
	}
	return out
}

// parseRipgrepLine splits `path:line:text` — paths on Windows can
// contain `:` after the drive letter, but we run on Linux/macOS where
// the first two `:` are always separators.
func parseRipgrepLine(s string) (path string, line int, text string, ok bool) {
	p, rest, found := strings.Cut(s, ":")
	if !found {
		return "", 0, "", false
	}
	lineStr, snippet, found := strings.Cut(rest, ":")
	if !found {
		return "", 0, "", false
	}
	if _, err := fmt.Sscanf(lineStr, "%d", &line); err != nil {
		return "", 0, "", false
	}
	return p, line, snippet, true
}

// ─── leg 5: git blame + CODEOWNERS (editing_context only) ────────────────

// enrichBlame populates LastCommit / LastAuthor on the meta for each
// path. One `git log -1` subprocess per path, bounded by blameTimeout
// individually. If `git` isn't available, returns silently.
func enrichBlame(ctx context.Context, projectRoot string, paths []string, meta map[string]*PathMeta) {
	if _, err := exec.LookPath("git"); err != nil {
		return
	}
	for _, p := range paths {
		cctx, cancel := context.WithTimeout(ctx, blameTimeout)
		// %h|%ad|%an with date=short keeps the field compact.
		cmd := exec.CommandContext(cctx, "git",
			"-C", projectRoot,
			"log", "-1",
			"--format=%h|%ad|%an",
			"--date=short",
			"--", p,
		)
		out, err := cmd.Output()
		cancel()
		if err != nil {
			continue
		}
		line := strings.TrimSpace(string(out))
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, "|", 3)
		if len(parts) < 3 {
			continue
		}
		m := getOrInit(meta, p)
		m.LastCommit = parts[0] + " " + parts[1]
		m.LastAuthor = parts[2]
	}
}

// codeownersPath returns the first CODEOWNERS file that exists in the
// standard locations, or "".
func codeownersPath(projectRoot string) string {
	for _, rel := range []string{"CODEOWNERS", ".github/CODEOWNERS", "docs/CODEOWNERS"} {
		abs := filepath.Join(projectRoot, rel)
		if _, err := os.Stat(abs); err == nil {
			return abs
		}
	}
	return ""
}

// codeownersRule is one parsed CODEOWNERS entry. Patterns are matched
// in the order they appear in the file; the LAST match wins (per
// GitHub's CODEOWNERS semantics).
type codeownersRule struct {
	pattern string
	owners  []string
}

// loadCodeowners parses the CODEOWNERS file. Returns nil if missing.
func loadCodeowners(projectRoot string) []codeownersRule {
	abs := codeownersPath(projectRoot)
	if abs == "" {
		return nil
	}
	f, err := os.Open(abs)
	if err != nil {
		return nil
	}
	defer func() { _ = f.Close() }()
	var rules []codeownersRule
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		rules = append(rules, codeownersRule{pattern: fields[0], owners: fields[1:]})
	}
	return rules
}

// matchOwners returns owners for path by walking rules in order and
// keeping the last match (GitHub semantics). Handles the common
// CODEOWNERS patterns: `*` matches every file, `*.ext` matches any
// file with that extension at any depth (applied to basename),
// `dir/` matches files under a directory, and exact-path patterns
// fall back to filepath.Match. We don't implement full gitignore
// semantics (e.g. recursive `**`); good enough for the bundle hint.
func matchOwners(rules []codeownersRule, relPath string) []string {
	var owners []string
	base := filepath.Base(relPath)
	for _, r := range rules {
		pat := strings.TrimPrefix(r.pattern, "/")
		switch {
		case pat == "*":
			owners = r.owners
		case strings.HasSuffix(pat, "/") && strings.HasPrefix(relPath, pat):
			owners = r.owners
		case !strings.Contains(pat, "/"):
			// basename glob like `*.go` or `CODEOWNERS`.
			if matched, _ := filepath.Match(pat, base); matched {
				owners = r.owners
			}
		default:
			if matched, _ := filepath.Match(pat, relPath); matched {
				owners = r.owners
			}
		}
	}
	return owners
}

// enrichOwners fills Owners on meta from a parsed CODEOWNERS file.
func enrichOwners(projectRoot string, paths []string, meta map[string]*PathMeta) {
	rules := loadCodeowners(projectRoot)
	if rules == nil {
		return
	}
	for _, p := range paths {
		owners := matchOwners(rules, p)
		if len(owners) == 0 {
			continue
		}
		m := getOrInit(meta, p)
		m.Owners = owners
	}
}

// ─── leg 6: build tags + package clause (Go files only) ──────────────────

// enrichBuildTags scans the first ~20 lines of each Go file for a
// //go:build (or legacy // +build) line and the `package` clause.
// No-op for non-.go paths.
func enrichBuildTags(projectRoot string, paths []string, meta map[string]*PathMeta) {
	for _, p := range paths {
		if filepath.Ext(p) != ".go" {
			continue
		}
		abs := p
		if !filepath.IsAbs(abs) {
			abs = filepath.Join(projectRoot, p)
		}
		tags, pkg := readBuildTagsAndPackage(abs)
		if tags == "" && pkg == "" {
			continue
		}
		m := getOrInit(meta, p)
		m.BuildTags = tags
		m.Package = pkg
	}
}

func readBuildTagsAndPackage(path string) (tags, pkg string) {
	f, err := os.Open(path)
	if err != nil {
		return "", ""
	}
	defer func() { _ = f.Close() }()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	for n := 0; n < 20 && sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		switch {
		case strings.HasPrefix(line, "//go:build "):
			tags = line
		case strings.HasPrefix(line, "// +build ") && tags == "":
			tags = line
		case strings.HasPrefix(line, "package "):
			pkg = strings.TrimPrefix(line, "package ")
			// package clause is the last thing we care about.
			return tags, pkg
		}
	}
	return tags, pkg
}

// ─── orchestration ───────────────────────────────────────────────────────

func getOrInit(m map[string]*PathMeta, key string) *PathMeta {
	if pm, ok := m[key]; ok {
		return pm
	}
	pm := &PathMeta{}
	m[key] = pm
	return pm
}

// uniquePaths gathers the deduplicated path set from suggested_reads
// and symbol hits. Order matches first appearance, with suggested
// reads coming first since they're the primary surface.
func uniquePaths(reads []SuggestedRead, syms []SymbolHit) []string {
	seen := map[string]struct{}{}
	var out []string
	add := func(p string) {
		if p == "" {
			return
		}
		if _, ok := seen[p]; ok {
			return
		}
		seen[p] = struct{}{}
		out = append(out, p)
	}
	for _, r := range reads {
		add(r.Path)
	}
	for _, s := range syms {
		add(s.Path)
	}
	return out
}

// enrich applies every leg appropriate for the given intent to the
// output bundle in place. Each leg is best-effort; failures leave
// fields empty rather than propagating errors. `k` is the request's
// per-lane cap; passed through to the references lane so wider
// requests get proportionally wider reference lists.
func enrich(ctx context.Context, projectRoot, intent string, k int, out *ContextOutput) {
	// Symbol-level enrichment is always on when we have symbol hits.
	if len(out.Symbols) > 0 {
		enrichSymbolsSigDoc(projectRoot, out.Symbols)
	}

	paths := uniquePaths(out.SuggestedReads, out.Symbols)
	if len(paths) == 0 && (intent != IntentCallers && intent != IntentCallees) {
		return
	}

	meta := map[string]*PathMeta{}

	// Always-on path heuristics.
	for _, p := range paths {
		tests := pairSiblingTests(projectRoot, p)
		nearest := findNearestDoc(projectRoot, p)
		if len(tests) == 0 && nearest == "" {
			continue
		}
		m := getOrInit(meta, p)
		m.Tests = tests
		m.NearestDoc = nearest
	}

	// editing_context: blame + owners.
	if intent == IntentEditingContext {
		enrichBlame(ctx, projectRoot, paths, meta)
		enrichOwners(projectRoot, paths, meta)
	}

	// editing_context, architecture, package_topology: build tags + pkg.
	if intent == IntentEditingContext || intent == IntentArchitecture || intent == IntentPackageTopology {
		enrichBuildTags(projectRoot, paths, meta)
	}

	if len(meta) > 0 {
		out.Annotations = make(map[string]PathMeta, len(meta))
		for k, v := range meta {
			out.Annotations[k] = *v
		}
	}

	// References: callers/callees with at least one symbol hit.
	if (intent == IntentCallers || intent == IntentCallees) && len(out.Symbols) > 0 {
		out.References = runReferencesLane(ctx, projectRoot, k, out.Symbols)
	}
}
