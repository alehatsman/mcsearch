// Package chunk splits a source file into retrieval-sized chunks.
//
// For languages with a tree-sitter grammar we extract top-level
// declarations (functions, methods, classes, types). For unknown
// languages (or when tree-sitter fails to parse), we fall back to a
// fixed-line sliding window with overlap.
//
// Chunks are capped at MaxBytes. Anything larger is split into
// MaxBytes-bounded line slices.
package chunk

import (
	"bytes"
	"context"
	"path/filepath"
	"sort"
	"strings"

	sitter "github.com/smacker/go-tree-sitter"
	"github.com/smacker/go-tree-sitter/bash"
	"github.com/smacker/go-tree-sitter/c"
	"github.com/smacker/go-tree-sitter/cpp"
	"github.com/smacker/go-tree-sitter/golang"
	"github.com/smacker/go-tree-sitter/java"
	"github.com/smacker/go-tree-sitter/javascript"
	"github.com/smacker/go-tree-sitter/lua"
	"github.com/smacker/go-tree-sitter/python"
	"github.com/smacker/go-tree-sitter/ruby"
	"github.com/smacker/go-tree-sitter/rust"
	"github.com/smacker/go-tree-sitter/typescript/typescript"
)

// MaxBytes is the upper bound on a single chunk's content (excluding
// the path/kind prefix added at embed time). Roughly 1024 tokens
// assuming the typical 4 chars/token ratio.
const MaxBytes = 4096

// WindowLines is the line-window fallback size.
const WindowLines = 40

// WindowOverlap lines repeat between consecutive line windows.
const WindowOverlap = 10

// Kind values for Chunk.Kind.
const (
	KindWindow         = "window"
	KindOrphan         = "orphan"
	KindFileSummary    = "file_summary"
	KindChunkSummary   = "chunk_summary"
	KindPackageSummary = "package_summary"
)

// LineCount returns the number of lines in data. A trailing newline is
// treated as a line terminator, not the start of an empty line, so a
// typical POSIX file ending in '\n' reports the same count as an editor
// would show.
func LineCount(data []byte) int {
	if len(data) == 0 {
		return 0
	}
	n := 1
	for _, b := range data {
		if b == '\n' {
			n++
		}
	}
	if data[len(data)-1] == '\n' {
		n--
	}
	return n
}

// Chunk is one retrievable slice of one file.
type Chunk struct {
	Path      string // relative to project root
	Kind      string // tree-sitter node kind or "window"
	Name      string // primary identifier (function/method/type name); empty for windows/orphans
	Parent    string // enclosing class/struct/impl name; empty for top-level chunks
	StartLine int    // 1-based, inclusive
	EndLine   int    // 1-based, inclusive
	Content   string

	// startByte/endByte mark the byte range this chunk covers in the
	// original source. Used internally to compute orphan windows (the
	// gaps between structural chunks); not persisted to the store.
	startByte int
	endByte   int
}

// langConfig pairs a tree-sitter language with the node kinds we want
// to surface as chunk roots.
type langConfig struct {
	lang  *sitter.Language
	kinds map[string]bool
}

var languages = map[string]langConfig{
	".go": {golang.GetLanguage(), set(
		"function_declaration", "method_declaration", "type_declaration",
	)},
	".py": {python.GetLanguage(), set(
		"function_definition", "class_definition", "decorated_definition",
	)},
	".js":  {javascript.GetLanguage(), jsKinds()},
	".jsx": {javascript.GetLanguage(), jsKinds()},
	".ts":  {typescript.GetLanguage(), tsKinds()},
	".tsx": {typescript.GetLanguage(), tsKinds()},
	".rs": {rust.GetLanguage(), set(
		"function_item", "struct_item", "enum_item", "impl_item",
		"trait_item", "mod_item",
	)},
	".java": {java.GetLanguage(), set(
		"method_declaration", "class_declaration", "interface_declaration",
		"enum_declaration",
	)},
	".c": {c.GetLanguage(), set(
		"function_definition", "struct_specifier",
	)},
	".h": {c.GetLanguage(), set(
		"function_definition", "struct_specifier", "declaration",
	)},
	".cc":  {cpp.GetLanguage(), cppKinds()},
	".cpp": {cpp.GetLanguage(), cppKinds()},
	".hpp": {cpp.GetLanguage(), cppKinds()},
	".rb": {ruby.GetLanguage(), set(
		"method", "class", "module", "singleton_method",
	)},
	".lua": {lua.GetLanguage(), set(
		"function_declaration_statement", "local_function_declaration_statement",
	)},
	".sh": {bash.GetLanguage(), set(
		"function_definition",
	)},
	".bash": {bash.GetLanguage(), set(
		"function_definition",
	)},
	".zsh": {bash.GetLanguage(), set(
		"function_definition",
	)},
}

// containerMethods maps top-level container node kinds to the method-level
// node kinds found inside them. We walk one level of body/block wrappers
// to reach the actual method nodes (e.g. Python's `block`, Java's
// `class_body`, JS's `class_body`).
var containerMethods = map[string]map[string]bool{
	"class_declaration": {
		"method_definition":  true, // JS/TS
		"method_declaration": true, // Java
	},
	"class_definition": {
		"function_definition": true, // Python
	},
	"class_specifier": {
		"function_definition": true, // C++
	},
	"impl_item": {
		"function_item": true, // Rust
	},
	"trait_item": {
		"function_item": true, // Rust
	},
	"interface_declaration": {
		"method_declaration": true, // Java / TS
	},
	"enum_declaration": {
		"method_declaration": true, // Java
	},
	"module": {
		"method":           true, // Ruby
		"singleton_method": true, // Ruby
	},
}

func set(items ...string) map[string]bool {
	m := make(map[string]bool, len(items))
	for _, k := range items {
		m[k] = true
	}
	return m
}

func jsKinds() map[string]bool {
	return set(
		"function_declaration",
		"class_declaration",
		"method_definition",
		"lexical_declaration", // top-level const/let with arrow-fn rhs
		"export_statement",
	)
}

func tsKinds() map[string]bool {
	k := jsKinds()
	k["interface_declaration"] = true
	k["type_alias_declaration"] = true
	k["enum_declaration"] = true
	return k
}

func cppKinds() map[string]bool {
	return set(
		"function_definition",
		"struct_specifier",
		"class_specifier",
		"namespace_definition",
	)
}

// Chunks splits the given source into chunks. relPath is used only to
// pick the language by extension and is stamped into each Chunk.
//
// For tree-sitter-supported languages we emit one chunk per recognized
// structural declaration AND additional "orphan" window chunks covering
// any byte range not claimed by a structural chunk — top-level
// constants, vars, imports, file headers, trailing comments. Without
// this hybrid pass, a file like `package foo; const X = 1; func F(){}`
// would only index F and silently drop X.
func Chunks(ctx context.Context, relPath string, src []byte) ([]Chunk, error) {
	ext := strings.ToLower(filepath.Ext(relPath))
	if cfg, ok := languages[ext]; ok {
		out, err := treeChunks(ctx, relPath, src, cfg)
		if err == nil && len(out) > 0 {
			out = append(out, orphanWindows(relPath, src, out)...)
			return out, nil
		}
		// tree-sitter empty or errored — fall through to line windows.
	}
	return windowChunks(relPath, src), nil
}

func treeChunks(ctx context.Context, relPath string, src []byte, cfg langConfig) ([]Chunk, error) {
	parser := sitter.NewParser()
	defer parser.Close()
	parser.SetLanguage(cfg.lang)
	tree, err := parser.ParseCtx(ctx, nil, src)
	if err != nil {
		return nil, err
	}
	defer tree.Close()
	root := tree.RootNode()
	var out []Chunk
	for i := range int(root.NamedChildCount()) {
		n := root.NamedChild(i)
		if n == nil {
			continue
		}
		kind := n.Type()
		if !cfg.kinds[kind] {
			continue
		}
		startByte := int(n.StartByte())
		endByte := int(n.EndByte())
		// Walk back to include leading line comments / docstrings.
		startByte = backfillComments(src, startByte)
		body := string(src[startByte:endByte])
		startLine := lineOf(src, startByte)
		endLine := max(lineOf(src, endByte-1), startLine)
		name := nodeIdentifier(n, src)
		if len(body) <= MaxBytes {
			out = append(out, Chunk{
				Path: relPath, Kind: kind, Name: name,
				StartLine: startLine, EndLine: endLine,
				Content:   body,
				startByte: startByte,
				endByte:   endByte,
			})
		} else {
			// Oversized declaration → fall back to line windows over its body.
			bodyLines := strings.Split(body, "\n")
			for _, w := range windowOver(bodyLines, startLine) {
				w.Path = relPath
				w.Kind = kind + ":window"
				w.Name = name
				w.startByte = startByte
				w.endByte = endByte
				out = append(out, w)
			}
		}
		// For container kinds (class, impl, trait, module), also extract
		// nested method chunks so each method gets its own index entry with
		// the parent name stamped in EmbedText for richer retrieval.
		if methodKinds, ok := containerMethods[kind]; ok {
			out = append(out, nestedChunks(relPath, src, n, methodKinds, name)...)
		}
	}
	return out, nil
}

// nestedChunks walks one node looking for method-level children and
// returns a Chunk per match. It descends one level of wrapper nodes
// (body, class_body, block) to reach the actual method nodes.
func nestedChunks(relPath string, src []byte, container *sitter.Node, methodKinds map[string]bool, parentName string) []Chunk {
	var out []Chunk
	collectMethods(relPath, src, container, methodKinds, parentName, 2, &out)
	return out
}

func collectMethods(relPath string, src []byte, n *sitter.Node, methodKinds map[string]bool, parentName string, depth int, out *[]Chunk) {
	if depth <= 0 {
		return
	}
	for i := range int(n.NamedChildCount()) {
		child := n.NamedChild(i)
		if child == nil {
			continue
		}
		if methodKinds[child.Type()] {
			if c, ok := buildNestedChunk(relPath, src, child, parentName); ok {
				*out = append(*out, c)
			}
		} else {
			collectMethods(relPath, src, child, methodKinds, parentName, depth-1, out)
		}
	}
}

func buildNestedChunk(relPath string, src []byte, n *sitter.Node, parentName string) (Chunk, bool) {
	startByte := int(n.StartByte())
	endByte := int(n.EndByte())
	startByte = backfillComments(src, startByte)
	body := string(src[startByte:endByte])
	if strings.TrimSpace(body) == "" {
		return Chunk{}, false
	}
	startLine := lineOf(src, startByte)
	endLine := max(lineOf(src, endByte-1), startLine)
	name := nodeIdentifier(n, src)
	if len(body) > MaxBytes {
		body = body[:MaxBytes]
	}
	return Chunk{
		Path: relPath, Kind: n.Type(), Name: name, Parent: parentName,
		StartLine: startLine, EndLine: endLine,
		Content:   body,
		startByte: startByte,
		endByte:   endByte,
	}, true
}

// nodeIdentifier extracts the primary identifier of a tree-sitter node by
// looking up its "name" field — the standard field name for the declared
// identifier in every tree-sitter grammar we target (Go functions, Python
// defs, JS/TS classes, Rust items, Java methods, etc.). Returns "" when the
// node has no such field (e.g. impl_item, lexical_declaration).
func nodeIdentifier(n *sitter.Node, src []byte) string {
	nameNode := n.ChildByFieldName("name")
	if nameNode == nil {
		return ""
	}
	return string(src[nameNode.StartByte():nameNode.EndByte()])
}

// orphanWindows emits window chunks over the parts of src that aren't
// covered by any structural chunk. It's the safety net that catches
// top-level non-function content (Go const/var, Rust statics, top-level
// Python statements outside def/class, etc.).
func orphanWindows(relPath string, src []byte, structural []Chunk) []Chunk {
	if len(structural) == 0 {
		return nil
	}
	// Sort the covered intervals by start byte.
	type iv struct{ s, e int }
	intervals := make([]iv, 0, len(structural))
	for _, c := range structural {
		if c.startByte == 0 && c.endByte == 0 {
			continue
		}
		intervals = append(intervals, iv{c.startByte, c.endByte})
	}
	if len(intervals) == 0 {
		return nil
	}
	// Sort by start; merge overlapping ranges.
	sort.Slice(intervals, func(i, j int) bool { return intervals[i].s < intervals[j].s })
	merged := intervals[:1]
	for _, x := range intervals[1:] {
		last := &merged[len(merged)-1]
		if x.s <= last.e {
			if x.e > last.e {
				last.e = x.e
			}
			continue
		}
		merged = append(merged, x)
	}

	var out []Chunk
	cursor := 0
	for _, m := range merged {
		if m.s > cursor {
			out = append(out, orphanRange(relPath, src, cursor, m.s)...)
		}
		cursor = m.e
	}
	if cursor < len(src) {
		out = append(out, orphanRange(relPath, src, cursor, len(src))...)
	}
	return out
}

// orphanRange window-chunks src[start:end], stamping chunks with the
// caller's path and kind="orphan". Empty/whitespace-only ranges yield
// no chunks. The line numbers are absolute (1-based) within src.
func orphanRange(relPath string, src []byte, start, end int) []Chunk {
	if start >= end {
		return nil
	}
	slice := string(src[start:end])
	if strings.TrimSpace(slice) == "" {
		return nil
	}
	firstLine := lineOf(src, start)
	lines := strings.Split(slice, "\n")
	wins := windowOver(lines, firstLine)
	for i := range wins {
		wins[i].Path = relPath
		wins[i].Kind = KindOrphan
	}
	return wins
}

func hasCommentPrefix(b []byte) bool {
	if len(b) == 0 {
		return false
	}
	switch b[0] {
	case '/':
		return len(b) >= 2 && (b[1] == '/' || b[1] == '*')
	case '#', '*':
		return true
	case '-':
		return len(b) >= 2 && b[1] == '-'
	}
	return false
}

// backfillComments walks backward from start to absorb a contiguous
// block of leading line comments (`//`, `#`, `--`) or block-comment
// remnants (`/*`, `*`) immediately above the declaration. Limited to
// 50 lines to avoid pulling in unrelated file headers.
//
// `start` must be at the beginning of a line — that is, `src[start-1]`
// is either '\n' or out of range. The function returns a new offset
// that's still at the start of a line.
func backfillComments(src []byte, start int) int {
	pos := start
	lines := 0
	for pos > 0 && lines < 50 {
		// pos points to the start of a line. The previous line ends at
		// pos-1 (a newline, when pos>0) and starts at lineStart where
		// src[lineStart-1] is '\n' or lineStart==0.
		if src[pos-1] != '\n' {
			break
		}
		lineStart := pos - 1 // index of the trailing newline
		// Walk lineStart back to the first byte of the previous line.
		for lineStart > 0 && src[lineStart-1] != '\n' {
			lineStart--
		}
		// src[lineStart:pos-1] is the previous line's content (no newline).
		trimmed := bytes.TrimLeft(src[lineStart:pos-1], " \t")
		if hasCommentPrefix(trimmed) {
			pos = lineStart
			lines++
			continue
		}
		break
	}
	return pos
}

func lineOf(src []byte, byteOffset int) int {
	if byteOffset < 0 {
		byteOffset = 0
	}
	if byteOffset > len(src) {
		byteOffset = len(src)
	}
	return 1 + bytes.Count(src[:byteOffset], []byte{'\n'})
}

func windowChunks(relPath string, src []byte) []Chunk {
	lines := strings.Split(string(src), "\n")
	wins := windowOver(lines, 1)
	for i := range wins {
		wins[i].Path = relPath
		wins[i].Kind = KindWindow
	}
	return wins
}

// windowOver slices `lines` into WindowLines-sized windows with
// WindowOverlap rows of repeat. firstLineNumber is the 1-based line
// number of lines[0]. Chunks larger than MaxBytes are further split by
// halving the window size until they fit.
func windowOver(lines []string, firstLineNumber int) []Chunk {
	var out []Chunk
	step := WindowLines - WindowOverlap
	if step <= 0 {
		step = WindowLines
	}
	for i := 0; i < len(lines); i += step {
		j := min(i+WindowLines, len(lines))
		content := strings.Join(lines[i:j], "\n")
		if len(content) > MaxBytes {
			// Halve and re-split this slice.
			out = append(out, halveAndChunk(lines[i:j], firstLineNumber+i)...)
		} else if strings.TrimSpace(content) != "" {
			out = append(out, Chunk{
				StartLine: firstLineNumber + i,
				EndLine:   firstLineNumber + j - 1,
				Content:   content,
			})
		}
		if j == len(lines) {
			break
		}
	}
	return out
}

func halveAndChunk(lines []string, firstLineNumber int) []Chunk {
	if len(lines) == 0 {
		return nil
	}
	if len(lines) == 1 {
		// A single oversized line (typical: minified JS bundle, generated
		// parser, single-line JSON config) cannot be split further on a
		// newline boundary. Fall back to byte-window slicing so we don't
		// silently lose the content from the index.
		return byteWindows(lines[0], firstLineNumber)
	}
	mid := len(lines) / 2
	first := lines[:mid]
	second := lines[mid:]
	var out []Chunk
	if c := strings.Join(first, "\n"); len(c) <= MaxBytes && strings.TrimSpace(c) != "" {
		out = append(out, Chunk{
			StartLine: firstLineNumber,
			EndLine:   firstLineNumber + len(first) - 1,
			Content:   c,
		})
	} else {
		out = append(out, halveAndChunk(first, firstLineNumber)...)
	}
	if c := strings.Join(second, "\n"); len(c) <= MaxBytes && strings.TrimSpace(c) != "" {
		out = append(out, Chunk{
			StartLine: firstLineNumber + len(first),
			EndLine:   firstLineNumber + len(lines) - 1,
			Content:   c,
		})
	} else {
		out = append(out, halveAndChunk(second, firstLineNumber+len(first))...)
	}
	return out
}

// byteWindows splits a single long line into MaxBytes-sized chunks. All
// chunks share the same start_line/end_line since they came from the
// same source line. Empty inputs yield no chunks. Cut points are
// snapped forward to UTF-8 boundaries so a multi-byte rune is never
// split.
func byteWindows(line string, lineNumber int) []Chunk {
	if strings.TrimSpace(line) == "" {
		return nil
	}
	var out []Chunk
	for i := 0; i < len(line); {
		j := min(i+MaxBytes, len(line))
		for j < len(line) && (line[j]&0xC0) == 0x80 {
			j++
		}
		out = append(out, Chunk{
			StartLine: lineNumber,
			EndLine:   lineNumber,
			Content:   line[i:j],
		})
		i = j
	}
	return out
}

// EmbedText is what's actually sent to the embedding endpoint. Stamping
// path + kind + name + parent into the embedded text improves retrieval
// for both top-level and nested declarations.
func (c Chunk) EmbedText() string {
	var b strings.Builder
	b.WriteString("// path: ")
	b.WriteString(c.Path)
	b.WriteByte('\n')
	if c.Kind != "" && c.Kind != KindWindow {
		b.WriteString("// kind: ")
		b.WriteString(c.Kind)
		b.WriteByte('\n')
	}
	if c.Parent != "" {
		b.WriteString("// parent: ")
		b.WriteString(c.Parent)
		b.WriteByte('\n')
	}
	if c.Name != "" {
		b.WriteString("// name: ")
		b.WriteString(c.Name)
		b.WriteByte('\n')
	}
	b.WriteString(c.Content)
	return b.String()
}
