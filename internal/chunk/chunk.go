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
	"context"
	"path/filepath"
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

// Chunk is one retrievable slice of one file.
type Chunk struct {
	Path      string // relative to project root
	Kind      string // tree-sitter node kind or "window"
	StartLine int    // 1-based, inclusive
	EndLine   int    // 1-based, inclusive
	Content   string
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
func Chunks(ctx context.Context, relPath string, src []byte) ([]Chunk, error) {
	ext := strings.ToLower(filepath.Ext(relPath))
	if cfg, ok := languages[ext]; ok {
		out, err := treeChunks(ctx, relPath, src, cfg)
		if err == nil && len(out) > 0 {
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
	for i := 0; i < int(root.NamedChildCount()); i++ {
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
		endLine := lineOf(src, endByte-1)
		if endLine < startLine {
			endLine = startLine
		}
		if len(body) <= MaxBytes {
			out = append(out, Chunk{
				Path: relPath, Kind: kind,
				StartLine: startLine, EndLine: endLine,
				Content: body,
			})
			continue
		}
		// Oversized declaration → fall back to line windows over its body.
		bodyLines := strings.Split(body, "\n")
		for _, w := range windowOver(bodyLines, startLine) {
			w.Path = relPath
			w.Kind = kind + ":window"
			out = append(out, w)
		}
	}
	return out, nil
}

// backfillComments walks backward from start to absorb a contiguous block
// of leading // or # line comments (and the blank line just above the
// declaration, if any). Limited to 50 lines to avoid pulling in unrelated
// file headers.
func backfillComments(src []byte, start int) int {
	pos := start
	lines := 0
	for pos > 0 && lines < 50 {
		// move to start of previous line
		prev := pos - 1
		if prev > 0 && src[prev-1] == '\n' {
			// pos is already at line start; step back one line
			// find start of previous line
			lineStart := prev - 1
			for lineStart > 0 && src[lineStart-1] != '\n' {
				lineStart--
			}
			trimmed := strings.TrimLeft(string(src[lineStart:prev-1]), " \t")
			if strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "#") || strings.HasPrefix(trimmed, "/*") || strings.HasPrefix(trimmed, "*") {
				pos = lineStart
				lines++
				continue
			}
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
	return 1 + strings.Count(string(src[:byteOffset]), "\n")
}

func windowChunks(relPath string, src []byte) []Chunk {
	lines := strings.Split(string(src), "\n")
	wins := windowOver(lines, 1)
	for i := range wins {
		wins[i].Path = relPath
		wins[i].Kind = "window"
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
		j := i + WindowLines
		if j > len(lines) {
			j = len(lines)
		}
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
	if len(lines) <= 1 {
		return nil
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

// EmbedText is what's actually sent to the embedding endpoint. Stamping
// path + kind into the embedded text consistently improves code-search
// retrieval.
func (c Chunk) EmbedText() string {
	var b strings.Builder
	b.WriteString("// path: ")
	b.WriteString(c.Path)
	b.WriteByte('\n')
	if c.Kind != "" && c.Kind != "window" {
		b.WriteString("// kind: ")
		b.WriteString(c.Kind)
		b.WriteByte('\n')
	}
	b.WriteString(c.Content)
	return b.String()
}
