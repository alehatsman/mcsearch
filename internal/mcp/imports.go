package mcp

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
)

// readLineRange returns the 1-indexed [start, end] line slice of a
// file, clipped at maxLines and maxBytes. truncated reports whether
// either cap fired before reaching end.
func readLineRange(path string, start, end, maxLines, maxBytes int) (string, bool, error) {
	if maxLines <= 0 || maxBytes <= 0 {
		return "", false, nil
	}
	if start <= 0 {
		start = 1
	}
	if end < start {
		end = start
	}
	f, err := os.Open(path)
	if err != nil {
		return "", false, err
	}
	defer func() { _ = f.Close() }()

	sc := bufio.NewScanner(f)
	// Lift the default 64 KB line cap so minified files don't bail —
	// we still bound the output via maxBytes below.
	sc.Buffer(make([]byte, 64*1024), 1024*1024)

	var buf strings.Builder
	lineNum := 0
	included := 0
	truncated := false
	for sc.Scan() {
		lineNum++
		if lineNum < start {
			continue
		}
		if lineNum > end {
			break
		}
		if included >= maxLines {
			truncated = true
			break
		}
		line := sc.Bytes()
		if buf.Len()+len(line)+1 > maxBytes {
			truncated = true
			break
		}
		buf.Write(line)
		buf.WriteByte('\n')
		included++
	}
	if err := sc.Err(); err != nil {
		return "", false, err
	}
	// If we exited the loop because the file ended before EndLine,
	// that's not truncation by the cap — leave truncated as-is.
	return buf.String(), truncated, nil
}

// importExtractors maps a lower-cased file extension to its per-language
// extractor. Adding a language is one registry entry, not a switch
// edit. The extractor receives the first N lines of the file and
// returns the contiguous import block as a single string (newlines
// preserved).
var importExtractors = map[string]func([]string) string{
	".go":  extractGoImports,
	".py":  extractPrefixImports([]string{"import ", "from "}),
	".pyi": extractPrefixImports([]string{"import ", "from "}),
	".ts":  extractJSImports,
	".tsx": extractJSImports,
	".js":  extractJSImports,
	".jsx": extractJSImports,
	".mjs": extractJSImports,
	".cjs": extractJSImports,
	".rs":  extractPrefixImports([]string{"use "}),
}

// extractImports reads the first 200 lines of absPath and returns the
// file's contiguous import block, or "" when no extractor matches the
// extension. Used by the inline-content pipeline to surface a callee's
// dependency surface alongside its summary.
func extractImports(absPath string) string {
	lines, err := readFirstNLines(absPath, 200)
	if err != nil || len(lines) == 0 {
		return ""
	}
	if fn, ok := importExtractors[strings.ToLower(filepath.Ext(absPath))]; ok {
		return fn(lines)
	}
	return ""
}

// readFirstNLines returns up to n lines from the start of path. A 1 MiB
// per-line cap covers minified bundles without OOM on a pathological
// input.
func readFirstNLines(path string, n int) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	out := make([]string, 0, n)
	for i := 0; i < n && sc.Scan(); i++ {
		out = append(out, sc.Text())
	}
	if err := sc.Err(); err != nil {
		return out, err
	}
	return out, nil
}

// extractGoImports captures `import (...)` blocks and consecutive
// single-line `import "..."` statements. Go is the only language with
// a block form, so it gets its own extractor instead of the prefix
// helper below.
func extractGoImports(lines []string) string {
	var out []string
	inBlock := false
	for _, l := range lines {
		t := strings.TrimSpace(l)
		if inBlock {
			out = append(out, l)
			if t == ")" {
				break
			}
			continue
		}
		if strings.HasPrefix(t, "import (") {
			inBlock = true
			out = append(out, l)
			continue
		}
		if strings.HasPrefix(t, "import \"") {
			out = append(out, l)
			continue
		}
		if len(out) > 0 {
			break // run of single-line imports ended
		}
		// Pre-import noise (package decl, copyright, build tags) — skip.
	}
	return strings.Join(out, "\n")
}

// extractPrefixImports builds an extractor that captures a contiguous
// run of lines whose trimmed prefix matches any of `prefixes`. Blank
// lines and (for Python) line comments inside the run are tolerated;
// the first non-matching, non-blank line after the run ends it.
//
// Covers Python (`import`, `from`), Rust (`use`), and any future
// keyword-prefix language. JS gets its own extractor because of
// `require(...)` and the eight prefix variants of ES `import`.
func extractPrefixImports(prefixes []string) func([]string) string {
	return func(lines []string) string {
		var out []string
		started := false
		for _, l := range lines {
			t := strings.TrimSpace(l)
			match := false
			for _, p := range prefixes {
				if strings.HasPrefix(t, p) {
					match = true
					break
				}
			}
			if match {
				out = append(out, l)
				started = true
				continue
			}
			if started && (t == "" || strings.HasPrefix(t, "#")) {
				out = append(out, l)
				continue
			}
			if started {
				break
			}
		}
		for len(out) > 0 && strings.TrimSpace(out[len(out)-1]) == "" {
			out = out[:len(out)-1]
		}
		return strings.Join(out, "\n")
	}
}

// extractJSImports captures top-of-file ES `import ... from "..."`
// statements and CommonJS `require("...")` lines. Distinct from the
// generic prefix helper because the start-of-line shape varies
// (`import {`, `import *`, `import "`, …) and `require` may appear
// after an LHS assignment.
func extractJSImports(lines []string) string {
	var out []string
	started := false
	for _, l := range lines {
		t := strings.TrimSpace(l)
		importish := strings.HasPrefix(t, "import ") ||
			strings.HasPrefix(t, "import{") ||
			strings.HasPrefix(t, "import*") ||
			strings.HasPrefix(t, "import\"") ||
			strings.HasPrefix(t, "import '") ||
			strings.Contains(t, "require(\"") ||
			strings.Contains(t, "require('")
		if importish {
			out = append(out, l)
			started = true
			continue
		}
		if started && t == "" {
			out = append(out, l)
			continue
		}
		if started {
			break
		}
	}
	for len(out) > 0 && strings.TrimSpace(out[len(out)-1]) == "" {
		out = out[:len(out)-1]
	}
	return strings.Join(out, "\n")
}
