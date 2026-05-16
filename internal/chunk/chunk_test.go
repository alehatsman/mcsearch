package chunk

import (
	"context"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestStructuralAndOrphanChunks(t *testing.T) {
	src := []byte(`package main

import "fmt"

const Important = "x"

// Greet prints a greeting.
func Greet() {
	fmt.Println("hi")
}

var TrailingVar = 42
`)
	chunks, err := Chunks(context.Background(), "x.go", src)
	if err != nil {
		t.Fatal(err)
	}
	var sawFunc, sawOrphanHead, sawOrphanTail bool
	for _, c := range chunks {
		switch {
		case c.Kind == "function_declaration" && strings.Contains(c.Content, "func Greet"):
			sawFunc = true
			// Backfilled doc comment must be included.
			if !strings.Contains(c.Content, "Greet prints a greeting") {
				t.Errorf("function chunk missing doc comment: %q", c.Content)
			}
		case c.Kind == "orphan" && strings.Contains(c.Content, "Important"):
			sawOrphanHead = true
		case c.Kind == "orphan" && strings.Contains(c.Content, "TrailingVar"):
			sawOrphanTail = true
		}
	}
	if !sawFunc {
		t.Error("expected function_declaration chunk for Greet")
	}
	if !sawOrphanHead {
		t.Error("expected orphan chunk covering top-level const")
	}
	if !sawOrphanTail {
		t.Error("expected orphan chunk covering trailing var")
	}
}

func TestLongLineFallsBackToByteWindows(t *testing.T) {
	// A single line longer than MaxBytes. Without byte-window fallback,
	// halveAndChunk returned nil and the file produced zero chunks.
	long := strings.Repeat("ab", MaxBytes) // 2*MaxBytes bytes, no newline
	chunks, err := Chunks(context.Background(), "blob.txt", []byte(long))
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least one chunk for an oversized single-line file")
	}
	for _, c := range chunks {
		if len(c.Content) > MaxBytes {
			t.Errorf("chunk len %d > MaxBytes %d", len(c.Content), MaxBytes)
		}
	}
	total := 0
	for _, c := range chunks {
		total += len(c.Content)
	}
	if total != len(long) {
		t.Errorf("byte-window coverage = %d bytes, want %d", total, len(long))
	}
}

func TestUnknownExtensionUsesWindow(t *testing.T) {
	src := []byte("title: hello\nbody: this is plain text\n")
	chunks, err := Chunks(context.Background(), "notes.unknown", src)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 window chunk; got %d", len(chunks))
	}
	if chunks[0].Kind != "window" {
		t.Errorf("kind = %q, want window", chunks[0].Kind)
	}
}

func TestEmptyAndWhitespaceOnly(t *testing.T) {
	for _, src := range []string{"", "   \n\n\t\n"} {
		chunks, err := Chunks(context.Background(), "blank.go", []byte(src))
		if err != nil {
			t.Fatal(err)
		}
		if len(chunks) != 0 {
			t.Errorf("expected 0 chunks for %q; got %d", src, len(chunks))
		}
	}
}

func TestEmbedTextStampsPathAndKind(t *testing.T) {
	c := Chunk{Path: "pkg/x.go", Kind: "function_declaration", Content: "func f(){}"}
	got := c.EmbedText()
	if !strings.HasPrefix(got, "// path: pkg/x.go\n// kind: function_declaration\nfunc f(){}") {
		t.Errorf("EmbedText prefix wrong: %q", got)
	}

	w := Chunk{Path: "x.md", Kind: "window", Content: "hello"}
	got = w.EmbedText()
	// "window" kind is suppressed in the embed-text header to keep
	// noise out of plain text embeddings.
	if strings.Contains(got, "kind:") {
		t.Errorf("window chunks shouldn't emit a kind header: %q", got)
	}
}

func TestUTF8BoundaryInByteWindows(t *testing.T) {
	// Each `é` is 2 bytes in UTF-8; an oversized line of them must
	// never be sliced through a rune boundary.
	line := strings.Repeat("é", MaxBytes) // 2*MaxBytes bytes
	chunks, err := Chunks(context.Background(), "u.txt", []byte(line))
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range chunks {
		if !utf8.ValidString(c.Content) {
			t.Errorf("chunk %d contains invalid UTF-8: % x", i, []byte(c.Content)[:32])
		}
	}
}

// Sanity that the structural pass picks up Python defs/classes.
func TestPythonStructural(t *testing.T) {
	src := []byte(`"""Module docstring."""

def hello(name):
    """Say hi."""
    return f"hello {name}"

class Greeter:
    def __init__(self):
        self.x = 1
`)
	chunks, err := Chunks(context.Background(), "g.py", src)
	if err != nil {
		t.Fatal(err)
	}
	var sawFn, sawCls bool
	for _, c := range chunks {
		if c.Kind == "function_definition" && strings.Contains(c.Content, "def hello") {
			sawFn = true
		}
		if c.Kind == "class_definition" && strings.Contains(c.Content, "class Greeter") {
			sawCls = true
		}
	}
	if !sawFn {
		t.Error("expected function_definition chunk for hello()")
	}
	if !sawCls {
		t.Error("expected class_definition chunk for Greeter")
	}
}

func TestLineCountsAreOneBased(t *testing.T) {
	src := []byte("package x\n\nfunc A() {}\n")
	chunks, _ := Chunks(context.Background(), "a.go", src)
	for _, c := range chunks {
		if c.StartLine < 1 {
			t.Errorf("StartLine = %d, want ≥1 (chunk: %q)", c.StartLine, c.Content)
		}
	}
}
