// `mcsearch compact` — concatenate a folder's text files for LLM prompts.
//
// Walks <path>, applies the same filters mcsearch uses for indexing
// (ignore.New + IndexableExt/IndexableBasename + LooksBinary +
// LooksLikeSecret), and emits each surviving file with an `===== <relpath> =====`
// header followed by its contents. Output goes to stdout, or to --out
// when provided.
package main

import (
	"bufio"
	"bytes"
	"context"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/alehatsman/mcsearch/internal/ignore"
)

func cmdCompact(_ context.Context, args []string) (err error) {
	flags := flag.NewFlagSet("compact", flag.ContinueOnError)
	out := flags.String("out", "", "write to file instead of stdout")
	maxBytes := flags.Int64("max-bytes", 1<<20, "skip individual files larger than N bytes")
	strip := flags.Bool("strip", false, "drop line comments (// and #), blank lines, and trailing whitespace")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() < 1 {
		return fmt.Errorf("usage: mcsearch compact <path> [--out FILE] [--max-bytes N]")
	}

	root, err := filepath.Abs(flags.Arg(0))
	if err != nil {
		return err
	}
	info, err := os.Stat(root)
	if err != nil {
		return err
	}
	if !info.IsDir() {
		return fmt.Errorf("not a directory: %s", root)
	}

	matcher, err := ignore.New(root)
	if err != nil {
		return err
	}

	osRoot, err := os.OpenRoot(root)
	if err != nil {
		return err
	}
	defer func() { _ = osRoot.Close() }()

	var w io.Writer = os.Stdout
	if *out != "" {
		f, ferr := os.Create(*out)
		if ferr != nil {
			return ferr
		}
		bw := bufio.NewWriter(f)
		w = bw
		defer func() {
			if ferr := bw.Flush(); ferr != nil && err == nil {
				err = ferr
			}
			if ferr := f.Close(); ferr != nil && err == nil {
				err = ferr
			}
		}()
	}

	return filepath.WalkDir(root, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, rerr := filepath.Rel(root, path)
		if rerr != nil {
			return rerr
		}
		if rel == "." {
			return nil
		}
		if matcher.Match(rel, d.IsDir()) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			return nil
		}
		if !ignore.IndexableExt(path) && !ignore.IndexableBasename(path) {
			return nil
		}
		fi, ferr := d.Info()
		if ferr != nil {
			return ferr
		}
		if fi.Size() > *maxBytes {
			fmt.Fprintf(os.Stderr, "skip %s (%d bytes > --max-bytes=%d)\n", rel, fi.Size(), *maxBytes)
			return nil
		}
		data, rerr := osRoot.ReadFile(rel)
		if rerr != nil {
			return rerr
		}
		if ignore.LooksBinary(data) {
			return nil
		}
		if !ignore.IsTestPath(rel) && ignore.LooksLikeSecret(data) {
			fmt.Fprintf(os.Stderr, "skip %s (looks like a secret)\n", rel)
			return nil
		}
		if *strip {
			data = stripContent(data, path)
		}

		relSlash := filepath.ToSlash(rel)
		if _, werr := fmt.Fprintf(w, "===== %s =====\n", relSlash); werr != nil {
			return werr
		}
		if _, werr := w.Write(data); werr != nil {
			return werr
		}
		if len(data) == 0 || data[len(data)-1] != '\n' {
			if _, werr := io.WriteString(w, "\n"); werr != nil {
				return werr
			}
		}
		return nil
	})
}

// stripContent drops blank lines, trailing whitespace, and line
// comments. `//` is treated as a line comment everywhere; `#` is
// stripped from code/config files but preserved in prose (.md/.rst/.txt)
// where it's a heading marker.
func stripContent(data []byte, path string) []byte {
	ext := strings.ToLower(filepath.Ext(path))
	stripHash := ext != ".md" && ext != ".markdown" && ext != ".rst" && ext != ".txt"

	var out bytes.Buffer
	out.Grow(len(data))
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 64*1024), 16*1024*1024)
	for sc.Scan() {
		line := bytes.TrimRight(sc.Bytes(), " \t\r")
		trimmed := bytes.TrimLeft(line, " \t")
		if len(trimmed) == 0 {
			continue
		}
		if bytes.HasPrefix(trimmed, []byte("//")) {
			continue
		}
		if stripHash && bytes.HasPrefix(trimmed, []byte("#")) {
			continue
		}
		out.Write(line)
		out.WriteByte('\n')
	}
	return out.Bytes()
}
