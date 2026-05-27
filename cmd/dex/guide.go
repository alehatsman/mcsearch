package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/alehatsman/dex/internal/guide"
	"github.com/alehatsman/dex/internal/proj"
)

func cmdGuide(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("guide", flag.ContinueOnError)
	setHelp(fs,
		"Render LLM_GUIDE.md from existing repo + package summaries in the index.",
		"dex guide [<path>] [--full] [--check] [--dry-run]")
	full := fs.Bool("full", false, "ignore manifest and re-render unconditionally (also bumps the manifest watermark)")
	check := fs.Bool("check", false, "exit non-zero if the guide is out of date; no write")
	dryRun := fs.Bool("dry-run", false, "report what would change without writing files")
	if err := fs.Parse(reorderFlags(fs, args)); err != nil {
		return err
	}
	path, rest := splitProjectArg(fs.Args())
	if len(rest) != 0 {
		return fmt.Errorf("guide takes no extra positional args (got %v)", rest)
	}

	base, err := indexDir()
	if err != nil {
		return err
	}
	p, err := proj.Resolve(path, base)
	if err != nil {
		return err
	}
	if _, err := os.Stat(p.DBPath); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("no index for %s — run `dex index %s --summarize` first", p.Root, p.Root)
		}
		return err
	}

	st, err := openStore(ctx, p.DBPath)
	if err != nil {
		return err
	}
	defer func() { _ = st.Close() }()

	cfg, err := guide.LoadConfig(p.Root)
	if err != nil {
		return err
	}

	opts := guide.Options{Force: *full, DryRun: *check || *dryRun}
	res, err := guide.Render(ctx, st, p.Root, cfg, opts)
	if err != nil {
		return err
	}

	// Print truncation warnings regardless of mode — these signal an
	// older summary in the index that the guard now rejects but that
	// still feeds the guide. Surface them so the user knows to
	// re-summarize.
	for _, w := range res.Warnings {
		fmt.Fprintf(os.Stderr, "warning: %s\n", w)
	}
	if len(res.Warnings) > 0 {
		fmt.Fprintln(os.Stderr, "  → re-run `dex index <path> --summarize` to refresh truncated summaries")
	}

	switch {
	case *check:
		if res.Dirty {
			fmt.Fprintf(os.Stderr, "guide out of date: %s\n", res.OutputPath)
			os.Exit(1)
		}
		if len(res.Warnings) > 0 {
			fmt.Fprintf(os.Stderr, "guide has %d malformed summary chunk(s): %s\n", len(res.Warnings), res.OutputPath)
			os.Exit(1)
		}
		fmt.Printf("✓ guide up to date: %s\n", res.OutputPath)
	case *dryRun:
		if res.Dirty {
			fmt.Printf("would re-render %s (%d modules)\n", res.OutputPath, res.ModuleCount)
		} else {
			fmt.Printf("up to date: %s\n", res.OutputPath)
		}
	case res.Wrote:
		fmt.Printf("✓ wrote %s (%d modules)\n", res.OutputPath, res.ModuleCount)
	default:
		fmt.Printf("up to date: %s\n", res.OutputPath)
	}
	return nil
}
