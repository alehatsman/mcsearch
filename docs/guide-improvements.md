# `dex guide` improvement handoff

Scope: `internal/guide/{render,config}.go`, plus one prompt change in `internal/index/`. Ranked by impact / effort ratio.

## Quick wins (renderer-only; deterministic)

1. **Filter `testdata/` directories from module list**
   Problem: 5 of 22 modules in `LLM_GUIDE.txt` are test fixtures (`internal/graph/testdata/...`). Pure noise.
   Fix: in `Render` (`internal/guide/render.go:50`), after loading `pkgRows`, drop rows whose `Path` contains a `testdata/` segment. Reuse `isFixturePath` from `internal/mcp/context.go` or inline.
   Done when: regenerated guide has 17 modules, none under `testdata/`.

2. **Drop `(root)` module section — keep only the Overview**
   Problem: `## Module: (root)` duplicates the `## Overview` content above it.
   Fix: in `buildMarkdown` (`render.go:98`), skip the `pkgs[i]` whose `Path == "."` or `""`. Overview already covers the root.
   Done when: guide has one root-level prose section (Overview), then per-subdirectory module sections.

3. **Use `###` for deterministic sections, not `**bold:**`**
   Problem: Renderer's `**Exported API**` collides visually with LLM-generated `**internal/graph**`-style bold prefixes inside summaries. Reader can't tell what's authoritative.
   Fix: in `appendGraphSections` (`render.go:132`), change `**Exported API**`, `**Key entry points**`, `**Depends on**`, `**Used by**` to `### Exported API`, etc.
   Done when: all renderer-emitted sections use H3; the LLM prose stays in bold/plain text and is visually distinct.

4. **Cap Exported API at top-N by centrality**
   Problem: `internal/store` lists 51 exported symbols flat. Buries the important ones.
   Fix: in `appendGraphSections`, sort `exported` by in-degree/PageRank (use the same `TopCentral*` data already fetched) and cap at 10. Append `_…and N more — search via `dex search symbol`._` when truncated.
   Done when: no module section has more than 10 Exported API bullets; large modules show the "and N more" footer.

5. **Add TOC at top**
   Problem: 846-line markdown, no jump list.
   Fix: in `buildMarkdown`, after the Overview and before the first module, emit a `## Contents` block with `- [<dir>](#module-<slug>)` lines. Slug = lowercase, `/` → empty, matches GitHub anchor rules.
   Done when: clicking a TOC entry in any markdown viewer jumps to the corresponding `## Module: ...` heading.

6. **Default output to `LLM_GUIDE.md`, not `.txt`**
   Problem: content is markdown; editors don't render it correctly with a `.txt` extension.
   Fix: change `DefaultConfig().Output` (`internal/guide/config.go:25`) from `"LLM_GUIDE.txt"` to `"LLM_GUIDE.md"`. Rename the committed artifact in the same PR; update `scripts/pre-commit-guide.sh` and any docs references.
   Done when: fresh `dex guide` writes `LLM_GUIDE.md`; existing `LLM_GUIDE.txt` is removed from the repo.

7. **Rename `docs/llm-guide.md`** (disambiguate from the artifact)
   Problem: artifact (`LLM_GUIDE.md`) and explainer (`docs/llm-guide.md`) look identical in directory listings.
   Fix: `git mv docs/llm-guide.md docs/how-dex-guide-works.md` (or `docs/guide-internals.md`). Update any links.
   Done when: `find . -iname 'llm*guide*'` returns one file (the generated artifact); the explainer has a clearly distinct name.

8. **Document `--full` bumps the manifest**
   Problem: `--full` is "re-render unconditionally" but it also updates `manifest.LastSummarySeenAt`. Subtle.
   Fix: update the `--full` flag description in `cmd/dex/guide.go:18` to: `"ignore manifest and re-render unconditionally (also updates the manifest)"`.
   Done when: `dex guide --help` makes the side effect explicit.

## Medium effort

9. **Detect truncated LLM summaries; surface in `--check`**
   Problem: committed `LLM_GUIDE.txt:29` ends a bullet mid-word (`- **`internal`). Indexer's chat call hit max-tokens; renderer silently shipped malformed output.
   Fix: in `Render`, after loading summaries, scan each for: unterminated backticks (odd count of `` ` ``), trailing `- **` patterns, sentences not ending in punctuation. Add `res.Warnings []string` to `Result`. CLI prints warnings; `--check` exits non-zero if any.
   Done when: a truncated summary causes `dex guide --check` to fail with a pointer to the offending package.

10. **Constrain the package-summary LLM prompt to a fixed schema**
    Problem: bold headers vary wildly across modules (`**Public Role:**`, `**Public Role in System:**`, plain text…). Caused by an unconstrained summarize prompt at index time.
    Fix: in `internal/index/index.go` (the `summarizePackage` path around `:921`), update the prompt to require exactly three sub-sections with fixed names: `**Role.**`, `**Exports.**`, `**Notes.**`. Add a regression check post-summarize that all three appear; if not, re-prompt once.
    Done when: every `package_summary` chunk in the index follows the same three-section structure. New `dex guide` output is uniform across modules.

## Optional / nice-to-haves (bottom of pile)

11. **`--stdout`** — render to stdout without writing files. Trivial in `cmdGuide`.
12. **`--module <path>`** — render only one module's section. Useful for "what does my touched file expose?" lookups.
13. **`--dry-run` reports size** — bytes + estimated tokens, so users can decide whether the guide fits an agent's context window.
14. **Module ordering by PageRank**, not alphabetical — surface architecturally important packages first.

## How to evaluate

- Re-run `dex index . --summarize` then `dex guide .` after each change.
- Compare diffs of `LLM_GUIDE.md` against the version before the change — confirm only the intended sections changed.
- For #9 and #10, force a truncation (lower `DEX_SUMMARY_MAX_TOKENS` to ~50) and verify the warning surfaces.
- `go test ./internal/guide/...` should pass throughout. Add a table-test for the testdata filter (#1), TOC slug generation (#5), and truncation detector (#9).
- Final acceptance: a fresh agent reading `LLM_GUIDE.md` end-to-end should never encounter testdata fixtures, never see a truncated bullet, and be able to jump to any module via the TOC.
