# `dex guide` improvement handoff

Scope: `internal/guide/{render,config}.go`, plus one prompt change in `internal/index/`. Ranked by impact / effort ratio.

## Shipped on `feat/guide-polish-v2`

1. Filter `testdata/` directories from module list — `f3d6380`.
2. Drop `(root)` module section; Overview covers it — `6a57a5e`.
3. Renderer sections use `###`, not `**bold:**` — `4cc581c`.
4. Cap Exported API at top 10 by centrality, with "…and N more" footer — `f0e318a`.
5. TOC under Overview — `b7ae763`.
6. Default output `LLM_GUIDE.md`, not `.txt`; `scripts/pre-commit-guide.sh` and docs updated — `e638280`.
7. `docs/llm-guide.md` → `docs/how-dex-guide-works.md` — `722f0f5`.
8. `--full` help text now notes the manifest bump — `d126d1e`.
9. Truncation detector on stored summaries; warnings on stderr; `--check` exits non-zero — `46b2ce0`.
10. Package-summary prompt locked to a single prose paragraph (no bold sub-headers, no markdown); `FinishReason=length` now errors out so new truncations can't enter the index — `e462cfa`.

Note on #10: the original proposal was a three-section schema (`Role` / `Exports` / `Notes`). Shipped a stricter variant — pure prose, no internal structure — because the renderer's `###` sections already carry every structured field. Same end state: uniform module sections, no collision with renderer headers.

## Open — nice-to-haves

11. **`--stdout`** — render to stdout without writing files. Trivial in `cmdGuide`.
12. **`--module <path>`** — render only one module's section. Useful for "what does my touched file expose?" lookups.
13. **`--dry-run` reports size** — bytes + estimated tokens, so users can decide whether the guide fits an agent's context window.
14. **Module ordering by PageRank**, not alphabetical — surface architecturally important packages first.

## How to evaluate (for 11–14)

- Re-run `dex index . --summarize` then `dex guide .` after each change.
- Diff `LLM_GUIDE.md` against the prior version — confirm only the intended sections changed.
- `go test ./internal/guide/...` should pass throughout.
