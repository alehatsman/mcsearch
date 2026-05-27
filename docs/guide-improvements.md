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

## Shipped on `feat/guide-nice-to-haves`

11. `--stdout` — render the guide to stdout without writing the file or bumping the manifest — `975e42f`.
12. `--module <dir>` — render only one module's section, stdout-only — `0b05382`.
13. `--dry-run` reports bytes and estimated tokens — `04b90b8`.
14. Module ordering by `SUM(pagerank)` per directory; alphabetical fallback for zero-score tail — `67da601`.

All 14 handoff items have shipped. Future improvements live in commit messages and code; this doc can be retired.
