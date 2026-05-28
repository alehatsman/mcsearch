# Agent roadmap — next 10 priorities

Handoff document for future agents. Each item is sized to be picked up
cold by a fresh session: title → motivation → entry points → success
criteria. Ranked by impact × tractability.

This list covers cross-cutting system quality and coherence. The
guide-renderer polish that used to live in `guide-improvements.md`
all shipped in late May 2026 (see commit `e7ed8ea`) — that doc was
dropped, and items here don't duplicate it.

---

## 1. Persistent per-tier model config (`.dex/models.toml`)

**Why.** Tier model env vars (`DEX_{CHUNK,FILE,PACKAGE,REPO}_SUMMARY_MODEL`)
are inherited from whatever process launched dex. The MCP server reads
them from Claude's MCP env block; `dex watch` reads them from the
systemd unit's `Environment=` lines; CLI invocations read them from the
shell. Three independent surfaces with no shared source of truth — the
operator can forget to set them in one place and silently get default
7b across the board. We hit this exact failure during the v2 rollout.

**Entry points.**
- Extend `internal/guide/config.go` schema, or add `internal/index/config.go`.
- Honor TOML at `<project_root>/.dex/models.toml`:
  ```toml
  [models]
  chunk   = "qwen2.5-coder:3b"
  file    = "qwen2.5-coder:7b"
  package = "qwen2.5-coder:14b"
  repo    = "qwen2.5-coder:32b"
  ```
- Precedence: env var > config file > inherited DEX_SUMMARY_MODEL.
- Surface in `dex env` output with the resolved source per tier.

**Done when.** A repo with `.dex/models.toml` uses the configured tier
models regardless of which process launched dex, no env vars set.

---

## 2. ✅ Garbage-collect stale `package_summary` rows on cache miss

**Shipped** in commit `04a872c`. `Store.DeleteOtherSummariesForPath`
(`internal/store/store.go`) runs right after each summary `UpsertMany`
at four sites — Pass 5 + Pass 6 in `internal/index/index.go`, and
`runPackageJobs` + `cascadeRepoSummary` in `internal/index/drain.go`.
GC failure is `Warn`-logged, not fatal: a stale row is recoverable, a
missing fresh row is not. `TestDeleteOtherSummariesForPath` asserts
exactly one row per `(path, kind)` after sequential writes and that
sibling paths / sibling kinds are untouched.

---

## 3. ✅ Ground summarizer prompts with graph data

**Shipped** in commit `98b354e`. New helpers in
`internal/index/grounding.go` feed `summarizePackage` an EXPORTED
SYMBOLS + PROJECT IMPORTS section (from `ExportedSymbolsByDir` +
module-prefix-trimmed `ImportsForDir`) and `summarizeRepo` a PACKAGES
(dir → top symbols) section (from `TopCentralByDir`). System prompts
gained a GROUNDING RULE: backtick-wrapped identifiers in the output
must come from those lists. Sections render only when grounding is
non-empty so the first-index / non-Go / empty-graph fallback keeps the
original prompt shape. `packageSummaryPromptVersion` and bumped
`repoSummaryPromptVersion` (v3→v4) fold into the respective cache keys
so prompt iterations re-run on the next index pass.
`grounding_test.go` exercises the prompt builders directly. A
full-pipeline sweep test that diffs generated `LLM_GUIDE.md`
identifiers against `graph_nodes` is still useful follow-up work — the
prompt-builder tests cover the structural guarantee but not the model's
adherence to it.

**Original entry points** (kept for reference):
- `internal/index/index.go:944` — `summarizePackage`. Today it gets
  `dir` + concatenated file summaries.
- Add an `extraContext` arg: pre-resolved list of exported symbols
  (`Store.ExportedSymbolsByDir`) and project-imports
  (`Store.ImportsForDir`, trimmed to project module prefix). Caller
  fetches once per directory.
- Update prompt: "constrain claims to the symbols and imports listed
  below; do not invent identifiers not in this list."
- Same treatment for `summarizeRepo`: pass the list of package paths +
  the top-N most central symbols per package.

**Done when.** A regenerated `LLM_GUIDE.md` has no module-summary
prose paragraph mentioning identifiers absent from the package's
`graph_nodes`. Validated by a sweep test.

---

## 4. Multi-language graph extraction (Python / TS / Rust)

**Status.** Tracked in `docs/vision.md` scope cut #2 with the
tree-sitter-based implementation sketch — per-language queries in
`internal/graph/sitter_calls.go`, edges tagged `provenance: "sitter"`
so the MCP layer distinguishes them from Go's type-resolved edges.
This entry is the broader extractor framing.

**Why.** Today's static graph is Go-only (`go/packages` + `go/types` in
`internal/graph`). Non-Go projects get a guide with empty "Depends on"
/ "Used by" sections — the most useful grounding signal disappears.
Mooncake (Python+YAML) is the immediate driver: it's our biggest
non-Go repo.

**Entry points.**
- `internal/graph/graph.go` — split the extractor interface so per-language
  backends plug in.
- Per-language tree-sitter queries (see vision scope cut #2). Same
  approach the chunker uses; no per-language subprocess needed.
- Schema reuse: same `graph_nodes` / `graph_edges` tables. `kind`
  enum covers function / method / class / import already.

**Done when.** `dex index <python-project>` populates `graph_nodes`
with classes + functions + imports; `dex guide` renders "Exported
API" and "Depends on" sections for Python packages.

---

## 5. End-to-end watcher pipeline test

**Why.** The defer-mode pruning bug (just fixed) would have been
caught by an integration test that:
1. Indexes a tiny project with `--summarize-defer=false` (writes
   package + repo summaries).
2. Runs a defer-mode index pass (mimicking the watcher).
3. Asserts summaries survive.
4. Runs `dex guide --check` — expects exit 0.

We have store-level tests and CLI-level tests but no test exercises
the full loop. The drainer + cascade + guide rendering interaction is
under-tested.

**Entry points.**
- `internal/mcp/server_test.go` is the closest existing harness — adds
  a watcher and exercises drains.
- New file `e2e/guide_pipeline_test.go` (or `internal/index/pipeline_test.go`)
  with a build tag so it can be opt-in (uses a fake chat server).

**Done when.** New test passes; deliberately reverting the
PruneUnseen fix causes it to fail with a clear "package_summary
unexpectedly missing" message.

---

## 6. ✅ Fix `TestExtractGoNoModule` flake

**Shipped** (pre-priorities sweep, no-op confirmation in this round).
`ExtractGo` now short-circuits at `internal/graph/go.go:62` via
`hasGoModule(projectRoot)` — non-Go trees return an empty
`ExtractResult` without invoking `go/packages`, so the driver warning
that used to leak into the test result is gone.

---

## 7. Structured logging schema (`slog` attributes lockdown)

**Why.** `slog` is in use throughout (`internal/index`, `internal/mcp`,
`internal/watch`) but each call site picks its own attribute keys
(`elapsed`, `took`, `count`, `n`, …). No tooling can reliably ingest
the stream into Grafana / Loki / OTLP. Future on-call work
(performance regressions, model swap measurements) needs this.

**Entry points.**
- Add `internal/obs/log.go` defining a small set of canonical attrs:
  `phase`, `kind`, `duration_ms`, `count_in`, `count_out`, `model`,
  `path`.
- Audit every `Logger.Info` / `Warn` call; replace ad-hoc keys with
  these canonicals.
- Document the schema in `docs/observability.md`.

**Done when.** A grep across the codebase shows every log entry uses
only the canonical attribute set. `slog` JSON output round-trips
through `jq` filters by canonical key.

---

## 8. `dex doctor` command

**Why.** Setup issues today need 6 commands to diagnose: `dex env`,
`dex index status`, `curl /health` on three endpoints, manifest
inspection. A single `dex doctor <path>` should walk the entire chain
and tell the user exactly what's wrong (missing model, stale index,
manifest drift, expired tunnel).

**Entry points.**
- `cmd/dex/main.go` — add `case "doctor":`
- New file `cmd/dex/doctor.go`. Reuse `internal/mcp` health helpers.
- Checks: embed reachable, chat reachable, rerank reachable, each
  configured model is pulled, index exists, manifest fresh, last
  `dex guide --check` would pass.
- Output: green checkmarks per check; first red gets a "to fix:"
  one-liner.

**Done when.** `dex doctor /home/aleh/projects/dex` prints a
diagnostic table; deliberately stopping the rerank container and
re-running produces a clear "rerank: container 'dex-rerank' is not
listening; check `docker ps`" message.

---

## 9. CONTRIBUTING.md + repo navigation guide for agents

**Why.** This roadmap, `PIPELINE.md`, `STORAGE.md`,
`architecture.md`, `internals.md`, `vision.md`,
`how-dex-guide-works.md`, `tuning.md` — already eight sources of
truth. A new agent (or human contributor) walks in cold and doesn't
know which to read first or how they relate. Coherence here is meta:
**a tool that helps agents understand code should be exemplary at
helping agents understand itself.**

**Entry points.**
- New `CONTRIBUTING.md` at repo root: workflow rules (worktrees,
  conventional commits, never push to main, etc. — extract from
  global CLAUDE.md / user habits).
- New `docs/READING_ORDER.md`: explicit ranked list of which doc to
  read for which question (new contributor onboarding vs. bug fix vs.
  feature design).
- Cross-link from `README.md`.

**Done when.** A fresh agent given just "improve dex" can find the
right reading order doc, then the right architecture doc, in two
clicks. Both new files exist and link bidirectionally with the
existing nine.

---

## 10. Index portability — share indexes across teammates

**Why.** `dex clone <src> <dst>` exists for git worktrees (same
machine), but there's no story for "you indexed mooncake, now let me
have your index" across teammates with identical checkouts. Each
person pays the full indexing cost. With per-tier 32b summaries
costing significant GPU minutes, this is now a meaningful concern.

**Entry points.**
- `cmd/dex/main.go:1825` — `cmdClone` is the closest precedent.
- New subcommand `dex export <src-path> <archive.tar.zst>` and
  `dex import <archive.tar.zst>` that ship the SQLite DB + manifest +
  optionally `.dex/models.toml`, **with project_root path
  re-mapping** at import time (each user has a different absolute
  path).
- Validation: import verifies `meta.dim` matches the local
  embedding model's dimension; rejects with a clear error otherwise.

**Done when.** `dex export` produces a portable archive; another
machine can `dex import` it and immediately serve queries without
re-indexing. Cross-machine smoke tested.

---

## Notes for the picker-upper

- **Sequence sensibly.** Items 1 + 2 + 3 compound (config persistence
  enables consistent quality across runs; stale GC keeps the table
  clean; prompt grounding kills the hallucination at source). Tackle
  them as a small series before moving on.
- **Items 4 (multi-language), 5 (e2e test), 6 (flake fix) are
  independent** — pick by mood/skill match.
- **Items 7 + 8 + 9 are observability/DX work** — they don't change
  user-visible behavior but compound future work. Worth a focused
  sprint.
- **Item 10 is a real feature ask** with implications for security
  (sharing indexes = sharing summaries that may quote private code).
  Don't ship without auth thought.

For each item: open a worktree, conventional commit prefix (`feat:`,
`fix:`, `docs:`), `--no-ff` merge to main, never auto-push. Mooncake's
patterns and the codebase's tests speak for themselves — read first,
write second.

Godspeed, trooper.
