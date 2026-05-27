# Next priorities — quality + coherence handoff

Scope: ten concrete items to raise the *quality* (correctness, robustness,
retrieval precision) and *coherence* (consistency, no surprises) of dex.
Written for a future agent to pick up. Each item names the file and line,
the diagnosis, the fix, and how to know you're done.

Ordered by impact / effort. The first three are landmines — fix them
before any new feature work. The last three align with [`docs/vision.md`](vision.md)
and are bigger but well-scoped.

## 1. Fix the broken test in `internal/graph`

CI red right now. `go test -tags sqlite_fts5 ./...` fails one case:

```
--- FAIL: TestExtractGoNoModule (0.08s)
    yaml_test.go:65: expected empty ExtractResult on non-Go tree;
      got packages=1 nodes=1 edges=0 warnings=[./...: pattern ./...:
      directory prefix . does not contain main module or its selected
      dependencies]
```

Diagnosis: `ExtractGo` on a tree with no `go.mod` (the `mooncake` fixture)
returns one synthetic package + node + warning. The test wants a fully
empty `ExtractResult`. Either:

- (a) update `ExtractGo` to treat "no module, no go files" as a true no-op
  (return zero packages, zero warnings — the current behaviour leaks a
  `go/packages` driver warning into a non-Go tree); or
- (b) update the test to assert on the structural invariants instead
  (zero nodes / zero edges, allow benign warnings).

Recommended: (a) — `ExtractGo` is already used as "best effort on any
tree" and a warning from the Go driver on a YAML-only project is noise.
Bail before invoking `go/packages` when no `.go` files exist under root.

Done when: `go test -tags sqlite_fts5 ./internal/graph/...` is green.

## 2. Record the embedding model name in `meta`, refuse silent swaps

`internal/store/store.go:41,76,553-588` records *only* `dim` in
`meta` and rejects on `len != dim`. A same-dim model swap (e.g.
`Qwen3-Embedding-4B` → another 2560-dim model) is undetectable — the
vec table happily mixes incompatible vectors and retrieval quality
silently collapses.

Diagnosis: dim alone is not a model identity. Two different models with
the same dimension produce vectors in different latent spaces.

Fix:
- Add a `metaEmbedModel = "embed_model"` key alongside `metaDim`.
- On first `UpsertMany`, write both dim and model name (caller supplies
  the model — `internal/embed.Client.ModelName()` exists already).
- On subsequent `UpsertMany`, refuse with a clear actionable error if
  the model name differs: *"embedding model changed (`A` → `B`); run
  `dex reindex <path>` to rebuild"*.
- Surface the recorded model in `dex index status` JSON.

Done when: `dex index status` includes the active embed model per
project, and changing `DEX_EMBED_MODEL` between runs is rejected with a
human-readable hint instead of corrupting the index.

## 3. FTS query builder drops Unicode identifiers and offers no precision knob

`internal/store/store.go:1274` (`buildFTSQuery`) accepts only ASCII
`[A-Za-z0-9_]` per rune. Two consequences:

- A query like *"ユーザー認証"* or *"ParseRFC3339Núñez"* loses tokens
  entirely. BM25 contributes nothing; only cosine fires.
- Default is `OR` across tokens. Comment says "bad lexical matches are
  sunk by their BM25 rank anyway" — true for natural language, but for
  symbol-shaped queries (e.g. *"GraphPruneUnseen returns count"*) we
  want `AND`-ish precision.

Fix:
- Use `unicode.IsLetter` / `IsDigit` / `r == '_'` (FTS5's `unicode61`
  tokenizer with `remove_diacritics 2` is already in the schema; the
  query side just needs to keep up).
- Quote multi-word phrases when the caller passes a single-quoted
  substring (so *"\"package boundary\""* survives as an FTS phrase).
- Add a `mode` knob to `Options` (default `Auto`): `OR` when query has
  ≥3 fields, `AND` when 1–2. Tiny heuristic, big precision lift on
  symbol-shaped questions.

Done when: a query with non-ASCII identifiers returns FTS hits; a
quoted-phrase query keeps the phrase together; a two-token query
defaults to `AND`. Add table-tests for each case in `store_search_test.go`.

## 4. Cross-encoder rerank: timeout, circuit breaker, in-process cache

`internal/store/store.go:Search` falls through to the fused order *silently*
on `rerank.ErrUnreachable` (per `STORAGE.md:81`). Two operational gaps:

- One slow rerank request can extend the whole `ask` round-trip past
  the MCP timeout. There's no per-call deadline split.
- Repeated queries (the user iterates) re-rerank the same `(query, id-set)`.

Fix:
- Wrap the rerank call in a `context.WithTimeout` derived from
  `Options.RerankTimeout` (new field, default 1500ms).
- Add a token-bucket / consecutive-failure circuit breaker on
  `internal/rerank.Client` — after 3 consecutive failures, open the
  breaker for 30s; status surfaces via `dex index status`.
- Add a small in-process LRU keyed on `sha1(query + sorted(ids))`. 256
  entries is plenty for an interactive session.

Done when: a hung rerank endpoint can't stall `ask` past the timeout;
`dex index status` shows breaker state; identical follow-up queries skip
the rerank network call.

## 5. Pending-summaries backpressure + visible queue health

`index_status` already surfaces `pending_summaries` count
(`internal/mcp/server.go:698`). Today's snapshot of this very project
shows the worktree at **362 pending** — proof that the queue can fall
arbitrarily behind without anyone noticing.

Fix:
- Add `pending_summaries_oldest_age_s` to `Stats` — single SQL
  `SELECT (strftime('%s','now') - MIN(queued_at)) FROM pending_summaries`.
- In `IdleSummaryDrainer` (`internal/index/drain.go:230`), expose a
  per-batch error budget: if N consecutive batches fail, back off
  exponentially up to 30 min and log a warning instead of busy-looping.
- In `dex index status`, when pending > 100 *or* oldest > 1h, print a
  one-line hint: *"summarization queue is behind; run `dex index summarize <path>`"*.

Done when: a stalled chat endpoint doesn't produce a thousand-line slog
spam loop; a stale queue is obvious from `dex index status`.

## 6. Context cancellation audit across embed / chat / rerank batches

`internal/embed`, `internal/chat`, `internal/rerank` all use
`net/http` correctly, but the *pipelines* that drive them are batch
loops that swallow `ctx.Done()` between batches. Cancel a long
`dex index --summarize` and observe how long it actually takes to stop.

Fix: grep for `for i := 0; i < len(batches); i++` style loops in
`internal/index/index.go` and `internal/index/drain.go`; add
`select { case <-ctx.Done(): return ctx.Err(); default: }` at the top of
each iteration. Same for the chunk-walk pool in Pass 1.

Done when: `^C` on a `dex index --summarize` against a 10k-file repo
returns within ~1 second instead of finishing the current batch wave.
Add a test that cancels mid-batch via a mock embed client.

## 7. Collapse path/import heuristics in `internal/mcp/context.go`

`internal/mcp/context.go` is 1577 lines. Most of it is fine, but two
clusters are heuristic-creep and will keep accreting bugs:

- **Path classifiers** at 693–791: `isDocPath`, `isBuildOrConfigPath`,
  `isTestPath`, `isFixturePath`, `isNonImplPath`. Each is an ad-hoc
  `strings.Contains` chain. They overlap (every fixture is also non-impl;
  every test is also non-impl) and they're called from `pickSuggestedReads`
  in priority order with no central rule table.

- **Per-language import extractors** at 1408–1524: `extractGoImports`,
  `extractPythonImports`, `extractJSImports`, `extractRustImports`. Same
  shape, switched in `extractImports` by extension — but the dispatch
  is a hand-rolled switch, not a registry, and adding a language means
  editing two places.

Fix:
- One `pathTag(path) []tag` returning a set drawn from
  `{doc, build, test, fixture}`. `isNonImpl` becomes `len(tags) > 0`.
  Callers ask for tag membership instead of running five `strings.Contains`.
- One `var importExtractors = map[string]func([]string) string{ ".go": …, … }`
  with a single dispatcher.

This is pure refactor — no behaviour change. Verify by running
`go test ./internal/mcp/...` (the router has good test coverage at
`context_test.go`, 1448 lines).

Done when: the five path predicates are one function backed by a tag
set; the five import extractors are one registry; `internal/mcp/context.go`
drops below 1300 lines without losing a test.

## 8. `references` edges via LSP-as-consumer (gopls first)

[`docs/vision.md:5`](vision.md) item 4 calls this out, and
[`internal/graph/graph.go:10`](../internal/graph/graph.go) says
*"`references` lands with the LSP-as-consumer work"*. Today
`graph_callers` / `graph_callees` use the static `go/types` `calls`
graph, which is precise for direct calls but misses interface method
sets and reflection.

Fix (Go-only for v1):
- New `internal/lsp` package: thin client over `gopls` stdio.
- After the graph pass, for each `function`/`method` node, ask gopls
  `textDocument/references` and persist hits as `references` edges
  (kind = `references`, distinct from `calls`).
- Gate behind `--lsp` flag and `gopls` on PATH. Falls back to
  static-only graph if gopls is missing.
- In `ask`, when intent is `callers` and `references` exist for the
  symbol, prefer them over `calls` for the response.

Done when: `dex graph callers <name>` returns interface-method-set
callers that the static `calls` extractor misses; with `--lsp=off` the
behaviour is identical to today.

## 9. Tree-sitter `calls` extraction for Python / JS-TS / Rust

Currently graph callers/callees are Go-only and non-Go projects fall
back to a ripgrep usage list (`ask` references lane). That's a coherence
gap — the chunker already supports 8+ languages
(`internal/chunk/chunk.go:110`), but the graph is mono-language.

Fix:
- New `internal/graph/sitter_calls.go`: per-language tree-sitter queries
  that emit `(caller, callee_name)` pairs. Resolution within a file is
  best-effort by name; cross-file resolution uses the existing
  `graph_nodes` qualified-name index.
- Wire `ExtractSitterCalls(ctx, root)` into the graph pass after
  `ExtractGo` / `ExtractYAML`.
- Mark these edges with a `provenance: "sitter"` metadata key so the
  MCP layer can distinguish them from the type-resolved Go edges.

Done when: `dex graph callers <fn>` on a Python or TS project returns
non-empty results sourced from the static graph (no ripgrep fallback);
results are flagged in the JSON output as `provenance: "sitter"` so
agents know the precision tier.

## 10. LSP read-side server (hover / semantic-goto / find-related)

[`docs/vision.md`](vision.md) scope cut #3. The single biggest leverage
move outside the Claude-Code-only loop: expose dex's primitives as an
LSP server so any LSP-aware editor (Neovim, VS Code, Zed, Helix) reads
the same live index.

Scope for v1 — read-side only, no completion, no edits:
- `textDocument/hover` → `view_summarize` result for the symbol under
  the cursor.
- `textDocument/definition` → exact `search_symbol` match.
- A custom `dex/findRelated` request → `RelatedChunks` (vector
  neighbours).
- Server runs as `dex lsp` over stdio; reuses the same `Store` /
  `Indexer` as `dex mcp`.

Done when: a fresh Neovim install with `dex lsp` configured shows a
summary-grounded hover on a Go function, jumps to definition via the
graph, and can request "related code" via a custom command — all
without any per-editor extension.

---

## Bonus: regression harness

Not in the top 10, but the moment any of these ship, we need a
golden-output retrieval harness:

- `testdata/queries.jsonl` — a few dozen `{repo, query, intent, expected_top_paths}` cases.
- A test that runs them against a known indexed fixture and asserts the
  expected paths appear in the top-k.
- Compare scores across the RRF / rerank legs to catch precision regressions.

Worth wiring up before #3 (FTS) and #4 (rerank) land.

## How to evaluate a change here

Same loop as `docs/guide-improvements.md`:

1. `mooncake task ci` should be green end-to-end before/after each change.
2. For retrieval-affecting items (#3, #4, #8, #9), spot-check the same
   handful of queries before and after — `dex ask . "<question>" --format=json`
   and diff the top-5 `suggested_reads`.
3. For pipeline items (#5, #6), force the failure mode (kill the chat
   endpoint mid-batch; spam ^C) and confirm the new behaviour.
4. Update [`LLM_GUIDE.md`](../LLM_GUIDE.md) via `dex guide` after any
   change that touches public API of `internal/store`, `internal/mcp`,
   or adds a new package.
