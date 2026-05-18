# Spec 02: Unified Context Router (`mcsearch_context`)

**Status:** ✅ Implemented (v1, 2026-05-18). Graph layer 1 (issue #4
phase 1, commit 765af69) is now wired in. Supported intents that
exercise the graph: `symbol_lookup`, `editing_context`, `architecture`,
`package_topology`. `callers` and `callees` still degrade — they need
`calls` edges which layer 1 does not emit.
**Effort:** S–M
**Value:** Adoption. Agents reach for `grep`/`Read` by default and
ignore the existing MCP tools unless explicitly instructed. The router
becomes the single entry point Claude calls first — it picks intent,
composes the right legs, and returns execution guidance in prose, not
"here's a list of chunks, you figure it out."

---

## Problem

Today's MCP surface is powerful but fragmented:

```
semantic_search
find_symbol
related_chunks
mcsearch_status
```

When graph tools land, that fragmentation gets worse: `find_callers`,
`find_callees`, `package_graph`, `code_graph_query`. Humans can
navigate this. Agents are worse at tool selection than people think,
so they fall back to grep/read loops and burn context rediscovering
the same topology over and over.

The fix is a **query planner for code understanding**, not another
search endpoint. The agent asks one question; the router picks a
strategy, runs the right legs, and returns a compact bundle plus
execution guidance.

---

## Wire shape

### Input (matches issue #5)

```json
{
  "project":  "/repo",
  "question": "where is filesystem event debouncing handled?",
  "intent":   "auto",
  "k":        8
}
```

`intent` accepts:

```
auto                 // default — let router decide
behavior_search      // "where is X handled" — conceptual
symbol_lookup        // exact identifier mentioned
callers              // "what calls X"
callees              // "what does X call"
architecture         // "how does X work overall"
package_topology     // package-level relations
editing_context      // "I want to edit X, what do I need"
```

### Output

```json
{
  "status":  "ok",
  "intent":  "behavior_search",
  "project": "/repo",
  "semantic_hits": [
    {
      "path": "internal/watch/watch.go",
      "start_line": 60, "end_line": 71,
      "score": 0.51,
      "reason": "markDirty"
    }
  ],
  "symbols": [
    {
      "qualified_name": "markDirty",
      "path": "internal/watch/watch.go",
      "start_line": 60, "end_line": 71,
      "kind": "func"
    }
  ],
  "graph": { "nodes": [], "edges": [] },
  "suggested_reads": [
    {
      "path": "internal/watch/watch.go",
      "start_line": 60, "end_line": 137,
      "reason": "semantic match + symbol agreement"
    }
  ],
  "next_action": "Read internal/watch/watch.go lines 60-137 to ground your answer.",
  "avoid": "Do not grep for the identifier; it is already located."
}
```

The two fields that matter most for agent compliance are
`next_action` and `avoid` — **prose**, not structured data. The issue
is explicit on this: "agents follow explicit execution guidance better
than generic APIs."

---

## Intent routing

`resolveIntent` picks a label using, in order:

1. Explicit `intent` field when valid and not `"auto"`.
2. Keyword regex on the question (`callers → callees → packages →
   architecture → editing` priority).
3. Identifier-shaped tokens in a short query → `symbol_lookup`.
4. Default: `behavior_search`.

Identifier detection runs three patterns in priority order:

```
(*Receiver).Method   // "(*Store).Search"
PascalCase           // "OpenWith" — length ≥ 3 to skip noise
snake_with_underscore
```

Spans of a matched qualified symbol mask out sub-token matches so
`(*Store).Search` doesn't also yield bare `Store` and `Search`.

---

## Lane composition per intent

| Intent             | Symbol lane            | Semantic lane | Graph lane (layer 1) |
|--------------------|------------------------|---------------|----------------------|
| `behavior_search`  | runs if id detected    | runs          | —                    |
| `symbol_lookup`    | runs                   | runs          | siblings via `has_method`/`has_field`/`embeds` |
| `callers`          | runs                   | runs          | **needs `calls` edges (deferred)** |
| `callees`          | runs                   | runs          | **needs `calls` edges (deferred)** |
| `architecture`     | runs if id detected    | runs          | package + top-level type/function rollup |
| `package_topology` | runs if id detected    | runs          | `imports` edges between pkgs in neighborhood |
| `editing_context`  | runs if id detected    | runs          | enclosing-type sibling neighborhood |

The graph lane is best-effort: it loads when graph data is present
(`mcsearch graph index <project>` has been run), populates
`ContextOutput.Graph` for the supported intents, and is skipped
otherwise. The `avoid` line tells the agent whether to nudge the user
toward `mcsearch graph index`. `callers`/`callees` always warn because
layer 1 doesn't emit `calls` edges regardless of whether the graph is
indexed.

---

## `suggested_reads` selection

Strategy by intent:

- **symbol_lookup / callers / callees**: prefer symbol-lane definition
  sites; one read per definition, capped at 2.
- **architecture / package_topology**: top 2–3 semantic hits across
  distinct files; widened to surrounding chunk extents.
- **behavior_search / editing_context**: top 2 semantic hits, biased
  toward paths that also appear in the symbol lane (cross-lane
  agreement bumps confidence).

---

## `next_action` shape

Always an imperative sentence with concrete paths/lines, **never** "do
more research." Examples (template, not literal):

- *symbol_lookup*: `"Read internal/store/store.go lines 1004-1031 to see the definition."`
- *editing_context*: `"Read internal/watch/watch.go lines 60-137 before editing — this is the primary site."`
- *callers* (graph deferred): `"Graph layer not available yet — start from internal/store/store.go (Search) and grep for callers."`
- *architecture*: `"Skim a.go lines 1-50; b.go lines 1-50 for the structural overview before editing."`
- empty results: `"Rephrase the question with concrete keywords or fall back to grep."`

---

## `avoid` shape

Strong claim when we have strong signal:

- Graph-deferred intent: `"Do not assume the symbols list is exhaustive — graph extraction is not yet wired in."`
- Symbol + semantic hits both present: `"Do not grep for the identifier; it is already located. Read the suggested ranges instead of opening whole files."`
- Symbol only: `"Do not grep for the identifier; it is already located."`
- Semantic only: `"Do not read entire files; the suggested ranges cover the relevant context."`

---

## Graceful degradation

| Failure mode               | Behavior                                          |
|----------------------------|---------------------------------------------------|
| `embed.ErrUnreachable`     | Symbol + graph lanes still run. If either produced hits, return `ok` with hint. If not, `embedding-service-unreachable` with `endpoint`. |
| No `ChatClient`            | Router never depends on chat. Summarize-style follow-ups are encouraged via `next_action` prose; agent decides. |
| Stale chunk index (>24h)   | `stale: true`, refresh hint, still returns results. |
| No chunk index             | `status: "no-index"`, hint references `mcsearch index`. |
| No graph indexed           | Graph lane skipped; supported intents get an `avoid` line nudging the user toward `mcsearch graph index`. |

---

## Adoption levers

1. **Tool description** on `mcsearch_context` is explicit: "PRIMARY
   ENTRY POINT … Call this BEFORE Grep/Glob/Read fan-out." Description
   on each leg (`semantic_search`, `find_symbol`, `related_chunks`,
   `summarize_path`) leads with "Prefer `mcsearch_context` …; use this
   directly only when …" so the model sees the routing intent from the
   tool list alone.
2. **CLAUDE.md snippet** (`docs/claude-md-snippet.md`) is a drop-in
   block that codifies the rule in the project's instruction file.
3. **CLI mirror** — `mcsearch context <path> "<question>"` gives a
   bash fallback when the MCP transport is unavailable.

---

## Files

- `internal/mcp/context.go` — handler, intent classifier, identifier
  detection, lane runners, fusion, prose builders.
- `internal/mcp/context_graph.go` — in-memory graph view loader and
  per-intent graph enrichment.
- `internal/mcp/context_test.go` — table-driven tests for intent,
  identifier extraction, suggested_reads selection, prose builders,
  and end-to-end with the fake embed server (including a graph-seeded
  symbol_lookup case).
- `cmd/mcsearch/main.go::cmdContext` — CLI mirror.
- `docs/claude-md-snippet.md` — CLAUDE.md drop-in.

---

## Graph integration

Today's wiring (commit on this branch):

- `internal/mcp/context_graph.go::loadGraphView` pulls every
  `graph_nodes` / `graph_edges` row via `store.GraphAllNodes` /
  `GraphAllEdges` and indexes them in memory by ID, name, qualified
  name, package, file path, and edge src/dst/kind. One load per
  request — at the current scale (~800 nodes for this repo, a few
  hundred KB) this is cheaper than building targeted SQL queries
  per call.
- `enrichGraph` dispatches on intent, walks edges, and populates
  `ContextOutput.Graph` with the relevant `{nodes, edges}` subset.

## Open questions for the next graph layer

- **Targeted SQL queries** — once node count outgrows a single load
  (10k+ symbols), `store_graph.go` needs `NodesByName`, `NodesByPackage`,
  `EdgesBySrc`, `EdgesByDst` index-backed methods. The in-memory view
  becomes a fallback for small repos only.

- **`calls` edges** — required to ungate `callers` / `callees`
  intents. Once they exist `enrichGraph` adds a case for them and
  `graphDeferredIntents` shrinks to empty.

- **Cross-lane scoring** — `suggested_reads` already biases toward
  semantic + symbol agreement; with call edges it should add
  "symbol is in caller neighborhood" as a third agreement signal.

- **Cache discipline** — when targeted queries land, the in-memory
  view can be removed or kept as an opt-in cache. No `mtime`-keyed
  invalidation needed yet because we load fresh per request.
