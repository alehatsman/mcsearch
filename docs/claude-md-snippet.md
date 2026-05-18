# CLAUDE.md snippet — wiring `mcsearch_context` into agent workflow

Drop the block below into a project's `CLAUDE.md` (or
`~/.claude/CLAUDE.md` for cross-project defaults) to make agents call
`mcsearch_context` before grep/Read loops.

The point is *forced adoption*: agents will reach for `grep` first
unless explicitly told otherwise, because grep is in their muscle
memory. The snippet routes them to the MCP tool whenever the request
shape matches — and falls back to grep cleanly when the index is
missing or the embedding service is offline.

---

## Full version

```markdown
# Repository understanding — tool routing

Before any Grep / Glob / Read fan-out on a code-understanding question,
call **`mcsearch_context`**. It is the single planner: it picks intent,
runs semantic search + symbol lookup (and graph queries when available),
and returns `suggested_reads`, a prose `next_action`, and an `avoid`
line you can follow verbatim.

**Inputs:**
- `project` — absolute path to the repo root (current dir if omitted).
- `question` — free text; required. ("where is auth validation
  implemented", "callers of (*Store).Search", "how does indexing work")
- `intent` — optional override:
  `auto | behavior_search | symbol_lookup | callers | callees |
   architecture | package_topology | editing_context`
- `k` — optional cap on hits per lane (default 8).

**What you get back:**
- `semantic_hits` — top semantic chunks (path + line range + score).
- `symbols` — exact-identifier hits with kind and location.
- `graph` — nodes/edges from the graph layer (empty until that lands).
- `suggested_reads` — file ranges to open in full. Prefer these over
  reading entire files.
- `next_action` — **prose** directive. Execute it as-written.
- `avoid` — what NOT to do. Honor it.

**Fallback rules:**
- `status: "no-index"` → run `mcsearch index <project>` once, or fall
  back to Grep if you can't.
- `status: "embedding-service-unreachable"` → embed is offline; fall
  back to Grep / Glob / ripgrep for this request.
- `stale: true` → results may be ~1 day behind HEAD; flag this if the
  fix depends on very recent code.
- `avoid` mentions "Run `mcsearch graph index`" → the project has no
  structural graph yet; symbol/architecture intents work but won't
  surface sibling methods, package imports, etc. Suggest indexing
  once.
- `avoid` mentions "`calls` edges are not yet extracted" → call-graph
  queries (callers/callees) fall back to grep on the symbol name.

**When NOT to call mcsearch_context:**
- You already have an exact file path and need to read it — use `Read`.
- You're hunting an exact literal (error message, magic number) — use
  Grep. Semantic search wins on intent; grep wins on exact strings.
- You're editing — use `Edit`.
```

---

## Short version

If your CLAUDE.md is already dense:

```markdown
# Tool routing
- Code understanding (where / how / callers / architecture / edit):
  **`mcsearch_context`** first, then follow its `next_action` and
  honor `avoid`.
- Exact strings / literals: `Grep`.
- Known path: `Read`.
- Edits: `Edit`.
```

---

## CLI fallback

The same router is available as `mcsearch context <path> <question>`
for shell-based agents or when the MCP transport is unavailable:

```sh
mcsearch context . "where is filesystem event debouncing handled"
mcsearch context . "callers of (*Store).Search"
mcsearch context . "how does indexing pipeline work" --intent architecture
mcsearch context . "..." --format=json   # raw output for piping
mcsearch context . "..." --k 12          # widen per-lane hits
```

Flags mirror the MCP input fields: `--intent`, `--k`, `--format`.

---

## Why this works

The model responds strongly to usage guidance embedded in tool
descriptions and instruction files — much more than to clever tool
APIs. Three reinforcing layers:

1. **Tool descriptions**: `mcsearch_context` is labeled "PRIMARY
   ENTRY POINT" and "Call this BEFORE Grep/Glob/Read fan-out." Each
   leg (`semantic_search`, `find_symbol`, `related_chunks`,
   `summarize_path`) begins with "Prefer `mcsearch_context` …; use
   this directly only when …".
2. **CLAUDE.md** (this snippet): codifies the rule in the
   project's instruction file Claude actually reads.
3. **Prose `next_action` / `avoid`**: every router response carries
   an imperative sentence the agent can execute verbatim, plus a
   "don't do that" line. Structured args lose to prose for agent
   compliance.
