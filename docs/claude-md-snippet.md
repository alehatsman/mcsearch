# CLAUDE.md snippet — wiring `ask` into agent workflow

Drop the block below into a project's `CLAUDE.md` (or
`~/.claude/CLAUDE.md` for cross-project defaults) to make agents call
`ask` before grep/Read loops.

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
call **`ask`**. It is the single planner: it picks intent, runs
semantic search + symbol lookup + graph expansion (including `calls`
edges for Go), and returns `suggested_reads`, a prose `next_action`,
and an `avoid` line you can follow verbatim.

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
- `graph` — nodes/edges from the static graph (includes `calls` for Go).
- `references` — call-site usages: precise for Go (from `calls`
  edges), ripgrep-backed for other languages.
- `suggested_reads` — file ranges to open in full. Prefer these over
  reading entire files.
- `next_action` — **prose** directive. Execute it as-written.
- `avoid` — what NOT to do. Honor it.

**Fallback rules:**
- `status: "no-index"` → run `dex index <project>` once, or fall
  back to Grep if you can't.
- `status: "embedding-service-unreachable"` → embed is offline; fall
  back to Grep / Glob / ripgrep for this request.
- `stale: true` → results may be ~1 day behind HEAD; flag this if the
  fix depends on very recent code.
- `avoid` mentions "Run `dex index`" → the project has no
  structural graph yet (likely indexed by an older binary that didn't
  run the graph phase, or re-indexed with `--graph=off`); symbol/
  architecture intents work but won't surface sibling methods, package
  imports, etc. Suggest re-running `dex index <project>` once.
- `avoid` mentions "`calls` edges are Go-only" → call-graph queries
  for non-Go languages fall back to a ripgrep `references` list;
  treat that list as best-effort and verify edge cases.

**When NOT to call ask:**
- You already have an exact file path and need to read it — use `Read`.
- You're hunting an exact literal (error message, magic number) — use
  Grep. Semantic search wins on intent; grep wins on exact strings.
- You're editing — use `Edit`.

**Sister tools** (call directly only when you already know the leg you want):
- `search_semantic`, `search_symbol` — raw retrieval legs.
- `graph_neighbors` — cosine neighbours of a known chunk.
- `graph_deps` — `imports` for a file or package.
- `graph_callers`, `graph_callees` — precise call edges (Go-only).
- `view_summarize` — chat-model file/range gist.
- `index_status` — endpoint health + indexed projects.
```

---

## Short version

If your CLAUDE.md is already dense:

```markdown
# Tool routing
- Code understanding (where / how / callers / architecture / edit):
  **`ask`** first, then follow its `next_action` and honor `avoid`.
- Exact strings / literals: `Grep`.
- Known path: `Read`.
- Edits: `Edit`.
```

---

## CLI fallback

The same router is available as `dex ask <path> <question>` for
shell-based agents or when the MCP transport is unavailable:

```sh
dex ask . "where is filesystem event debouncing handled"
dex ask . "callers of (*Store).Search"
dex ask . "how does indexing pipeline work" --intent architecture
dex ask . "..." --format=json   # raw output for piping
dex ask . "..." --k 12          # widen per-lane hits
```

Flags mirror the MCP input fields: `--intent`, `--k`, `--format`.

---

## Why this works

The model responds strongly to usage guidance embedded in tool
descriptions and instruction files — much more than to clever tool
APIs. Three reinforcing layers:

1. **Tool descriptions**: `ask` is labeled "PRIMARY ENTRY POINT" and
   "Call this BEFORE Grep/Glob/Read fan-out." Each leg
   (`search_semantic`, `search_symbol`, `graph_neighbors`,
   `graph_deps`, `graph_callers`, `graph_callees`, `view_summarize`)
   begins with "Prefer `ask` …; use this directly only when …".
2. **CLAUDE.md** (this snippet): codifies the rule in the
   project's instruction file Claude actually reads.
3. **Prose `next_action` / `avoid`**: every router response carries
   an imperative sentence the agent can execute verbatim, plus a
   "don't do that" line. Structured args lose to prose for agent
   compliance.
