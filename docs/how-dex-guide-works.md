# How LLM_GUIDE.md is generated

Two-phase model. The LLM work happens at **index time**, not guide
time. Each module section in the output is a hybrid: an LLM-generated
prose paragraph for narrative flavor, followed by mechanical
ground-truth data pulled from the static call graph (no LLM,
hallucination-proof).

## Phase 1: index produces summary chunks

`dex index <path> --summarize --summarize-defer=false` runs six passes
(`internal/index/index.go`):

| Pass | What | Produces |
|---|---|---|
| 1 | walk + tree-sitter chunk | raw code chunks |
| 2 | embed + upsert | `chunks` rows + sqlite-vec + FTS5 |
| 3 | per-chunk summary | `chunk_summary` chunks |
| 4 | prune unseen | drops stale rows |
| **5** | **per-package summary** | **`package_summary` chunks (one per directory)** |
| **6** | **repo summary** | **`repo_summary` chunk at path="."** |

Pass 5 (`summarizePackage`, `internal/index/index.go:921`) feeds the
`file_summary` content of every file in a directory into the chat
model, gets back a package-level overview.

Pass 6 (`summarizeRepo`, `internal/index/index.go:947`) feeds all
`package_summary` rows into the chat model, gets back the repo
overview. Capped at 1200 tokens with `FinishReason=length` treated as
an error so a truncated overview never replaces a good one.

The chat client is OpenAI-compatible (`internal/chat/client.go`) —
Ollama, vLLM, anything that speaks `/v1/chat/completions`. Default for
summary work is `qwen2.5-coder:7b` via `DEX_SUMMARY_URL`.

**Cache keys**: each summary is stored with a deterministic SHA over
its inputs (file SHAs for package, package contents for repo).
Re-indexing with no changes → cache hits → no LLM calls.

## Phase 2: dex guide renders

`dex guide .` does **zero LLM calls**. The flow lives in
`internal/guide/render.go`:

1. Load `repo_summary` + `package_summary` chunks via
   `SummariesByKindWithMeta` (path, content, last_seen_at).
2. Read `.dex/llm_guide.manifest.json` — get `last_summary_seen_at`.
3. Dirty check: any summary chunk's `last_seen_at` greater than the
   manifest's recorded value, OR the guide file is missing, OR `--full`
   was passed.
4. If clean → exit. If dirty → format markdown, write `LLM_GUIDE.md`,
   update manifest.

## Output shape

Each module section combines LLM prose with graph-grounded data:

```
## Module: <path>

<package_summary content from LLM>          ← narrative

**Exported API** (N)                        ← from graph_nodes
- `func` Name — file:line
- `method` Type.Name — file:line
...

**Key entry points** (top 5 by PageRank)    ← from graph_nodes.pagerank
- `Name` — file:line — in-degree N
...

**Depends on**                              ← from graph_nodes (kind=import)
- project: internal/foo, internal/bar
- external: context, fmt, github.com/...

**Used by**                                 ← reverse import edges
- cmd/dex, internal/mcp
```

### Section sources

| Section | Source | Filter |
|---|---|---|
| Exported API | `graph_nodes` kind ∈ {function, method, struct, interface, type} | name starts with capital |
| Key entry points | `graph_nodes` kind ∈ {function, method}, ORDER BY pagerank DESC | exported preferred; falls back to internal hot spots (with a visible heading change) when no exported nodes have centrality |
| Depends on | `graph_nodes` kind=import, scoped to the directory's Go package paths | split into project (matches go.mod module prefix) vs external (stdlib + third-party) |
| Used by | inverse of Depends on — packages whose import nodes name this module's package paths | strips module prefix to display directories |

### Quirks

- `file_path` is empty on `kind='import'` rows (imports are a
  package-level fact, not per-file). Queries resolve via `package_path`
  for these rows; only declaration nodes (`function`, `method`, etc.)
  carry `file_path`.
- Non-Go directories (`testdata/`, `scripts/`, `docs/`) get only the
  LLM prose section — graph queries return empty and each subsection
  is omitted gracefully.
- The renderer reads `go.mod` once per render to discover the module
  prefix used to split project vs. external imports.

## Why the split

Splitting "produce summaries" from "format guide" gives:

- **Cheap re-renders.** The guide can re-run instantly because nothing
  about formatting needs an LLM.
- **Incremental.** Only changed files re-summarize during `dex index`
  (mtime + content_sha1 fast paths). The guide notices via
  `max(last_seen_at)` ticking forward.
- **Reusable summaries.** The same `package_summary` chunks already
  power `view summarize`, `ask`'s suggested-reads, and MCP context
  routing — the guide is a new consumer, not a new producer.
- **Hallucination resistance.** LLM prose carries the narrative; graph
  data carries the facts. If they disagree, the facts are the source
  of truth and a reader can see both.

## Pre-commit chain

```
dex index . --summarize    # fast path on unchanged files; LLM only on touched
dex guide .                # format → LLM_GUIDE.md + manifest
```

First half does the (potentially) slow work; second half is always
near-instant. See `scripts/pre-commit-guide.sh` for an installable
hook.

## Configuration

`.dex/guide.toml` (optional):

```toml
[guide]
output = "LLM_GUIDE.md"
```

Missing file → defaults. There is no `[ollama]` block — the renderer
makes no LLM calls. Summarization itself is configured via
`DEX_SUMMARY_URL` / `DEX_SUMMARY_MODEL` environment variables, the
same as the rest of the index pipeline.
