# How LLM_GUIDE.txt is generated

Two-phase model. The LLM work happens at **index time**, not guide time.

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
overview.

The chat client is OpenAI-compatible (`internal/chat/client.go`) —
Ollama, vLLM, anything that speaks `/v1/chat/completions`. Default for
summary work is `qwen2.5-coder:7b` via `DEX_SUMMARY_URL`.

**Cache keys**: each summary is stored with a deterministic SHA over
its inputs (file SHAs for package, package contents for repo).
Re-indexing with no changes → cache hits → no LLM calls.

## Phase 2: dex guide renders

`dex guide .` does **zero LLM calls**. The flow lives in
`internal/guide/render.go`:

1. `SELECT path, content, last_seen_at FROM chunks WHERE kind='repo_summary'`
2. `SELECT path, content, last_seen_at FROM chunks WHERE kind='package_summary' ORDER BY path`
3. Read `.dex/llm_guide.manifest.json` — get `last_summary_seen_at`.
4. Dirty check: any summary chunk's `last_seen_at` greater than the
   manifest's recorded value?
5. If clean → exit. If dirty → format markdown, write `LLM_GUIDE.txt`,
   update manifest.

Markdown is mechanical concatenation in `buildMarkdown`
(`internal/guide/render.go:90`):

```
# Project Guide
## Overview          ← repo_summary.content
## Module: <path>    ← package_summary[i].content
...
```

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

## Pre-commit chain

```
dex index . --summarize    # fast path on unchanged files; LLM only on touched
dex guide .                # format → LLM_GUIDE.txt + manifest
```

First half does the (potentially) slow work; second half is always
near-instant. See `scripts/pre-commit-guide.sh` for an installable
hook.
