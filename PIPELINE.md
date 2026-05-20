# mcsearch Data Pipeline Architecture

mcsearch is a local semantic code-search service. The pipeline has **three indexers writing to one SQLite per project** (keyed by `sha256(realpath(root))` at `$MCSEARCH_INDEX_DIR/<hash>/index.db`).

## Three indexers, one DB

| Indexer | Package | Output |
|---|---|---|
| **Chunk indexer** | `internal/index` | `chunks` (+ `chunk_vecs` vec0, `chunks_fts` FTS5) — the semantic + lexical search corpus |
| **Graph indexer** | `internal/graph` | `graph_nodes`, `graph_edges` — Go static call graph + YAML, with PageRank centrality |
| **Watcher** | `internal/watch` | fsnotify → debounce → re-runs the two above |

## Chunk pipeline (`Indexer.Run`, `internal/index/index.go:117`)

Comment at top of the file is literal: *"walk → chunk → embed → upsert"*. Six passes:

1. **Pass 1 — walk + chunk** (`internal/index/index.go:1` header). Single-threaded directory walk; per-file work (read, `ignore.Match`/binary/secret heuristics from `internal/ignore`, tree-sitter parse in `internal/chunk`) runs on `Options.Concurrency` workers (defaults to `GOMAXPROCS`).
   - **Mtime fast-path** — file mtime ≤ last index run → `UPDATE last_seen_at` only.
   - **SHA fast-path** — content unchanged → bump `last_seen_at`, backfill `name`, no embed.
   - **Slow path** — surviving files become `slowFile{rel, data, chunks}` for Pass 2.
2. **Pass 2 — embed + upsert**. Batches go to `internal/embed` (OpenAI-compatible `/v1/embeddings`, e.g. vLLM/TEI Qwen3). Result rows go through `store.UpsertMany`; triggers keep `chunk_vecs` (sqlite-vec) and `chunks_fts` in sync (`docs/internals.md:28-37`).
3. **Pass 3 — per-chunk summaries** (optional, `Options.Summarize`). Calls `internal/chat` per non-tiny chunk. With `DeferSummaries=true`, just enqueues `pending_summaries` rows.
4. **Pass 4 — prune unseen**. `PruneUnseen` deletes rows whose `last_seen_at < startTime` (files removed since the run started).
5. **Pass 5 — package summary**. For each directory, summarize from its file-summary chunks.
6. **Pass 6 — repo summary** (`internal/index/index.go:794`). One `path="."` summary built from all package summaries; protected from pruning.

## Graph pipeline (`internal/graph/graph.go:254`)

Independent of chunks; runs after them in `cmd/mcsearch/main.go:cmdIndex`. `ExtractGo` (go/types) + `ExtractYAML` → `linkChunks` joins nodes to their chunk rows → `GraphUpsertNodes`/`GraphUpsertEdges` → `GraphPruneUnseen` → `ComputeCentrality` (PageRank + in/out degree + cross-pkg callers) → `GraphSetCentrality`. Skippable via `--graph=off`; `--graph=only` skips chunk passes.

## Retrieval (read side)

`store.Search` runs **two rankers in parallel and fuses with RRF** (`docs/internals.md:61-87`):
- **cosine** — `SELECT … FROM chunk_vecs WHERE embedding MATCH :blob AND k=:pool` (sqlite-vec); query is embedded via the same `/v1/embeddings` endpoint.
- **BM25** — `bm25()` against `chunks_fts`.
- final score `Σ 1/(60 + rank)`; optional cross-encoder rerank (`MCSEARCH_RERANK_URL`).

`internal/mcp` wraps this for the MCP tool surface (`ask`, `search_semantic`, `search_symbol`, `graph_*`, `view_summarize`). `cmd/mcsearch/main.go` provides the CLI mirrors and the MCP stdio server entrypoint.

## Live updates (`internal/watch/watch.go`)

`Watcher.Run`: fsnotify subscribes to the project tree → events filtered through the same `ignore.Matcher` → debounced (`Options.Debounce`, default 500ms) → dirty set drained by re-invoking `Indexer.Run` → `AfterIndex` hook re-runs the graph phase. Used by `mcsearch watch` (`cmd/mcsearch/main.go:1512`).

## Flow at a glance

```
files ──► walk (ignore) ──► chunk (tree-sitter) ──► embed (HTTP) ──► chunks/chunk_vecs/chunks_fts
                                                  │
                                                  └─► (opt.) chat summaries ──► file_summary → package_summary → repo_summary
files ──► ExtractGo/YAML ──► linkChunks ──► graph_nodes/graph_edges ──► PageRank → centrality

query ──► embed ──► chunk_vecs (cosine)  ┐
query ──► tokens ──► chunks_fts (BM25)   ├─► RRF ──► (opt.) rerank ──► hits
```
