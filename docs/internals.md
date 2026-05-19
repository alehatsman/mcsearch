# Internals

Technical details that don't belong on the README's front page. The
[README](../README.md) covers the user-facing surface; this file covers
storage, retrieval math, the vector cache, the multi-worktree workflow,
and the embedding contract.

## Storage

One SQLite per project at `$MCSEARCH_INDEX_DIR/<sha256(realpath(project_root))>/index.db`:

```
meta(key, value)                                                              -- dim, last_indexed_at, project_root
chunks(id, path, kind, name, start_line, end_line, content_sha1, content,
       vec BLOB, last_seen_at)                                                -- UNIQUE(path, content_sha1)
graph_nodes(id, kind, name, qualified_name, package_path, file_path,
            start_line, end_line, chunk_id, metadata_json, content_hash,
            last_seen_at)                                                     -- Go static graph
graph_edges(id, kind, src_id, dst_id, file_path, start_line, end_line,
            metadata_json, content_hash, last_seen_at)
```

Vectors are packed `float32` BLOBs. A virtual `chunks_fts` table mirrors
`content` for FTS5/BM25 lookups, kept in sync via AFTER triggers so
upsert cost is unchanged. `graph_*` tables are written by the graph
phase of `mcsearch index` (skipped only with `--graph=off`); chunk-side
code never reads them. `last_seen_at` is Unix nanoseconds so two index
runs in the same millisecond still prune correctly.

## Incremental re-index

`mcsearch index` is safe to re-run. Three fast-paths:

| Fast-path | Condition | Cost |
|---|---|---|
| **Mtime** | File mtime ≤ last index run | One `UPDATE last_seen_at` per file — no read, no parse, no embed |
| **SHA** | File changed but chunk content unchanged | Re-parse + SHA, then `UPDATE last_seen_at, name` — no embed call |
| **Full** | New or changed chunk | Parse + embed + upsert |

The SHA fast-path also backfills the `name` column on unchanged chunks,
so upgrading to a binary with identifier extraction (used by
`find_symbol`) doesn't need a full `reindex` — the next ordinary `index`
populates names for free. Changing the embedding model (different
vector dim) does require `mcsearch reindex <path>`; mixed dims are
rejected at upsert.

## Hybrid retrieval — semantic + BM25 + optional rerank

Every `Search` runs two rankers and fuses them via Reciprocal Rank
Fusion (Cormack et al., 2009):

- cosine path scores every chunk against the embedded query vector;
- BM25 path runs literal query tokens against `chunks_fts` via
  SQLite's `bm25()`;
- final score is `Σ 1/(60 + rank_in_list)` summed across whichever
  lists the chunk appeared in.

Semantic alone catches paraphrase ("debounce filesystem events") but
misses rare literal tokens (`MCSEARCH_DISABLE_VEC_CACHE`,
`compileDoubleStar`). BM25 alone is the inverse failure. RRF is
scale-free — no per-corpus tuning. Set `MCSEARCH_DISABLE_BM25=1` (or
pass an empty query text) to get pre-hybrid semantic ranking.

Hits expose `score` (cosine, for human comparability), `bm25_score`
(when surfaced via the lexical leg), and `rrf_score` (fused, used for
ordering).

**Cross-encoder rerank** is off by default. Set `MCSEARCH_RERANK_URL`
to enable; design and migration notes live in
[specs/spec-01-rerank.md](specs/spec-01-rerank.md). Per-call opt-out:
`mcsearch query --rerank=off`. Process-wide off:
`MCSEARCH_DISABLE_RERANK=1`. Reranker outages never break search —
on unreachable, results fall back to the pre-rerank fused order silently.

## Vector cache

On first `Search`, every chunk vector is decoded once into a flat
`[]float32` slab + precomputed `|v|` norms. Subsequent queries score
against the slab with zero hot-path allocations and one small `SELECT`
to fetch content for the top-k IDs. Mutating operations
(`UpsertMany`, `DeletePath`, `DeletePathPrefix`, `PruneUnseen`)
invalidate the slab.

Brute-force cosine post-cache, measured on a Ryzen 9 9950X:

| Chunks | Dim  | Search (top-k=8) |
|-------:|-----:|-----------------:|
|   1 k  |   16 | 0.1 ms           |
|   5 k  | 1024 | 2.7 ms           |
|  20 k  | 1024 | 12 ms            |
| 100 k* | 1024 | ~60 ms           |
| 100 k* | 2560 | ~150 ms          |
| 200 k* | 2560 | ~300 ms          |

(* extrapolated from measured rows — see `internal/store/bench_test.go`.)

At realistic project sizes (<50 k chunks) search is never the
bottleneck — per-query embed round-trip dominates. The ceiling is
**RAM**: slab is `chunks × dim × 4 B`, so 100 k chunks at 2560 dim is
~1 GB. Memory-constrained: `MCSEARCH_DISABLE_VEC_CACHE=1` (slower
per-row SQL path, bounded RAM). Past ~500 k chunks or sub-50 ms p99
goals, swap in a real ANN index (HNSW via `coder/hnsw`, `sqlite-vec`,
LanceDB) — the rest of the store stays unchanged.

## Multi-worktree workflow

Indexes are keyed by `sha256(realpath(project_root))`, so
`git worktree add ../proj-feature` looks like a brand-new project even
though the trees are nearly identical. `mcsearch clone` seeds the new
worktree's index from a sibling (a `cp` of one SQLite file, ~5 ms);
chunks are keyed by `(relative path, content sha1)`, so anything
unchanged between trees rides along for free.

```console
$ mcsearch clone . /tmp/mcsearch-feature
✓ cloned /home/aleh/projects/mcsearch → /tmp/mcsearch-feature

$ mcsearch index -v /tmp/mcsearch-feature
INFO msg=indexed chunks_seen=467 files_fast_path=31 embedded=12 pruned=0
```

The two indexes are independent after the clone. `--force` overwrites
an existing destination; `mcsearch nuke <dst>` deletes it.

## Embedding contract

OpenAI-compatible `/v1/embeddings`:

```json
POST /v1/embeddings
{ "model": "Qwen/Qwen3-Embedding-4B", "input": ["chunk-text-1", "chunk-text-2"] }
```

Both vLLM (`vllm serve … --task embed`) and TEI's compatibility shim
implement this. Vector dimension is discovered on the first call and
recorded on the project; mixed dimensions across re-indexes are
rejected at upsert time.

## Offline behavior

Endpoint unreachable: `mcsearch query` exits non-zero with an
informative error. The MCP `semantic_search` tool returns
`{ "status": "embedding-service-unreachable", ... }` so Claude can
fall back to grep without crashing.

## Code generation

`mcsearch generate <path> "<prompt>"` reuses the hybrid retrieval as
`query`, prepends the top-k chunks as a `CONTEXT` block, and sends the
result to `MCSEARCH_CHAT_URL`. Flags: `-k`, `--no-rag`, `--system`,
`--temperature`, `--max-tokens`, `--show-context`. Mid-size local
chat models (≤32B) tend to generate from training data rather than
strictly from `CONTEXT` — use `semantic_search` for ground-truth
retrieval; treat generated output as a starting point.
