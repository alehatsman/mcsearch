# mcsearch Storage Architecture (`internal/store`)

## One SQLite file per project

`$MCSEARCH_INDEX_DIR/<sha256(realpath(project_root))>/index.db`. Driver: `mattn/go-sqlite3` with `sqlite_fts5`; sqlite-vec is statically linked via `asg017/sqlite-vec-go-bindings/cgo` and **auto-registered globally** in `init()` (`store.go:34-38`) so every connection has `vec0` and `vec_distance_cosine()` available.

Connection settings (`store.go:92-98`): WAL journal, `synchronous=NORMAL`, `busy_timeout=5000` (so concurrent `mcsearch index` + `mcsearch watch` don't crash on writer lock), foreign keys on.

## Schema (set up in `migrate`, `store.go:132`)

| Table | Type | Role |
|---|---|---|
| `meta(key, value)` | regular | `dim`, `last_indexed_at` (ns), `project_root` |
| `chunks` | regular | id PK, path, kind, name, start_line, end_line, content_sha1, content, **vec BLOB**, last_seen_at; **UNIQUE(path, content_sha1)** |
| `chunks_fts` | FTS5 virtual | external-content (`content='chunks', content_rowid='id'`) — no duplicate text on disk, only tokenizer state (`unicode61 remove_diacritics 2`) |
| `chunk_vecs` | sqlite-vec `vec0` virtual | `embedding FLOAT[dim] distance_metric=cosine` — KNN index |
| `graph_nodes` | regular | TEXT PK (stable across re-extraction), kind, name, qualified_name, package_path, file_path, lines, `chunk_id` (loose join to chunks.id; no FK because chunks rowids churn), metadata_json, content_hash, last_seen_at |
| `graph_edges` | regular | TEXT PK, kind, src_id, dst_id, file_path, lines, metadata_json, content_hash, last_seen_at |
| `pending_summaries` | regular | deferred summarization queue; UNIQUE(path, kind, content_sha1) makes Enqueue idempotent |

Indexes: `idx_chunks_path`, `idx_chunks_last_seen`; `idx_graph_nodes_{kind,name,package,file,last_seen}`; `idx_graph_edges_{src,dst,last_seen}`; `idx_pending_summaries_queued`.

## Triggers keep both virtual tables in sync

Three pairs on `chunks`, all created in `migrate`/`ensureVecTable`:

```sql
-- chunks → chunks_fts (FTS5)
chunks_ai AFTER INSERT  → INSERT into chunks_fts
chunks_ad AFTER DELETE  → 'delete' INSERT into chunks_fts (FTS5 tombstone)
chunks_au AFTER UPDATE  → delete + reinsert

-- chunks → chunk_vecs (sqlite-vec)
chunks_vec_ai AFTER INSERT          → INSERT (rowid, embedding)
chunks_vec_ad AFTER DELETE          → DELETE WHERE rowid=old.id
chunks_vec_au AFTER UPDATE OF vec   → delete + reinsert
```

`chunks.vec` is kept as the canonical packed-LE float32 BLOB so `chunk_vecs` is rebuildable and `vec_distance_cosine()` can score BM25-only hits cheaply.

## Lazy vec table + one-shot backfill

`Store.dim` is an `atomic.Int64`, set once: either recovered from `meta.dim` on Open, or set on the first `UpsertMany` (`store.go:519-528`). Until dim is known, `ensureVecTable` is a no-op (`store.go:323-327`). After that:

1. Create `chunk_vecs` with the known dim and the three triggers.
2. **Backfill for pre-vec0 indexes**: if `chunk_vecs` is empty but `chunks` is not, run one `INSERT INTO chunk_vecs(rowid, embedding) SELECT id, vec FROM chunks` (`store.go:365-369`). Idempotent — a populated table skips it.

**Dim is fixed for the life of the index.** `UpsertMany` rejects vectors with `len != dim` (`store.go:553-555`); changing the embedding model requires `mcsearch reindex`.

## Write path — `UpsertMany` (`store.go:508-567`)

One transaction per batch (≈32× fewer fsyncs than per-chunk). Prepared `INSERT … ON CONFLICT(path, content_sha1) DO UPDATE` so re-indexing the same chunk is idempotent. The three trigger pairs do the FTS/vec maintenance automatically — callers never touch `chunks_fts` or `chunk_vecs` directly.

Companion mutators on `chunks`:
- `TouchSeen` — fast-path bump of `last_seen_at` (+ name backfill) without re-embedding.
- `TouchPath` — bulk version for the mtime fast-path: `UPDATE chunks SET last_seen_at=? WHERE path=?`.
- `PruneUnseen(cutoff)` — `DELETE … WHERE last_seen_at < ?` to drop chunks belonging to vanished files. **Timestamps are Unix nanoseconds** (`store.go:9-13`) — millisecond resolution would let two index runs in the same ms collide on the strict-less-than cutoff.
- `DeletePath`, `DeletePathPrefix` — explicit removal.

## Read path — `Search` (`store.go:851-988`)

`Store.Search` → `searchRaw` (the inner hybrid worker; `Search` adds optional `MaxHitsPerFile` diversification via `diversify`).

```
fused pool = max(5·k, 30)   (capped by Options.RerankPool if Reranker != nil)

scoreSemantic(queryVec, pool)   ─► sqlite-vec KNN: SELECT rowid, distance
                                   FROM chunk_vecs WHERE embedding MATCH :blob AND k=:pool
                                   ORDER BY distance     (similarity = 1 - distance)

scoreBM25(queryText, pool)      ─► FTS5: bm25(chunks_fts) over buildFTSQuery(queryText)
                                   (FTS5 parse errors fall back to semantic-only,
                                    never surface to caller)

RRF: rrf[id] = Σ 1 / (rrfK + rank_in_list)    rrfK = 60 (Cormack 2009)

BM25-only fused IDs get cosine backfilled via scoreSemanticForIDs
   (so Hit.Score is populated for every result)

if Reranker wired && len(fused) > k:
   rerank top pool with cross-encoder; on rerank.ErrUnreachable, silent fall-through
else:
   take fused[:k]

fetchHits → SELECT path, kind, name, … FROM chunks WHERE id IN (…), zipped with
            scoreContext{semCosine, bm25Score, rrfScore, rerankScore}
```

Disable knob: `Options.DisableBM25` (or empty query text) → semantic-only path that skips BM25 and RRF entirely.

## Symbol & related lookups

- **`FindSymbol(name, k)`** (`store.go:1303`) — exact match against `chunks.name`; **falls back to `findSymbolInGraph`** if no chunk row matches, which queries `graph_nodes` and synthesizes a `Hit` from `chunk_id` linkage.
- **`FindSymbolCandidates`** — `LIKE %name%` for autocorrect suggestions.
- **`RelatedChunks(path, line, k)`** — same `vec0 MATCH` query using the source chunk's BLOB at `k+1`, then drops the self-hit.

## Graph methods (`store_graph.go`)

Separate file. `GraphUpsertNodes`, `GraphUpsertEdges` are `INSERT … ON CONFLICT(id) DO UPDATE` keyed on the **string id**, so re-extraction doesn't churn rowids and `chunk_id` linkage stays meaningful. `GraphPruneUnseen(cutoff)` deletes by `last_seen_at`, same pattern as `chunks`. `GraphSetCentrality` writes PageRank / degree results computed by `internal/graph.ComputeCentrality`.

## Summary queue (`store_pending.go`)

Backs `DeferSummaries` mode in the chunk indexer. Enqueue is idempotent via the UNIQUE(path, kind, content_sha1). Drained by `mcsearch index summarize` and by watch idle ticks.

## Layout at a glance

```
                     ┌─────────────────────────────────────────────┐
 UpsertMany ────────►│ chunks (id, path, content_sha1, vec, …)     │
 TouchSeen/TouchPath │   UNIQUE(path, content_sha1)                │
 PruneUnseen         └──┬───────────────────────┬──────────────────┘
                        │ triggers              │ triggers
                        ▼                       ▼
                 chunks_fts (FTS5)        chunk_vecs (vec0)
                        ▲                       ▲
 scoreBM25 ─────────────┘                       └──── scoreSemantic / RelatedChunks
                                                            │
                                         RRF fuse + (opt.) rerank
                                                            ▼
                                                      Search → Hits

 graph_nodes (id TEXT) ──chunk_id──► chunks.id        pending_summaries (queue)
 graph_edges (src_id,dst_id TEXT)                     meta (dim, last_indexed_at, root)
```
