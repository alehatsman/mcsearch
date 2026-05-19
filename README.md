# mcsearch

mcsearch — a local code-search MCP server for Claude (and a CLI).

It pre-indexes a project into a SQLite vector store and exposes it to Claude over MCP, so the agent can ask conceptual questions ("where is filesystem event debouncing handled?") and get back the exact function/type/doc block — instead of grepping and reading files to orient.

  How it works:
  - Tree-sitter parses source into named chunks (functions, methods, types).
  - Each chunk is embedded via a self-hosted /v1/embeddings endpoint (ollama / vLLM / TEI, local or SSH-tunneled).
  Source never leaves your machine.
  - Retrieval fuses cosine similarity + BM25 via RRF, with optional cross-encoder rerank.

```console
$ mcsearch query ./ "where is filesystem event debouncing handled"
─── #1 markDirty  internal/watch/watch.go:60-71  (method_declaration)
// markDirty resets the debounce timer; on expiry it runs an index pass.
func (w *Watcher) markDirty() { … }
```

**How it works:** tree-sitter parses your source into named chunks
(functions, methods, types, classes). Each chunk is embedded via a
self-hosted `/v1/embeddings` endpoint you control — a local GPU running
ollama, vLLM, or TEI; or an SSH-tunneled remote. At query time, cosine
similarity and BM25 are fused via RRF, then optionally reranked by a
cross-encoder. Source code never leaves your machine.

## Primary entry point: `mcsearch_context`

The headline tool is **`mcsearch_context`** — a query planner for code
understanding. The agent (or you, from the shell) asks one free-text
question; the router picks an intent, composes the right legs
(`semantic_search` + `find_symbol` + graph expansion), and returns a
compact bundle:

- `semantic_hits` — top semantic chunks (path + line range + score).
- `symbols` — exact-identifier hits with kind and location.
- `graph` — nodes/edges from the Go graph layer when available.
- `suggested_reads` — file ranges to open in full. Prefer these over
  reading whole files.
- `next_action` — a **prose** directive the agent can execute verbatim.
- `avoid` — what NOT to do (e.g. "don't read entire files").

Intent is inferred automatically from the question shape; override via
`intent` if needed. Supported intents:

```
auto                 // default — let the router decide
behavior_search      // "where is X handled"
symbol_lookup        // exact identifier mentioned
callers              // "what calls X"            (degrades — needs `calls` edges)
callees              // "what does X call"        (degrades — needs `calls` edges)
architecture         // "how does X work overall"
package_topology     // package-level relations
editing_context      // "I want to edit X, what do I need"
```

From the shell:

```console
$ mcsearch context . "where is filesystem event debouncing handled"
$ mcsearch context . "callers of (*Store).Search"
$ mcsearch context . "how does the indexing pipeline work" --intent architecture
$ mcsearch context . "..." --format=json    # raw output for piping
```

From an agent, call the MCP tool `mcsearch_context`. The other MCP
tools below are the building blocks the router composes — call them
directly only when you already know exactly which leg you want.

Drop the snippet at [`docs/claude-md-snippet.md`](docs/claude-md-snippet.md)
into your `CLAUDE.md` to route the agent to `mcsearch_context` before
its grep/Read reflex kicks in.

**MCP tools exposed to Claude:**

- **`mcsearch_context`** — the primary entry point. Pick this for any
  code-understanding question; the tools below are what it composes.
- `semantic_search` — ask in natural language; returns ranked chunks. Hybrid cosine + BM25 + optional rerank. Supports `exclude` path filter and `k` up to 30.
- `find_symbol` — exact identifier lookup by name (SQL, no embedding). Use when you already know the function or type name.
- `related_chunks` — vector neighbours of a known chunk at `path:start_line`. Explore the neighbourhood of a function without a query string.
- `summarize_path` — one-shot file-or-range gist sent directly to the chat model. No retrieval.
- `mcsearch_status` — endpoint health (embed / chat / rerank) and indexed project list.

`mcsearch_context`, `semantic_search`, `find_symbol`, `related_chunks`,
and `mcsearch_status` are always available. `summarize_path` registers
only when `MCSEARCH_CHAT_URL` points at a live `/v1/chat/completions`
server. See [MCP tools](#mcp-tools) for the full input/output contract.

## Install

```bash
git clone https://github.com/alehatsman/mcsearch.git
cd mcsearch
make install        # builds and installs to ~/.local/bin (no sudo)
```

For a system-wide install, override `INSTALL_PATH`:

```bash
sudo make install INSTALL_PATH=/usr/local/bin
```

`install` uses an atomic `rename(2)` swap, so it's safe to re-run while
`mcsearch mcp` or `mcsearch watch` is currently using the binary —
the running process keeps its old inode, and the next invocation picks
up the new one.

This repo is normally deployed by the [`mcsearch` component in
dotfiles](https://github.com/alehatsman/dotfiles/tree/main/components/mcsearch) —
which is how the embedding endpoint, SSH tunnel, and MCP registration are
also wired up.

## CLI

```bash
mcsearch index <path>            # index a project (or re-index incrementally)
mcsearch context <path> "..."    # one-shot router: picks intent, fuses semantic
                                 # + symbol (+ graph when available) and returns
                                 # suggested_reads + a prose next_action. Use
                                 # this BEFORE grep loops.
                                 #   --intent=<name>       force a strategy (see above)
                                 #   --k=<n>               per-lane hit cap (default 8)
                                 #   --format=text|json    raw output for piping
mcsearch query <path> "..."      # raw top-k chunks for a query (no planner)
                                 #   --rerank=off          skip rerank for this call
                                 #   --format=json         emit hits as JSON (rerank_score included)
                                 #   --explain             show per-chunk score breakdown + stage timings
mcsearch generate <path> "..."   # generate code grounded in the project's index
                                 # (RAG: top-k chunks → chat endpoint)
mcsearch status [<path>]         # show indexed projects and endpoint health
mcsearch nuke <path>             # delete the on-disk index for a project
mcsearch reindex <path>          # drop and re-embed a project from scratch
mcsearch reindex --all --yes     # drop and re-embed every known project
                                 # (skips pre-migration indexes; one fresh
                                 # `mcsearch index <path>` registers them)
mcsearch mcp                     # run as an MCP server over stdio
mcsearch watch <path>            # keep the index fresh as files change (fsnotify)
mcsearch clone <src> <dst>       # seed dst's index from src's (e.g. for a new
                                 # git worktree); follow with `mcsearch index
                                 # <dst>` to reconcile any chunks that differ
mcsearch graph export <path>     # dump nodes.jsonl + edges.jsonl
                                 #   --output=<dir>        default <path>/.mcsearch/graph
                                 # (the graph itself is built by `mcsearch index`)
mcsearch version                 # print the build version
```

## Go static graph

`mcsearch index` adds a Go-specific structural layer on top of the
chunk/vector index by default. Where semantic search answers *"where
does auth happen"*, the graph answers *structural* questions: *"what
methods belong to (*Store)"*, *"which packages import internal/store"*,
*"what does this struct embed"*.

It is built on `go/packages` + `go/types` (not regex / not tree-sitter),
so type-resolved symbol names are accurate. Layer 1 (this release)
emits:

- **Nodes:** `package`, `file`, `function`, `method`, `type` (struct /
  interface / other), `field`, `import`.
- **Edges:** `contains`, `imports`, `has_method`, `has_field`, `embeds`.

`calls` and `references` edges land in follow-up releases, along with
`mcsearch graph query|callers|callees` and matching MCP tools.

Storage reuses the per-project SQLite — two new tables (`graph_nodes`,
`graph_edges`) sit alongside `chunks` in `~/.cache/mcsearch/<id>/
index.db`. Function/method nodes link back to chunks (`graph_nodes.
chunk_id`) wherever the chunk's line range covers the symbol, so a
single SQL join surfaces *graph neighborhood + source code* for any
hit.

```console
$ mcsearch index --graph=only .
  graph: 12 packages  797 nodes  866 edges  144 linked  pruned 0/0  in 1.86s

$ mcsearch graph export --output=/tmp/g .
✓ graph exported to /tmp/g
  nodes: /tmp/g/nodes.jsonl
  edges: /tmp/g/edges.jsonl

$ jq -r 'select(.qualified_name == "(*Store).Search") | "\(.file_path):\(.start_line)"' /tmp/g/nodes.jsonl
internal/store/store.go:624
```

Graph extraction runs as a default phase of `mcsearch index`. Use
`--graph=off` to skip it (handy when iterating on chunking) or
`--graph=only` to refresh just the graph layer without re-embedding.
A failure in the graph phase warns and continues; the chunk/vector
index is independently usable.

The graph is incremental on the same prune-by-cutoff discipline as
chunks: rerunning `index` on an unchanged tree upserts every row with
a fresh `last_seen_at` and prunes zero. Removing a file drops the rows
that came from it on the next run.

## Demo

Indexing this very repository against `qwen3-embedding:4b` running on a
local RTX 5090 via ollama (`ollama pull qwen3-embedding:4b`, then point
`MCSEARCH_EMBED_URL=http://127.0.0.1:11434`):

```console
$ mcsearch status
embed  http://127.0.0.1:11434  qwen3-embedding:4b  ok
mcsearch dev

$ mcsearch index ./
✓ indexed /home/aleh/projects/mcsearch
  chunks: 455  files: 32  dim: 2560
```

455 chunks across 32 Go files, ~8 s on a 5090 (a no-change re-run
finishes in ~80 ms thanks to the mtime fast-path).

Now ask in natural language; each query returns the chunk whose meaning
matches, regardless of whether the words line up:

```console
$ mcsearch query -k 1 ./ "where do we debounce filesystem events"
─── #1 markDirty  internal/watch/watch.go:60-71  (method_declaration)  score=0.5102
// markDirty resets the debounce timer; on expiry it runs an index pass.
func (w *Watcher) markDirty() {
    w.mu.Lock()
    defer w.mu.Unlock()
    w.dirty = true
    if w.timer != nil {
        w.timer.Stop()
    }
    w.timer = time.AfterFunc(w.opts.Debounce, w.flush)
}
```

```console
$ mcsearch query -k 1 ./ "code that catches files with literal AWS access keys"
─── #1 internal/ignore/ignore.go:233-252  (orphan)  score=0.6430
// secretPatterns are checked against the first 4 KB of any candidate file.
// A match causes the file to be skipped with a logged warning.
var secretPatterns = []*regexp.Regexp{
    regexp.MustCompile(`AKIA[0-9A-Z]{16}`),                       // AWS access key
    regexp.MustCompile(`ASIA[0-9A-Z]{16}`),                       // AWS STS temporary access key
    regexp.MustCompile(`-----BEGIN [A-Z ]*PRIVATE KEY-----`),     // PEM private key
    regexp.MustCompile(`ghp_[A-Za-z0-9]{36}`),                    // GitHub PAT (classic)
    // …
}
```

```console
$ mcsearch query -k 1 ./ "function that computes cosine similarity"
─── #1 Search  internal/store/store.go:532-540  (method_declaration)  score=0.5024
// Search returns the top-k chunks ranked by hybrid scoring with optional
// per-file diversity via Options.MaxHitsPerFile.
func (s *Store) Search(ctx context.Context, queryVec []float32, queryText string, k int) ([]Hit, error) {
    hits, err := s.searchRaw(ctx, queryVec, queryText, k)
    …
}
```

```console
$ mcsearch query -k 1 ./ "single-flight guard so a second flush waits for the first"
─── #1 TestWatchSingleFlight  internal/watch/watch_test.go:122-176  (function_declaration)  score=0.4890
// TestWatchSingleFlight verifies that bursts of events while a re-index
// is in flight do not spawn a second concurrent indexer (which would
// race on the SQLite writer lock and surface "database is locked"
// errors to the operator). All events end up reflected in the index
// regardless of how rapidly they arrived.
func TestWatchSingleFlight(t *testing.T) {
```

Same query through MCP returns the structured form Claude actually
consumes:

```json
{
  "status": "ok",
  "project": "/home/aleh/projects/mcsearch",
  "hits": [
    {
      "path": "internal/watch/watch.go",
      "kind": "method_declaration",
      "start_line": 128,
      "end_line": 137,
      "score": 0.4793,
      "content": "// markDirty resets the debounce timer; ..."
    }
  ]
}
```

The `status` field is one of `ok` / `no-index` / `embedding-service-unreachable` /
`error`, with a human-readable `hint` so Claude can fall back to `grep` /
`Glob` when the index isn't ready instead of pretending success.

## Environment

The five vars most setups touch:

| Variable               | Default                          | Meaning                                                            |
| ---------------------- | -------------------------------- | ------------------------------------------------------------------ |
| `MCSEARCH_EMBED_URL`   | `http://127.0.0.1:8082`          | OpenAI-compatible `/v1/embeddings` base URL.                       |
| `MCSEARCH_EMBED_MODEL` | `Qwen/Qwen3-Embedding-4B`        | Model name forwarded as `model`.                                   |
| `MCSEARCH_INDEX_DIR`   | `~/.cache/mcsearch`              | Where per-project index files live.                                |
| `MCSEARCH_CHAT_URL`    | `http://127.0.0.1:8081`          | OpenAI-compatible `/v1/chat/completions` base URL (used by `generate` / `summarize_path`). |
| `MCSEARCH_CHAT_MODEL`  | `Qwen/Qwen2.5-Coder-7B-Instruct` | Model for the chat leg.                                            |

Run `mcsearch env` to print the effective configuration with sources
(`env` / `default` / `unset` / `disabled`). Append `--all` to include
the tuning knobs (batch sizes, HTTP timeouts, rerank/compress/draft
endpoints, cache toggles) — see [docs/tuning.md](docs/tuning.md) for
when to touch each one.

## Storage

One SQLite file per project at
`$MCSEARCH_INDEX_DIR/<sha256(realpath(project_root))>/index.db`. Schema:

```
meta(key, value)                                                            -- dim, last_indexed_at, project_root
chunks(id, path, kind, name, start_line, end_line, content_sha1, content,
       vec BLOB, last_seen_at)                                              -- UNIQUE(path, content_sha1)
graph_nodes(id PK, kind, name, qualified_name, package_path, file_path,
            start_line, end_line, chunk_id, metadata_json, content_hash,
            last_seen_at)                                                   -- written by the graph phase of `mcsearch index`
graph_edges(id PK, kind, src_id, dst_id, file_path, start_line, end_line,
            metadata_json, content_hash, last_seen_at)
```

Vectors are stored as packed `float32` BLOBs. A second virtual table,
`chunks_fts`, indexes the same `content` for FTS5/BM25 lookups —
external-content style, kept in sync with `chunks` via AFTER triggers
so it costs nothing extra at upsert time. The `graph_*` tables are
empty until the graph phase of `mcsearch index` runs (skipped only
with `--graph=off`); the chunk side never reads
them.

### Incremental re-index

`mcsearch index` is safe to run repeatedly. Three fast-paths keep it cheap:

| Fast-path | Condition | Cost |
|---|---|---|
| **Mtime** | File mtime ≤ last index run | One `UPDATE last_seen_at` per file — no read, no parse, no embed |
| **SHA** | File changed but chunk content unchanged | Re-parse + SHA, then `UPDATE last_seen_at, name` — no embed call |
| **Full** | New or changed chunk | Parse + embed + upsert |

The SHA fast-path also **backfills the `name` column** on unchanged chunks,
so upgrading to a binary with identifier extraction (used by `find_symbol`)
doesn't require a full `reindex` — the next ordinary `index` run populates
names for free as each file is walked.

If you add a new embedding model (different vector dimension), use
`mcsearch reindex <path>` to drop and re-embed from scratch; mixed
dimensions within one index are rejected at upsert time.

### Hybrid search (semantic + BM25 via RRF)

Every `Search` runs two rankers and fuses them via Reciprocal Rank
Fusion (Cormack et al., 2009):

- the cosine path scores every chunk against the embedded query vector,
- the BM25 path runs the literal tokens of the query text against
  `chunks_fts` via SQLite's `bm25()`,
- each chunk's final score is `Σ 1/(60 + rank_in_list)` summed across
  whichever lists it appeared in.

Why fuse? Semantic alone catches paraphrase ("how do we debounce
filesystem events"), but misses rare literal tokens like
`compileDoubleStar` or `MCSEARCH_DISABLE_VEC_CACHE` that the embedding
model can't anchor on. BM25 alone is the inverse failure mode. RRF is
scale-free, so we don't need to retune weights per corpus.

When the caller hands `Search` an empty query text (or
`MCSEARCH_DISABLE_BM25=1` is set), the BM25 leg is skipped and the
result is the pre-hybrid semantic ranking — same behaviour the
internal tests already exercise.

Hits surface the underlying numbers: `score` is always the cosine for
human comparability, `bm25_score` (larger = better) is filled when the
chunk surfaced through the lexical leg, and `rrf_score` is the fused
rank used for ordering.

### Cross-encoder rerank (optional)

Hybrid RRF is strong on recall but mis-orders top-k on conceptual
queries — the right chunk often sits at position 3 or 4. A
cross-encoder reranker scores `(query, chunk)` pairs *jointly* with
cross-attention and reorders the fused pool before truncation. Off by
default — set `MCSEARCH_RERANK_URL` to enable:

```bash
# TEI serving qwen3-reranker:4b on port 8083
MCSEARCH_RERANK_URL=http://127.0.0.1:8083 mcsearch query ./ "configure embedding model"
```

When reachable, each `Hit` gains a `rerank_score` in `[0, 1]` (larger =
more relevant), visible via `mcsearch query --format=json` and in the
MCP `semantic_search` response. When the endpoint is unreachable the
search falls back to pre-rerank fused order with no error surfaced —
reranker outages never break the search path.

The reranker lives on `Store.Options`, so it applies to every
`store.Search` caller — `semantic_search` and the CLI `query`.
`summarize_path` does not touch `Search` and is unaffected. Per-call
opt-out is `mcsearch query --rerank=off`; process-wide off is
`MCSEARCH_DISABLE_RERANK=1`.

Design and migration notes: see
[`docs/specs/spec-01-rerank.md`](docs/specs/spec-01-rerank.md).

### Code generation (CLI only)

`mcsearch generate <path> "<prompt>"` runs the same hybrid retrieval as
`query`, then prepends the top-k chunks as a `CONTEXT` block and sends
the result to the chat endpoint set via `MCSEARCH_CHAT_URL`. CLI flags
`-k`, `--no-rag`, `--system`, `--temperature`, `--max-tokens`, and
`--show-context` let you steer or bypass RAG from the terminal.

Note: mid-size local chat models (≤32B) tend to generate plausible
code from training data rather than strictly from the retrieved CONTEXT.
Use `semantic_search` for ground-truth chunk retrieval; treat generated
output as a starting point to verify against the actual source.

### Vector cache

On first `Search`, every chunk's vector is decoded once into a flat
in-RAM `[]float32` slab plus precomputed `|v|` norms; subsequent
queries score against the slab with zero hot-path allocations and one
small `SELECT` to fetch content for the top-k IDs. Mutating operations
(`UpsertMany`, `DeletePath`, `DeletePathPrefix`, `PruneUnseen`)
invalidate the slab so the next `Search` rebuilds.

Measured on a Ryzen 9 9950X, brute-force cosine post-cache:

| Chunks | Dim | Search latency (top-k=8) |
| ------:| ---:| ------------------------:|
|   1 k  |  16 | 0.1 ms                   |
|   5 k  | 1024 | 2.7 ms                  |
|  20 k  | 1024 | 12 ms                   |
| 100 k* | 1024 | ~60 ms                  |
| 100 k* | 2560 (Qwen3-Embedding-4B) | ~150 ms |
| 200 k* | 2560 | ~300 ms                  |

(* extrapolated linearly from the measured rows — see
`internal/store/bench_test.go`.)

At realistic project sizes (<50 k chunks) search is never the
bottleneck — the per-query embed round-trip to vLLM/TEI/ollama
dominates total user-perceived latency. The actual ceiling is **RAM**:
the cache slab is `chunks × dim × 4 B`, so 100 k chunks at 2560 dim is
~1 GB. For memory-constrained deployments, set
`MCSEARCH_DISABLE_VEC_CACHE=1` to keep the pre-cache per-row SQL path
(slower but bounded RAM). A real ANN index (HNSW via `coder/hnsw`,
`sqlite-vec`, LanceDB) is the right swap once you push past ~500 k
chunks or want sub-50 ms p99 — the rest of the store stays unchanged.

`last_seen_at` is stored in Unix nanoseconds so the strict-less-than
prune filter correctly distinguishes two index runs that complete in
the same millisecond.

## Multi-worktree workflow

Each `mcsearch` index is keyed by `sha256(realpath(project_root))`, so
`git worktree add ../proj-feature` looks like a brand-new project even
though the trees are nearly identical. `mcsearch clone` seeds the new
worktree's index from a sibling — chunks are keyed by
`(relative path, content sha1)`, so anything unchanged between the two
trees rides along for free. Captured live against this repo:

```console
$ # main checkout already indexed: 455 chunks.
$ git worktree add /tmp/mcsearch-feature -B feature/foo
Preparing worktree (new branch 'feature/foo')
HEAD is now at 24f6497 ux: improve status and query CLI output

$ mcsearch status /tmp/mcsearch-feature
embed  http://127.0.0.1:11434  qwen3-embedding:4b  ok
mcsearch dev

/tmp/mcsearch-feature
  not indexed — run: mcsearch index /tmp/mcsearch-feature

$ mcsearch clone . /tmp/mcsearch-feature
✓ cloned /home/aleh/projects/mcsearch → /tmp/mcsearch-feature
  next: `mcsearch index /tmp/mcsearch-feature` will reconcile any files
        that differ between the two trees (incremental — only changed
        chunks are re-embedded).

$ mcsearch status /tmp/mcsearch-feature
embed  http://127.0.0.1:11434  qwen3-embedding:4b  ok
mcsearch dev

/tmp/mcsearch-feature
  455 chunks  32 files  dim=2560
  last indexed: just now
```

The clone is a `cp` of one SQLite file — ~5 ms in practice. Diverge the
worktree, then run `mcsearch index` to reconcile:

```console
$ echo 'const FeatureXFlag = true' >> /tmp/mcsearch-feature/internal/index/index.go
$ mcsearch index -v /tmp/mcsearch-feature
INFO msg="embedding chunks" count=12
INFO msg=indexed chunks_seen=467 files_fast_path=31 embedded=12 pruned=0 skipped=0
✓ indexed /tmp/mcsearch-feature
  chunks: 467  files: 32  dim: 2560
```

`embedded=12` is the new chunks for the one file that changed (a few
window chunks shift when the file grows). The other 31 files were
skipped via the mtime fast-path without re-reading them — that's the whole
point. On a real branch with a few edits this turns a multi-minute first
index into seconds.

The two indexes remain independent after the clone; subsequent
`mcsearch index` / `mcsearch watch` on each path only touches that
project's cache directory. Pass `--force` to `clone` to overwrite an
existing destination index, or `mcsearch nuke <dst>` first.

## Embedding contract

mcsearch speaks the OpenAI-compatible `/v1/embeddings` shape:

```json
POST /v1/embeddings
{ "model": "Qwen/Qwen3-Embedding-4B", "input": ["chunk-text-1", "chunk-text-2"] }
```

Both vLLM (`vllm serve … --task embed`) and TEI's OpenAI compatibility
shim implement this. Returned vector dimension is discovered on the first
call and recorded on the project; mixed dimensions across re-indexes are
rejected.

## Offline behavior

If the endpoint is unreachable, `mcsearch query` exits non-zero with an
informative error. The MCP `semantic_search` tool returns a structured
result `{ "status": "embedding-service-unreachable", ... }` so Claude can
fall back to grep without crashing.

## Ignore rules

`.gitignore` is respected. In addition, a built-in default
`.mcsearch-ignore` skips `.env*`, `*.pem`, `*.key`, `id_rsa*`,
`id_ed25519*`, `secrets.yml`, `*.tfvars`, `.terraform/`, `node_modules/`,
`vendor/`, `.venv/`, `__pycache__/`, `target/`, `dist/`, `build/`. Files
matching common secret patterns in their first 4 KB are skipped at index
time with a warning.

## MCP tools

When running as `mcsearch mcp`, the server registers the following tools
for the calling agent. All tools except `summarize_path` are always available.

| Tool | Always on? | What it does | Key inputs |
| --- | --- | --- | --- |
| **`mcsearch_context`** | yes | **Primary entry point.** Query planner: picks intent, fuses `semantic_search` + `find_symbol` (+ graph when available), returns `semantic_hits`, `symbols`, `suggested_reads`, plus a prose `next_action` and an `avoid` line the agent can execute verbatim. Call this before grep/Read fan-out. | `question`, `project?`, `intent?` (`auto` default; `behavior_search` / `symbol_lookup` / `callers` / `callees` / `architecture` / `package_topology` / `editing_context`), `k?` (default 8, max 30) |
| `semantic_search` | yes | Hybrid (cosine + BM25 + optional rerank) retrieval. Returns top-k chunks with scores. Use for natural-language queries. Supports `exclude` path-prefix list for noisy directories. Prefer `mcsearch_context` unless you already know you only need this leg. | `query`, `project_root?`, `k?` (default 8, max 30), `exclude?` |
| `find_symbol` | yes | Exact identifier lookup by name — SQL index scan, no embedding. Fast and reliable for known symbol names (`MyFunc`, `HTTPHandler`). | `name`, `project_root?`, `k?` (default 10) |
| `related_chunks` | yes | Vector neighbours of a known chunk at `path:start_line`. Finds semantically related code without a query string — use after `find_symbol` or `semantic_search` to explore the neighbourhood. | `path`, `start_line`, `project_root?`, `k?` (default 8, max 30) |
| `mcsearch_status` | yes | Reports embed / chat / rerank endpoint health and the list of indexed projects with chunk counts, dim, and `last_indexed`. Use it before chasing a "missing result" through the code — the index may be absent, stale, or the embedding endpoint may be down. | — |
| `summarize_path` | needs chat | One-shot file-or-range gist. No retrieval — reads the path directly and sends the slice to the chat model. Path must resolve inside `project_root`; slices larger than 64 KB are truncated (`truncated: true` in the response). Use `focus` to steer (`"public API surface"`, `"side effects"`, etc.). | `path`, `project_root?`, `start_line?`, `end_line?`, `focus?`, `temperature?`, `max_tokens?` |

Every tool returns a structured `status` field — `ok` / `no-index` /
`embedding-service-unreachable` / `chat-service-unreachable` / `error` —
with a human-readable `hint` so the agent can recover (fall back to
`grep`, run `mcsearch index`, etc.) instead of pretending success.

## Docker

A self-contained image is provided. Tree-sitter requires CGO, so the
build stage uses Alpine's musl toolchain to produce a fully static binary
that runs on `distroless/static` (final image ~36 MB, no shell).

```bash
docker build -t mcsearch .

# One-shot index into a named volume.
docker run --rm \
    -v "$PWD":/work:ro -v mcsearch-cache:/cache \
    -e MCSEARCH_EMBED_URL=http://host.docker.internal:8082 \
    mcsearch index /work

# Run as an MCP server over stdio (the default CMD).
docker run --rm -i \
    -v "$PWD":/work:ro -v mcsearch-cache:/cache \
    -e MCSEARCH_EMBED_URL=http://host.docker.internal:8082 \
    mcsearch
```

If you'd rather bind-mount a host directory for `/cache`, pass
`--user "$(id -u):$(id -g)"` — the image runs as the distroless `nonroot`
uid (65532) and otherwise can't write to a host-owned mount point.

## License

MIT — see [LICENSE](./LICENSE).
