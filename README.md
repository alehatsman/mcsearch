# mcsearch

Local semantic-search helper for Claude Code. Indexes a project on-disk,
embeds chunks against a self-hosted OpenAI-compatible `/v1/embeddings`
endpoint (vLLM, TEI, or ollama — local GPU or SSH-tunneled to a remote
host), and exposes a `semantic_search` tool over MCP so Claude can ask
for ranked code chunks instead of fanning out grep calls.

A second MCP tool, `generate_code`, reuses the same index as RAG context
and routes the prompt through a self-hosted `/v1/chat/completions`
endpoint — same wire shape, different model — so generation is grounded
in real symbols and paths from your project.

Source code never leaves the calling machine — only chunk text crosses
the wire to the embedding / chat endpoints, which you control.

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
mcsearch query <path> "..."      # query an indexed project from the terminal
                                 #   --rerank=off          skip rerank for this call
                                 #   --format=json         emit hits as JSON (rerank_score included)
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
mcsearch version                 # print the build version
```

## Demo

Indexing this very repository against `qwen3-embedding:4b` running on a
local RTX 5090 via ollama (`ollama pull qwen3-embedding:4b`, then point
`MCSEARCH_EMBED_URL=http://127.0.0.1:11434`):

```console
$ mcsearch status
embed endpoint: http://127.0.0.1:11434   ok
model: qwen3-embedding:4b
mcsearch version: dev
index dir: /home/aleh/.cache/mcsearch

$ mcsearch index ./
✓ indexed /home/aleh/projects/mcsearch
  chunks: 221  files: 21  dim: 2560
```

221 chunks across 21 Go files, ~6.6 s on a 5090 (a no-change re-run
finishes in ~80 ms thanks to the mtime fast-path).

Now ask in natural language; each query returns the chunk whose meaning
matches, regardless of whether the words line up:

```console
$ mcsearch query -k 1 ./ "where do we debounce filesystem events"
─── #1 internal/watch/watch.go:128-137  (method_declaration)  score=0.4793
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
─── #1 internal/store/store.go:328-417  (method_declaration)  score=0.4146
// Search returns the top-k chunks by cosine similarity to query.
//
// Hot path scores against the in-RAM vector cache (a single flat
// []float32 slab plus precomputed |v| norms) and then issues exactly
// one SELECT to fetch path/kind/line/content for the top-k IDs.
// …
func (s *Store) Search(ctx context.Context, query []float32, k int) ([]Hit, error) {
```

```console
$ mcsearch query -k 1 ./ "single-flight guard so a second flush waits for the first"
─── #1 internal/watch/watch_test.go:122-176  (function_declaration)  score=0.4890
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

| Variable                  | Default                            | Meaning                                       |
| ------------------------- | ---------------------------------- | --------------------------------------------- |
| `MCSEARCH_EMBED_URL`      | `http://127.0.0.1:8082`            | OpenAI-compatible `/v1/embeddings` base URL.  |
| `MCSEARCH_EMBED_MODEL`    | `Qwen/Qwen3-Embedding-4B`          | Model name forwarded as `model` field.        |
| `MCSEARCH_INDEX_DIR`      | `~/.cache/mcsearch`                | Where per-project index files live.           |
| `MCSEARCH_EMBED_TIMEOUT`  | `60s`                              | HTTP timeout for each embedding request.      |
| `MCSEARCH_EMBED_BATCH`    | `32`                               | Max chunks per `/v1/embeddings` call.         |
| `MCSEARCH_DISABLE_VEC_CACHE` | unset                           | Set to `1` to skip the in-RAM vector cache and use the per-row SQL hot path (slower; bounded RAM for very large indexes). |
| `MCSEARCH_DISABLE_BM25`   | unset                              | Set to `1` to skip the BM25 leg of hybrid search and rank by cosine similarity alone. |
| `MCSEARCH_CHAT_URL`       | `http://127.0.0.1:8081`            | OpenAI-compatible `/v1/chat/completions` base URL (used by `generate`). |
| `MCSEARCH_CHAT_MODEL`     | `Qwen/Qwen2.5-Coder-7B-Instruct`   | Model name forwarded as `model` for the chat leg. |
| `MCSEARCH_ALLOW_PATHS`    | unset                              | Colon-separated path prefixes (`:` on POSIX, `;` on Windows) that `index`/`watch` accept even when the target isn't inside a git work tree. Entries support `~` and `$HOME` expansion. |
| `MCSEARCH_CHAT_TIMEOUT`   | `120s`                             | HTTP timeout for each chat-completion request. |
| `MCSEARCH_RERANK_URL`     | unset                              | Base URL of a Cohere-shape `/rerank` server (TEI, Infinity, vLLM with reranker). Unset = rerank disabled. |
| `MCSEARCH_RERANK_MODEL`   | `qwen3-reranker:4b`                | Model name forwarded to the reranker. |
| `MCSEARCH_RERANK_POOL`    | `40`                               | Fused candidates fed to the reranker. Clamped to `[1, 100]`. Larger = better recall, slower call. |
| `MCSEARCH_RERANK_TIMEOUT` | `5s`                               | HTTP timeout for each rerank request. |
| `MCSEARCH_DISABLE_RERANK` | unset                              | Set to `1` to short-circuit rerank even when `MCSEARCH_RERANK_URL` is set. For A/B comparison. |

## Storage

One SQLite file per project at
`$MCSEARCH_INDEX_DIR/<sha256(realpath(project_root))>/index.db`. Schema:

```
meta(key, value)                                                            -- dim, last_indexed_at
chunks(id, path, kind, start_line, end_line, content_sha1, content,
       vec BLOB, last_seen_at)                                              -- UNIQUE(path, content_sha1)
```

Vectors are stored as packed `float32` BLOBs. A second virtual table,
`chunks_fts`, indexes the same `content` for FTS5/BM25 lookups —
external-content style, kept in sync with `chunks` via AFTER triggers
so it costs nothing extra at upsert time.

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
`store.Search` caller — `semantic_search`, `generate_code`'s RAG,
`ask_codebase`'s RAG, and the CLI `query`. `summarize_path` does not
touch `Search` and is unaffected. Per-call opt-out is
`mcsearch query --rerank=off`; process-wide off is
`MCSEARCH_DISABLE_RERANK=1`.

Design and migration notes: see
[`docs/specs/spec-01-rerank.md`](docs/specs/spec-01-rerank.md).

### Code generation (RAG + chat)

`mcsearch generate <path> "<prompt>"` runs the same hybrid retrieval as
`query`, then prepends the top-k chunks to the prompt as a `CONTEXT`
block and sends the result to a self-hosted `/v1/chat/completions`
endpoint:

```console
$ mcsearch generate ./ "add a flag to skip the BM25 leg in cmdQuery"
```

The CLI flags (`-k`, `--no-rag`, `--system`, `--temperature`,
`--max-tokens`, `--show-context`) let you steer or short-circuit RAG
from the terminal. Over MCP the same capability surfaces as a
`generate_code` tool, returning both the generated text and the chunks
that were fed in as context — so Claude can verify the model didn't
hallucinate names that weren't in the project:

```json
{
  "status": "ok",
  "project": "/home/aleh/projects/mcsearch",
  "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "content": "```go\nif *noBM25 {\n    opts.DisableBM25 = true\n}\n```",
  "finish_reason": "stop",
  "context": [
    { "path": "cmd/mcsearch/main.go", "start_line": 221, "end_line": 275, "kind": "function_declaration", "score": 0.51 },
    { "path": "internal/store/store.go", "start_line": 328, "end_line": 417, "kind": "method_declaration", "score": 0.42 }
  ]
}
```

The `status` field follows the same convention as `semantic_search`:
`ok` / `no-index` / `embedding-service-unreachable` /
`chat-service-unreachable` / `error`, each with a `hint` so the caller
can recover gracefully. Point `MCSEARCH_CHAT_URL` at any
OpenAI-compatible `/v1/chat/completions` server — vLLM
(`vllm serve … --task generate`), TEI's compat shim, or ollama (e.g.
`ollama pull qwen2.5-coder:7b-instruct`, then
`MCSEARCH_CHAT_URL=http://127.0.0.1:11434`).

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
$ # main checkout already indexed: 221 chunks.
$ git worktree add /tmp/mcsearch-feature -B feature/foo
Preparing worktree (new branch 'feature/foo')
HEAD is now at bca65ea docs: README demo section with real captured output

$ mcsearch status /tmp/mcsearch-feature
project: /tmp/mcsearch-feature
  no index — run `mcsearch index /tmp/mcsearch-feature`

$ mcsearch clone . /tmp/mcsearch-feature
✓ cloned /home/aleh/projects/mcsearch → /tmp/mcsearch-feature
  next: `mcsearch index /tmp/mcsearch-feature` will reconcile any files
        that differ between the two trees (incremental — only changed
        chunks are re-embedded).

$ mcsearch status /tmp/mcsearch-feature
project: /tmp/mcsearch-feature
  chunks: 221  files: 21  dim: 2560  last_indexed: 2026-05-16T18:47:58+02:00
```

The clone is a `cp` of one SQLite file — ~5 ms in practice. Diverge the
worktree, then run `mcsearch index` to reconcile:

```console
$ echo 'const FeatureXFlag = true' >> /tmp/mcsearch-feature/internal/index/index.go
$ mcsearch index -v /tmp/mcsearch-feature
INFO msg="pruned stale chunks (files removed since last index)" count=5
INFO msg=indexed chunks_seen=226 files_fast_path=0 embedded=10 pruned=5 skipped=0
✓ indexed /tmp/mcsearch-feature
  chunks: 226  files: 21  dim: 2560
```

`embedded=10` is the new chunks for the one file that changed (a few
window chunks shift when the file grows). The other 20 files were
content-sha matched against the cloned index and reused without an
embedding call — that's the whole point. On a real branch with a few
edits this turns a multi-minute first index into seconds.

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
for the calling agent. `semantic_search` and `mcsearch_status` are always
available; the three chat-backed tools register only when
`MCSEARCH_CHAT_URL` is set and the chat client initialises cleanly.

| Tool | Always on? | What it does | Key inputs |
| --- | --- | --- | --- |
| `semantic_search` | yes | Hybrid (cosine + BM25 + optional rerank) retrieval over the project's index. Returns the top-k chunks with `path`, `kind`, `start_line`, `end_line`, `score`, `bm25_score`, `rrf_score`, `rerank_score`, `content`. Prefer this over fanning out `grep` when the query is described in natural language. | `query`, `project_root?`, `k?` (default 8, max 30) |
| `mcsearch_status` | yes | Reports embed / chat / rerank endpoint health and the list of indexed projects with chunk counts, dim, and `last_indexed`. Use it before chasing a "missing result" through the code — the index may be absent, stale, or the embedding endpoint may be down. | — |
| `ask_codebase` | needs chat | Natural-language Q&A grounded in the index. Same retrieval pipeline as `generate_code` but tuned for prose answers: emits a `CITATIONS:` / `ANSWER:` block where every claim carries a `[n]` tag back to a real chunk. Use this instead of `Read`+synthesise for "how does X work?" / "where is Y handled?". | `prompt`, `project_root?`, `k?`, `use_index?` (default true), `system?`, `temperature?`, `max_tokens?` |
| `generate_code` | needs chat | Code generation / edit / explanation. Same retrieval as `ask_codebase`, but the system prompt steers the model toward fenced code blocks. Returns both the generated `content` and the `context` chunks that were fed in, so callers can verify the model didn't invent symbols. | same as `ask_codebase` |
| `summarize_path` | needs chat | One-shot file-or-range gist. No retrieval — reads the path directly and sends the slice to the chat model. Path must resolve inside `project_root`; slices larger than 64 KB are truncated (`truncated: true` in the response). Use `focus` to steer (`"public API surface"`, `"side effects"`, etc.). For whole-repo overviews use `ask_codebase` instead. | `path`, `project_root?`, `start_line?`, `end_line?`, `focus?`, `temperature?`, `max_tokens?` |

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
