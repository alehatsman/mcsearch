# mcsearch

Local semantic-search helper for Claude Code. Indexes a project on-disk,
embeds chunks against a self-hosted embedding endpoint (e.g. vLLM or TEI on
an RTX 5090), and exposes a `semantic_search` tool over MCP so Claude can
ask for ranked code chunks instead of fanning out grep calls.

Source code never leaves the calling machine — only chunk text crosses the
wire to the embedding endpoint (typically over an SSH tunnel to your GPU
host).

## Install

```bash
git clone https://github.com/alehatsman/mcsearch.git
cd mcsearch
make install        # builds and copies to /usr/local/bin
```

This repo is normally deployed by the [`mcsearch` component in
dotfiles](https://github.com/alehatsman/dotfiles/tree/main/components/mcsearch) —
which is how the embedding endpoint, SSH tunnel, and MCP registration are
also wired up.

## CLI

```bash
mcsearch index <path>          # index a project (or re-index incrementally)
mcsearch query <path> "..."    # query an indexed project from the terminal
mcsearch status [<path>]       # show indexed projects and endpoint health
mcsearch nuke <path>           # delete the on-disk index for a project
mcsearch mcp                   # run as an MCP server over stdio
mcsearch watch <path>          # keep the index fresh as files change (fsnotify)
mcsearch clone <src> <dst>     # seed dst's index from src's (e.g. for a new
                               # git worktree); follow with `mcsearch index
                               # <dst>` to reconcile any chunks that differ
```

## Environment

| Variable                  | Default                            | Meaning                                       |
| ------------------------- | ---------------------------------- | --------------------------------------------- |
| `MCSEARCH_EMBED_URL`      | `http://127.0.0.1:8082`            | OpenAI-compatible `/v1/embeddings` base URL.  |
| `MCSEARCH_EMBED_MODEL`    | `Qwen/Qwen3-Embedding-4B`          | Model name forwarded as `model` field.        |
| `MCSEARCH_INDEX_DIR`      | `~/.cache/mcsearch`                | Where per-project index files live.           |
| `MCSEARCH_EMBED_TIMEOUT`  | `60s`                              | HTTP timeout for each embedding request.      |
| `MCSEARCH_EMBED_BATCH`    | `32`                               | Max chunks per `/v1/embeddings` call.         |

## Storage

One SQLite file per project at
`$MCSEARCH_INDEX_DIR/<sha256(realpath(project_root))>/index.db`. Schema:

```
meta(key, value)                                                            -- dim, last_indexed_at
chunks(id, path, kind, start_line, end_line, content_sha1, content,
       vec BLOB, last_seen_at)                                              -- UNIQUE(path, content_sha1)
```

Vectors are stored as packed `float32` BLOBs. Query is brute-force cosine
similarity over all chunks — fine at <100 k chunks per project, ~30–80 ms
on a modern laptop. If a project outgrows this, swap the store backend for
an HNSW index (e.g. via `hnswlib-go` or LanceDB) without changing the rest
of the architecture.

`last_seen_at` is stored in Unix nanoseconds so the strict-less-than prune
filter correctly distinguishes two index runs that complete in the same
millisecond.

## Multi-worktree workflow

Each `mcsearch` index is keyed by `sha256(realpath(project_root))`, so a
fresh `git worktree add ../proj-feature` (or a sibling clone) looks like
a brand-new project even though most of the content is identical. Use
`mcsearch clone` to seed the new worktree's index from an already-indexed
sibling — chunks are addressed by `(relative path, content sha1)`, so
anything unchanged between the two trees comes along for free:

```bash
# Original tree, already indexed.
mcsearch index ~/proj

# New worktree on a feature branch.
git worktree add ~/proj-feature feature/foo
mcsearch clone ~/proj ~/proj-feature      # copy the SQLite index
mcsearch index ~/proj-feature             # reconcile — only files that
                                          # diverged get re-embedded
```

The two indexes remain independent after the clone; subsequent
`mcsearch index` / `mcsearch watch` on each path only touches that
project's cache directory.

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
