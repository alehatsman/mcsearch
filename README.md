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
mcsearch status                # show indexed projects and endpoint health
mcsearch nuke <path>           # delete the on-disk index for a project
mcsearch mcp                   # run as an MCP server over stdio
mcsearch watch <path>          # keep the index fresh as files change (fsnotify)
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
projects(id, root, last_indexed_at)
chunks(id, path, kind, start_line, end_line, content_sha1, content, vec BLOB, last_seen_at)
```

Vectors are stored as packed `float32` BLOBs. Query is brute-force cosine
similarity over all chunks — fine at <100 k chunks per project, ~30–80 ms
on a modern laptop. If a project outgrows this, swap the store backend for
an HNSW index (e.g. via `hnswlib-go` or LanceDB) without changing the rest
of the architecture.

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

## License

MIT — see [LICENSE](./LICENSE).
