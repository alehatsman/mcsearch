# mcsearch

Local semantic code-search MCP server for Claude (and a CLI). Tree-sitter
chunks → self-hosted embeddings → SQLite vector store + Go static graph.
Source never leaves your machine.

```console
$ mcsearch query ./ "where is filesystem event debouncing handled"
─── #1 markDirty  internal/watch/watch.go:60-71  (method_declaration)
// markDirty resets the debounce timer; on expiry it runs an index pass.
func (w *Watcher) markDirty() { … }
```

## Primary entry point: `mcsearch_context`

Claude's headline tool. One free-text question, one compact bundle:
`semantic_hits`, `symbols`, `suggested_reads`, plus a prose
`next_action` directive and an `avoid` line. Intent
(`behavior_search` / `symbol_lookup` / `callers` / `callees` /
`architecture` / `package_topology` / `editing_context`) is inferred
from the question shape; pass `intent` to override.

The other MCP tools are the legs `mcsearch_context` composes — call
them directly only when you already know which leg you want.

Drop [`docs/claude-md-snippet.md`](docs/claude-md-snippet.md) into your
`CLAUDE.md` to route the agent here before its grep/Read reflex kicks
in.

## Install

```bash
git clone https://github.com/alehatsman/mcsearch.git && cd mcsearch
make install        # ~/.local/bin, no sudo; atomic rename-swap so it's
                    # safe to re-run while `mcsearch mcp`/`watch` is live
```

Or via the [`mcsearch` dotfiles component](https://github.com/alehatsman/dotfiles/tree/main/components/mcsearch) —
which wires the embedding endpoint, SSH tunnel, and MCP registration.

## CLI

```bash
mcsearch index <path>          # build or refresh the index (chunks + Go graph)
                               #   --graph=off  skip graph phase
                               #   --graph=only refresh just the graph
mcsearch context <path> "..."  # one-shot router (use this BEFORE grep)
                               #   --intent --k --format=text|json
mcsearch query <path> "..."    # raw top-k chunks
                               #   --rerank=off --explain --format=json
mcsearch generate <path> "..." # RAG: top-k chunks → chat endpoint
mcsearch status [<path>]       # endpoint health + indexed projects
mcsearch watch <path>          # fsnotify-driven auto-reindex
mcsearch clone <src> <dst>     # seed a worktree's index from a sibling
mcsearch reindex <path>        # drop and re-embed from scratch
mcsearch graph export <path>   # dump nodes.jsonl + edges.jsonl
mcsearch mcp                   # MCP server over stdio
```

`mcsearch env` prints effective config with sources. `mcsearch -h` for
the full list.

## Environment

| Variable               | Default                          | Meaning                                                                          |
| ---------------------- | -------------------------------- | -------------------------------------------------------------------------------- |
| `MCSEARCH_EMBED_URL`   | `http://127.0.0.1:8082`          | OpenAI-shape `/v1/embeddings` base URL.                                          |
| `MCSEARCH_EMBED_MODEL` | `Qwen/Qwen3-Embedding-4B`        | Embedding model name forwarded as `model`.                                       |
| `MCSEARCH_INDEX_DIR`   | `~/.cache/mcsearch`              | Per-project index files.                                                         |
| `MCSEARCH_CHAT_URL`    | `http://127.0.0.1:8081`          | `/v1/chat/completions` — `generate`, `summarize_path`, index-time summaries.     |
| `MCSEARCH_CHAT_MODEL`  | `Qwen/Qwen2.5-Coder-7B-Instruct` | Chat model.                                                                      |

Tuning knobs (rerank, compress, draft, summary, batch sizes, timeouts,
cache toggles) — see [docs/tuning.md](docs/tuning.md).

## How it works

Tree-sitter parses source into named structural chunks (functions,
methods, types, classes). Each chunk hits a self-hosted
`/v1/embeddings` endpoint (ollama, vLLM, or TEI; local or
SSH-tunneled). At query time, cosine similarity and BM25 are fused via
RRF; an optional cross-encoder rerank reorders the fused pool.
Architecture diagram: [docs/architecture.md](docs/architecture.md).
Storage schema, RRF math, vector cache, multi-worktree workflow,
embedding contract, code-gen details: [docs/internals.md](docs/internals.md).

## Go static graph

`mcsearch index` adds a Go-specific structural layer built on
`go/packages` + `go/types` (type-resolved, not regex). Layer 1 emits
`package` / `file` / `function` / `method` / `type` / `field` /
`import` nodes and `contains` / `imports` / `has_method` / `has_field`
/ `embeds` edges. Function and method nodes link back to chunks via
`graph_nodes.chunk_id`, so a single SQL join surfaces graph
neighbourhood + source code for any hit. `calls` and `references`
edges land in follow-up releases.

## MCP tools

When running as `mcsearch mcp`, the server registers:

| Tool                | Always on? | What it does                                                                |
| ------------------- | ---------- | --------------------------------------------------------------------------- |
| `mcsearch_context`  | yes        | **Primary entry point.** Router + composed bundle.                          |
| `semantic_search`   | yes        | Hybrid (cosine + BM25 + optional rerank) top-k chunks. Supports `exclude`.  |
| `find_symbol`       | yes        | Exact identifier lookup (SQL scan, no embedding).                           |
| `related_chunks`    | yes        | Vector neighbours of a known chunk at `path:start_line`.                    |
| `mcsearch_status`   | yes        | Endpoint health (embed / chat / rerank) + indexed projects.                 |
| `summarize_path`    | needs chat | One-shot file-or-range gist via the chat model. No retrieval.               |

All tools return `status` (`ok` / `no-index` /
`embedding-service-unreachable` / `chat-service-unreachable` / `error`)
with a human-readable `hint` so Claude can fall back to grep instead of
pretending success.

## Docker

```bash
docker build -t mcsearch .
docker run --rm -v "$PWD":/work:ro -v mcsearch-cache:/cache \
    -e MCSEARCH_EMBED_URL=http://host.docker.internal:8082 \
    mcsearch index /work
```

Tree-sitter needs CGO, so the build stage uses Alpine's musl toolchain
to produce a static binary on `distroless/static` (final image ~36 MB,
no shell). For a host-bound `/cache`, add `--user "$(id -u):$(id -g)"`
(image runs as distroless `nonroot`, uid 65532).

## Ignore rules

`.gitignore` is respected. A built-in `.mcsearch-ignore` skips
`.env*`, `*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`, `secrets.yml`,
`*.tfvars`, `.terraform/`, `node_modules/`, `vendor/`, `.venv/`,
`__pycache__/`, `target/`, `dist/`, `build/`. Files matching common
secret patterns in their first 4 KB are skipped at index time.

## License

MIT — see [LICENSE](./LICENSE).
