# dex

Local semantic code-search and summarization MCP server for Claude (and
a CLI). Tree-sitter chunks → self-hosted embeddings (+ optional
chat-model file/chunk summaries) → SQLite (vectors + BM25 FTS) with
hybrid RRF retrieval and an optional cross-encoder rerank, plus a Go
static graph from `go/packages` + `go/types`. Source never leaves your
machine.

```console
$ dex search semantic ./ "where is filesystem event debouncing handled"
─── #1 markDirty  internal/watch/watch.go:60-71  (method_declaration)
// markDirty resets the debounce timer; on expiry it runs an index pass.
func (w *Watcher) markDirty() { … }
```

## Primary entry point: `ask`

Claude's headline tool. One free-text question, one compact bundle:
`semantic_hits`, `symbols`, `suggested_reads`, plus a prose
`next_action` directive and an `avoid` line. Intent
(`behavior_search` / `symbol_lookup` / `callers` / `callees` /
`architecture` / `package_topology` / `editing_context`) is inferred
from the question shape; pass `intent` to override.

The other MCP tools are the legs `ask` composes — call them directly
only when you already know which leg you want.

Drop [`docs/claude-md-snippet.md`](docs/claude-md-snippet.md) into your
`CLAUDE.md` to route the agent here before its grep/Read reflex kicks
in.

### When to call it via MCP

Reach for `ask` whenever you'd otherwise start with a broad grep, a
speculative Read, or a "find references" fan-out. One
MCP call collapses the loop: intent routing + semantic top-k + symbol
hits + (where available) call-site references, each carrying enough
inline content that you usually don't need a follow-up Read.

**Example 1 — free-text behaviour search.** Question:
*"where is filesystem event debouncing handled?"* Intent auto-routes
to `behavior_search`. The response pins the file from the question
alone:

```jsonc
"suggested_reads": [
  { "path": "internal/watch/watch.go", "start_line": 1, "end_line": 215,
    "reason": "top semantic match",
    "content": "Implements a Watcher that re-indexes a project on
                fsnotify events using debounced timers..." }
],
"next_action": "Read internal/watch/watch.go to ground your answer.",
"avoid":       "Do not read entire files; the suggested ranges cover
                the relevant context."
```

No grep, no exploratory Read — Claude jumps straight to the named
range with a file-level summary already in hand.

**Example 2 — symbol with an explicit verb.** Question:
*"callers of buildNextAction"*. Intent auto-routes to `callers` and
the bundle ships call sites pre-resolved by the static graph's `calls`
edges (Go) or a ripgrep pass (other languages), so Claude never has to
run one itself:

```jsonc
"symbols":    [{ "path": "internal/mcp/context.go", "qualified_name": "buildNextAction",
                 "start_line": 766, "end_line": 815 }],
"references": [
  { "path": "internal/mcp/context.go",      "line": 341,
    "snippet": "out.NextAction = buildNextAction(intent, out.SuggestedReads, ...)" },
  { "path": "internal/mcp/context_test.go", "line": 597,
    "snippet": "got := buildNextAction(tc.intent, tc.reads, tc.syms, tc.topSem)" }
],
"avoid": "Do not grep for the identifier — the `references` field
          already lists usages."
```

The declaration and all three usages come back in a single round-trip;
the `avoid` line tells Claude not to second-guess it with grep.

## Install

```bash
git clone https://github.com/alehatsman/dex.git && cd dex
mooncake task install   # ~/.local/bin, no sudo; atomic rename-swap so it's
                        # safe to re-run while `dex mcp`/`watch` is live
```

Or via the [`dex` dotfiles component](https://github.com/alehatsman/dotfiles/tree/main/components/dex) —
which wires the embedding endpoint, SSH tunnel, and MCP registration.

dex needs CGO (tree-sitter grammars + the embedded sqlite-vec
extension) and the `sqlite_fts5` build tag (FTS5 powers the BM25 leg).
The `tasks.yml` and Dockerfile already pass both — use them. If
you invoke `go build` / `go install` directly, add `-tags sqlite_fts5`
and ensure `CGO_ENABLED=1` plus a C toolchain are available.

## CLI

The query-side commands mirror the MCP tool surface 1:1 (subcommand-group
form): `dex ask` ↔ `ask`, `dex search semantic` ↔
`search_semantic`, `dex graph callers` ↔ `graph_callers`, etc. CLI
and MCP feel like the same tool.

```bash
# query (mirrors MCP tools)
dex ask <path> "..."                       # primary entry point (use BEFORE grep)
                                                #   --intent --k --no-inline --format=text|json
dex search semantic <path> "..."           # hybrid top-k chunks
                                                #   --k --rerank=off --explain --format=json
dex search symbol   <path> <name>          # exact identifier lookup
dex graph neighbors <path> <file> <line>   # vector neighbours of a chunk
dex graph deps      <path> [--file=<rel>|--package=<full>]
dex graph callers   <path> <name>          # incoming calls edges (Go-only)
dex graph callees   <path> <name>          # outgoing calls edges (Go-only)
dex graph export    <path>                 # dump nodes.jsonl + edges.jsonl
dex view summarize  <path> <file>          # one-shot file/range gist via chat
dex index status    [<path>]               # endpoint health + indexed projects

# build / maintenance (CLI-only)
dex index <path>           # build or refresh (chunks + Go graph)
                                #   --graph=off  skip graph phase
                                #   --graph=only refresh just the graph
dex index summarize <path> # drain pending_summaries queue
dex generate <path> "..."  # RAG: top-k chunks → chat endpoint
dex watch <path>           # fsnotify-driven auto-reindex
dex clone <src> <dst>      # seed a worktree's index from a sibling
dex reindex <path>         # drop and re-embed from scratch
dex nuke <path>            # delete the on-disk index
dex mcp                    # MCP server over stdio
```

`dex env` prints effective config with sources. `dex -h` for
the full list.

## Environment

| Variable               | Default                          | Meaning                                                                          |
| ---------------------- | -------------------------------- | -------------------------------------------------------------------------------- |
| `DEX_EMBED_URL`   | `http://127.0.0.1:8082`          | OpenAI-shape `/v1/embeddings` base URL.                                          |
| `DEX_EMBED_MODEL` | `Qwen/Qwen3-Embedding-4B`        | Embedding model name forwarded as `model`.                                       |
| `DEX_INDEX_DIR`   | `~/.cache/dex`              | Per-project index files.                                                         |
| `DEX_CHAT_URL`    | `http://127.0.0.1:8081`          | `/v1/chat/completions` — `generate`, `view_summarize`, index-time summaries.     |
| `DEX_CHAT_MODEL`  | `Qwen/Qwen2.5-Coder-7B-Instruct` | Chat model.                                                                      |

Tuning knobs (rerank, compress, draft, summary, batch sizes, timeouts,
cache toggles) — see [docs/tuning.md](docs/tuning.md).

## How it works

Tree-sitter parses source into named structural chunks (functions,
methods, types, classes). Each chunk hits a self-hosted
`/v1/embeddings` endpoint (ollama, vLLM, or TEI; local or
SSH-tunneled). Embeddings land in a `sqlite-vec` (`vec0`) virtual
table; at query time, vec0 cosine KNN and SQLite FTS5/BM25 are fused
via RRF, with an optional cross-encoder rerank over the fused pool.
Architecture diagram: [docs/architecture.md](docs/architecture.md).
Storage schema, RRF math, vec0 KNN, multi-worktree workflow,
embedding contract, code-gen details: [docs/internals.md](docs/internals.md).

## Go static graph

`dex index` adds a Go-specific structural layer built on
`go/packages` + `go/types` (type-resolved, not regex). The extractor
emits `package` / `file` / `function` / `method` / `type` / `field` /
`import` nodes and `contains` / `imports` / `has_method` / `has_field`
/ `embeds` / `implements` / `calls` edges. Function and method nodes
link back to chunks via `graph_nodes.chunk_id`, so a single SQL join
surfaces graph neighbourhood + source code for any hit. `references`
edges land with the planned LSP integration.

## MCP tools

When running as `dex mcp`, the server registers:

| Tool              | Always on? | What it does                                                                |
| ----------------- | ---------- | --------------------------------------------------------------------------- |
| `ask`             | yes        | **Primary entry point.** Router + composed bundle.                          |
| `search_semantic` | yes        | Hybrid (cosine + BM25 + optional rerank) top-k chunks. Supports `exclude`.  |
| `search_symbol`   | yes        | Exact identifier lookup (SQL scan, no embedding).                           |
| `graph_neighbors` | yes        | Vector neighbours of a known chunk at `path:start_line`.                    |
| `graph_deps`      | yes        | `imports` edges for a file or package. Sourced from the static graph.       |
| `graph_callers`   | yes        | Incoming `calls` edges (Go-only today).                                     |
| `graph_callees`   | yes        | Outgoing `calls` edges (Go-only today).                                     |
| `index_status`    | yes        | Endpoint health (embed / chat / rerank) + indexed projects.                 |
| `view_summarize`  | needs chat | One-shot file-or-range gist via the chat model. No retrieval.               |

All tools return `status` (`ok` / `no-index` / `no-graph` /
`embedding-service-unreachable` / `chat-service-unreachable` / `error`)
with a human-readable `hint` so Claude can fall back to grep instead of
pretending success.

## Docker

```bash
docker build -t dex .
docker run --rm -v "$PWD":/work:ro -v dex-cache:/cache \
    -e DEX_EMBED_URL=http://host.docker.internal:8082 \
    dex index /work
```

Tree-sitter needs CGO, so the build stage uses Alpine's musl toolchain
to produce a static binary on `distroless/static` (final image ~36 MB,
no shell). For a host-bound `/cache`, add `--user "$(id -u):$(id -g)"`
(image runs as distroless `nonroot`, uid 65532).

## Ignore rules

`.gitignore` is respected. A built-in `.dex-ignore` skips
`.env*`, `*.pem`, `*.key`, `id_rsa*`, `id_ed25519*`, `secrets.yml`,
`*.tfvars`, `.terraform/`, `node_modules/`, `vendor/`, `.venv/`,
`__pycache__/`, `target/`, `dist/`, `build/`. Files matching common
secret patterns in their first 4 KB are skipped at index time.

## License

MIT — see [LICENSE](./LICENSE).
