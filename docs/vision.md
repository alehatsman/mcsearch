# Vision

Where dex could go beyond "an MCP server that does semantic search".
This is a direction note, not a committed roadmap.

## The pitch

A single Go binary that's both an always-on local code-intelligence
daemon and a CLI. One process owns the index, the warm connections to
local model servers, and the cached language-server handles. It's
reachable three ways:

- **MCP** — for coding agents (Claude Code, Cursor, etc.). What dex
  is today.
- **LSP** — for editors. Same primitives exposed as
  `textDocument/hover`, semantic goto, `inlineCompletion`, etc. Any
  LSP-aware editor (VS Code, Neovim, Helix, Zed, JetBrains) plugs in
  without a per-editor extension.
- **CLI** — for humans and CI. `dex ask`, `dex graph deps`,
  `dex index status`.

The point of "one daemon, three faces" is that the agent and the editor
hit the *same* live index and the *same* warm models. No duplicated
state, no second startup cost.

## Three audiences, what each gets

**Coding agent.** A small, composable tool surface it can learn:
`search.*`, `graph.*`, `view.*`, `index.*`, plus one high-level `ask`
that routes internally. Most agents will only call `ask` and
`graph.callers/callees`.

**Editor user.** Hover that returns a generated summary, not just a
docstring. "Find semantic references" alongside LSP's exact references.
Inline completion that reads the local index for context instead of
shipping the file to a cloud model.

**CLI user / CI.** Scriptable access to everything the agent has —
useful for `pre-commit`, code review bots, ad-hoc investigation.

## Local-GPU awareness is the differentiator

dex already does the right thing on inference: talk to a local
OpenAI-compatible server (llama-server / vLLM / Ollama / TEI) rather
than embed inference in-process. Keep that. The binary's job is
*awareness*, not bundling llama.cpp via cgo:

- Detect what's running locally (which servers, which models, free
  VRAM).
- Route embed / summary / completion to the right endpoint.
- Fall back gracefully when nothing is up.

On modern hardware (e.g. a 5090 with 32 GB VRAM, see
`memory/hardware_main_pc.md`) a resident 1–3B code model can hit
Copilot-grade latency (<200 ms time-to-first-token) while embeddings
and summaries coexist. That's a product story most "local copilot"
tools can't tell.

## Tree-sitter vs LSP — different layers

- **Tree-sitter** = structural chunking, symbol extraction, "what kind
  of thing is this". Cheap, no language servers needed. Already how
  the chunker works.
- **LSP-as-consumer** = ground-truth references, types, callers via
  `gopls`, `rust-analyzer`, `pyright`, etc. Expensive (the daemon
  spawns and caches them), but unlocks a correctness tier semantic
  search can't match.

The model: LSP is an *optional accuracy upgrade*. If `gopls` is on
PATH, `graph.callers` and `graph.usages` become precise; otherwise
they're tree-sitter-approximate. The daemon owns the language-server
lifecycle and the cache.

## Tool surface

The MCP tools are grouped by verb. Underscore separators (not dots)
for cross-client compatibility:

```
ask                       # high-level router; what most agents call
search_semantic           # cosine + BM25 + RRF
search_symbol             # exact / qualified name
search_text               # ripgrep-equivalent (not yet shipped)
graph_deps                # file→pkg, pkg→pkg imports
graph_callers             # incoming calls edges (Go-only today)
graph_callees             # outgoing calls edges (Go-only today)
graph_neighbors           # cosine neighbours of a known chunk
view_summarize            # chat-model file/range gist
view_expand               # widen a chunk to its enclosing scope (not yet shipped)
index_status              # endpoint health + indexed projects
index_refresh             # force reindex (not yet shipped)
```

## Scope cuts, in priority order

1. ✅ **Tighten the agent API.** Done. Tools regrouped into `search_*`,
   `graph_*`, `view_*`, `index_*`; `dex_context` is now `ask`.
2. ✅ **Add `graph_deps` and `graph_callers`/`graph_callees`.** Done
   for Go via `go/types`. Tree-sitter-based extraction for Python /
   JS / Rust is deferred — non-Go callers fall back to the ripgrep
   `references` lane in the `ask` bundle. Implementation sketch when
   it lands: per-language tree-sitter queries in
   `internal/graph/sitter_calls.go` emitting `(caller, callee_name)`
   pairs; resolution within a file is best-effort by name, cross-file
   via the existing `graph_nodes` qualified-name index. Mark edges
   with `provenance: "sitter"` metadata so the MCP layer can
   distinguish them from the type-resolved Go edges.
3. **Ship the LSP server, read-side only.** Hover-with-summary,
   semantic goto, find-related. No completion yet. Unlocks every
   editor without a per-editor extension.
4. **LSP-as-consumer for precision.** Cache `gopls` etc. inside the
   daemon; upgrade graph.* answers when available.
5. **Inline completion.** Separate product on the same daemon. Latency
   budget, prompt construction, FIM model selection are their own
   rabbit hole, and the failure mode (slow/wrong completions) is much
   more visible than a slow agent search. Last, not first.

## Quality infrastructure

Open work that protects retrieval quality across future changes:

- ✅ **Retrieval regression harness.** Shipped in
  `internal/store/regression_test.go` with the fixture under
  `internal/store/testdata/regression/`. Synthetic 25-chunk corpus +
  18 queries spanning pure-semantic, pure-lexical, and hybrid cases,
  driven by a deterministic two-hash mock embedder so the harness is
  hermetic and runs in `go test` in under a second. Asserts expected
  paths land in fused top-k per query and prints a per-leg rank table
  (`semantic / bm25 / fused / rerank`) on failure so the regressing
  leg is pinpointable. A second test wires a substring-counting stub
  reranker and asserts the rerank path stays alive across the suite.
  See the test file's package comment for how to add a query.

## What this is *not*

- Not an inference engine. dex will not embed llama.cpp.
- Not a cloud product. Local-first; if your hardware can't host the
  models, you bring your own endpoint.
- Not a VS Code extension. The LSP server is the integration point;
  editor-specific UI is downstream.
