# Vision

Where mcsearch could go beyond "an MCP server that does semantic search".
This is a direction note, not a committed roadmap.

## The pitch

A single Go binary that's both an always-on local code-intelligence
daemon and a CLI. One process owns the index, the warm connections to
local model servers, and the cached language-server handles. It's
reachable three ways:

- **MCP** — for coding agents (Claude Code, Cursor, etc.). What mcsearch
  is today.
- **LSP** — for editors. Same primitives exposed as
  `textDocument/hover`, semantic goto, `inlineCompletion`, etc. Any
  LSP-aware editor (VS Code, Neovim, Helix, Zed, JetBrains) plugs in
  without a per-editor extension.
- **CLI** — for humans and CI. `mcsearch ask`, `mcsearch deps`,
  `mcsearch status`.

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

mcsearch already does the right thing on inference: talk to a local
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

## Tool surface, after the rename

Today the MCP tools are `mcsearch_context`, `semantic_search`,
`find_symbol`, `related_chunks`, `summarize_path`, `mcsearch_status`.
The naming mixes verbs and nouns and the prefix is noise.

Proposed shape — verbs grouped into four namespaces:

```
ask                       # high-level router; what most agents call
search.semantic           # cosine + BM25 + RRF
search.symbol             # exact / qualified name
search.text               # ripgrep-equivalent, for completeness
graph.deps                # file→file, module→module imports
graph.callers             # who calls X
graph.callees             # what X calls
graph.neighbors           # current related_chunks
view.summarize            # current summarize_path
view.expand               # widen a chunk to its enclosing scope
index.status              # current mcsearch_status
index.refresh             # force reindex
```

Renames that matter most:

- `mcsearch_context` → `ask`. It's the verb the agent is performing.
- `related_chunks` → `graph.neighbors`. Says what it does.
- `summarize_path` → `view.summarize`. Groups with other read-side ops.

## Scope cuts, in priority order

1. **Tighten the agent API.** Rename and regroup into the four
   namespaces above. One high-level `ask` that internally routes. This
   is pure refactor — no new capability — but it's the foundation
   everything else hangs off.
2. **Add `graph.deps` and `graph.callers/callees`.** Tree-sitter-based
   first. This is the missing primitive agents most want, and the
   current `related_chunks` only half-delivers.
3. **Ship the LSP server, read-side only.** Hover-with-summary,
   semantic goto, find-related. No completion yet. Unlocks every
   editor without a per-editor extension.
4. **LSP-as-consumer for precision.** Cache `gopls` etc. inside the
   daemon; upgrade graph.* answers when available.
5. **Inline completion.** Separate product on the same daemon. Latency
   budget, prompt construction, FIM model selection are their own
   rabbit hole, and the failure mode (slow/wrong completions) is much
   more visible than a slow agent search. Last, not first.

## What this is *not*

- Not an inference engine. mcsearch will not embed llama.cpp.
- Not a cloud product. Local-first; if your hardware can't host the
  models, you bring your own endpoint.
- Not a VS Code extension. The LSP server is the integration point;
  editor-specific UI is downstream.
