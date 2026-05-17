# Spec 01: Cross-Encoder Reranker

**Status:** 📝 Draft
**Effort:** S–M (3–5 days of focused work)
**Value:** Quality. Today's hybrid RRF (semantic ⊕ BM25) is strong on
recall but mis-orders the top-k on conceptual queries — the top hit
often belongs in position 3 or 4. A cross-encoder reranker reorders the
existing fused candidate pool, raising answer quality at the
top of the list (the only positions Claude usually reads).

---

## Problem

`store.Search` fuses cosine and BM25 rankings via Reciprocal Rank Fusion
(`store.go:444`). Both legs are "bi-encoder" style: query and chunk are
embedded independently, then compared. That trades a lot of signal for
speed — there's no cross-attention between the query tokens and the
chunk tokens, so subtle topical mismatches (e.g. a chunk that *mentions*
"debouncing" but is really about cache invalidation) score high.

Symptom observed in the dotfiles repo: queries like *"where do we
configure the embedding model"* return the correct chunk in the top-8
but not at position 1. A cross-encoder that scores `(query, chunk)`
pairs jointly fixes this with no index changes.

Three things are missing:

1. **A rerank endpoint client** alongside `internal/embed/` and
   `internal/chat/`. Cohere-style `/v1/rerank` is the de facto
   wire shape (TEI, Jina, Cohere itself, and vLLM with a reranker
   adapter all speak it).
2. **A pluggable stage in `store.Search`** that runs after RRF fusion
   on a larger candidate pool, then truncates to `k`. Today the
   pool→top-k cut happens directly after fusion; we want
   pool → fuse → rerank → top-k.
3. **Graceful degradation.** Rerank must be optional: when the URL is
   unset or the endpoint is unreachable, search must return today's
   fused results unchanged. Mirror `embed.ErrUnreachable`.

---

## Goals

- **G1** New package `internal/rerank/` with a `Client` exposing
  `Rerank(ctx, query string, docs []string) ([]Score, error)`. Cohere
  `/v1/rerank` wire format. Returns `ErrUnreachable` on dial/network
  errors (mirrors `embed.ErrUnreachable`).
- **G2** Plug into `store.Search`: when a `*rerank.Client` is wired
  into `Store.opts`, the existing fused pool (size `5×k`, capped) is
  reranked by the client, then truncated to `k`. When the client is
  nil or returns `ErrUnreachable`, the existing post-fusion truncation
  runs as today (no behavioral change).
- **G3** Surface `RerankScore` on `Hit` (parallel to `Score`,
  `BM25Score`, `RRFScore`). Propagate to `mcp.SearchHit` so callers
  can see why a chunk surfaced.
- **G4** Config: `MCSEARCH_RERANK_URL`, `MCSEARCH_RERANK_MODEL`,
  `MCSEARCH_RERANK_POOL` (candidate count before rerank, default
  `5×k`, capped 100), `MCSEARCH_RERANK_TIMEOUT` (default 5s),
  `MCSEARCH_DISABLE_RERANK=1` (kill switch).
- **G5** CLI flag `--rerank=off` on `mcsearch query` for ablation.
- **G6** A/B harness: a thin `mcsearch eval` subcommand (or test) that
  runs a fixed query set with and without rerank against the same
  index and prints MRR@10 + nDCG@10 deltas. Optional in v1 but
  designed for: the wiring point in `store.Search` makes both modes
  callable.

**Out of scope (separate spec):**

- Rewriting RRF — the cross-encoder reranks the *fused* list, not the
  individual semantic/BM25 lists. RRF stays.
- Learned sparse retrieval (SPLADE), ColBERT-style late interaction,
  or any reranker that needs index-side changes.
- Reranking inside `generate_code`'s RAG context selection — same
  store, same client, but a separate decision (does the chat model
  benefit from rerank as much as the human/agent reader does?).
  Punt to a follow-up after we see v1 query metrics.
- Server-side provisioning of the rerank endpoint (that lives in
  [dotfiles `components/mcsearch/server.yml`]). This spec
  defines the *client*; the dotfiles change is a one-line
  `ollama pull` or TEI container alongside the existing ollama
  service.

---

## Design

### Wire format

Cohere `/v1/rerank` — request:

```json
{
  "model": "qwen3-reranker:4b",
  "query": "where do we debounce filesystem events",
  "documents": ["...chunk 1...", "...chunk 2...", "..."],
  "top_n": 8,
  "return_documents": false
}
```

Response:

```json
{
  "results": [
    {"index": 17, "relevance_score": 0.94},
    {"index":  3, "relevance_score": 0.71},
    ...
  ],
  "model": "qwen3-reranker:4b"
}
```

`index` is the position in the request `documents` array, not a chunk
ID — the client maps back. `top_n` lets the server cap network egress;
we'll pass `pool` (not `k`) so the caller-side truncation is the source
of truth.

### `internal/rerank/client.go`

```go
package rerank

import (
    "context"
    "errors"
    "net/http"
    "time"
)

var ErrUnreachable = errors.New("rerank service unreachable")

type Client struct {
    BaseURL string
    Model   string
    HTTP    *http.Client
}

// Score is one (document-index, relevance) pair returned by the server,
// in descending relevance order.
type Score struct {
    Index int
    Score float32
}

func New(baseURL, model string, timeout time.Duration) *Client { ... }

// Rerank sends (query, docs) to /v1/rerank and returns the server's
// ordering of indices into `docs` along with relevance scores. The
// returned slice is sorted descending by Score.
//
// Returns ErrUnreachable for dial/timeout failures so the caller can
// degrade to non-reranked results.
func (c *Client) Rerank(ctx context.Context, query string, docs []string) ([]Score, error) { ... }
```

Mirrors `embed.Client` deliberately: same struct shape, same
`ErrUnreachable`, same `New` signature. No batching loop — the server
takes the full doc list in one POST (pool ≤ 100).

### `store.Search` integration

Today, after RRF:

```go
sort.Slice(fused, func(i, j int) bool { return fused[i].score > fused[j].score })
if len(fused) > k {
    fused = fused[:k]
}
return s.fetchHits(ctx, fused, semCosine, bm25Score)
```

Becomes:

```go
sort.Slice(fused, func(i, j int) bool { return fused[i].score > fused[j].score })

// Rerank stage: only if a client is wired and we have more
// candidates than k (otherwise it's a no-op cost).
if s.opts.Reranker != nil && len(fused) > k {
    reranked, err := s.rerank(ctx, queryText, fused, k)
    switch {
    case err == nil:
        return s.fetchHits(ctx, reranked, semCosine, bm25Score) // populates RerankScore
    case errors.Is(err, rerank.ErrUnreachable):
        // Fall through to non-reranked truncation. Log once per process.
    default:
        return nil, err
    }
}
if len(fused) > k {
    fused = fused[:k]
}
return s.fetchHits(ctx, fused, semCosine, bm25Score)
```

The `s.rerank` helper fetches `Content` for the pool (cheap — one
`SELECT ... WHERE id IN (?)`), POSTs to the client, maps result
indices back to chunk IDs, and returns the top-k slice.

`Reranker rerank.Reranker` is added to `store.Options` as an
interface (not the concrete `*rerank.Client`), so tests can swap in a
deterministic stub.

### `Hit.RerankScore`

```go
type Hit struct {
    // ... existing fields ...

    // RerankScore is the cross-encoder relevance score in [0, 1] for
    // the (query, chunk) pair. Zero when rerank didn't run (no client
    // wired, pool ≤ k, or endpoint unreachable). Larger = more relevant.
    RerankScore float32
}
```

Plumbs through to `mcp.SearchHit` and the CLI `mcsearch query` output.

### Config & CLI

| Env var                       | Default                        | Notes                                                        |
| ----------------------------- | ------------------------------ | ------------------------------------------------------------ |
| `MCSEARCH_RERANK_URL`         | *(empty)*                      | Empty → rerank disabled. Set to enable.                      |
| `MCSEARCH_RERANK_MODEL`       | `qwen3-reranker:4b`            | Recommendation; any Cohere-compat model works.               |
| `MCSEARCH_RERANK_POOL`        | `40` (`max(5×k, 30)`, cap 100) | Candidates fetched before rerank. Larger = better recall but slower. |
| `MCSEARCH_RERANK_TIMEOUT`     | `5s`                           | HTTP timeout. On expiry → `ErrUnreachable` path.             |
| `MCSEARCH_DISABLE_RERANK`     | unset                          | `=1` short-circuits the stage even if URL is set. For A/B.   |

CLI: `mcsearch query --rerank=off ...` toggles per invocation
(equivalent to setting `MCSEARCH_DISABLE_RERANK=1`).

### Model choice

Recommend `qwen3-reranker:4b` as default — same family as the
default embedder (`qwen3-embedding:4b`), trained on the same data
distribution, ~4 GB VRAM, runs alongside the embedder on a single
5090. Alternatives:

- `bge-reranker-v2-m3` (568M params, fastest, multilingual, slightly
  lower English code-search quality)
- `Qwen3-Reranker-8B` (better quality, ~8 GB, fine on 5090)

ollama does not currently serve rerankers natively (no `/v1/rerank`
endpoint). Three deployment options for the server side:

1. **TEI** (Text Embeddings Inference) — Hugging Face's serving stack,
   speaks Cohere wire format on `/rerank`. Same docker model as TEI's
   embedding mode.
2. **vLLM with a reranker model** — recent vLLM ships a `/rerank`
   endpoint when started with `--task=score`. Heavier dep, but already
   in some users' stacks.
3. **Cohere API directly** — for users who want to skip self-hosting.
   Same wire format; just point `MCSEARCH_RERANK_URL` at
   `https://api.cohere.com` and add an `Authorization` header (new
   `MCSEARCH_RERANK_AUTH` env var — minor addition to the client).

This spec is agnostic; the dotfiles follow-up picks one. Likely TEI,
to keep ollama as the embedding/chat path and add a second container
just for rerank.

---

## Migration / compatibility

- **Off by default.** Empty `MCSEARCH_RERANK_URL` ≡ today's behaviour.
- **Existing indexes are unchanged.** Reranker is purely a query-time
  reordering — no schema migration, no re-embedding.
- **`Hit.RerankScore` is a new field**, additive. Existing JSON
  consumers (the MCP server, `mcsearch query` text output) keep
  working; new field appears as `0` when rerank didn't run.
- **No breaking changes to `store.Options`.** The `Reranker` field is
  optional; nil means "skip the stage".

---

## Risks / open questions

1. **Latency.** A 4B cross-encoder on 40 docs is ~100–300ms on a 5090.
   That's a 2–4× hit on query time (today's median is ~50ms warm).
   Mitigations: (a) cap pool at 40 for the default, (b) short timeout
   (5s), (c) easy off switch. For interactive `mcsearch query` users
   this is fine; for the MCP path Claude is already waiting on its
   own model latency, so 200ms is in the noise.
2. **Pool size tuning.** 40 is a guess. The `mcsearch eval` harness
   (G6) is what justifies the default; v1 ships with 40 and we move
   it based on data.
3. **Score scale.** Reranker scores from different models aren't
   comparable. Surface them as-is (in [0, 1] for most models) but
   don't try to fuse them with cosine — the cross-encoder is the
   final word once it's run.
4. **Cohere `Authorization` header.** If we want first-class hosted
   Cohere support in v1, the client needs `MCSEARCH_RERANK_AUTH`.
   Small addition; flagged here so it's not a surprise during review.
5. **Generate-code RAG context.** Out of scope here, but the same
   reranker should plausibly improve RAG selection in
   `generate_code`. Punt to a follow-up so v1 stays focused.

---

## Phases

| Phase | Scope                                                                                                       | Done when                                                                                       |
| ----- | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **1** | `internal/rerank/` package: `Client`, `Score`, `New`, `Rerank`, `ErrUnreachable`. Unit tests with httptest. | `go test ./internal/rerank/...` green.                                                          |
| **2** | `Store.Options.Reranker` field; `store.Search` rerank stage; `Hit.RerankScore`. Store tests cover both paths. | `go test ./internal/store/...` green; rerank stage exercised by a stub `Reranker` in test.      |
| **3** | Wire env vars in `cmd/mcsearch/main.go`. Plumb `*rerank.Client` from MCP server + CLI `query` into `Store.Options`. | `MCSEARCH_RERANK_URL=… mcsearch query ./ "…"` reorders results visibly; unset → no behaviour change. |
| **4** | Surface `RerankScore` in MCP `SearchHit` JSON and `mcsearch query` text output.                             | A reranked hit shows a non-zero `rerank_score` in `mcsearch query --format=json`.               |
| **5** | `mcsearch eval` (optional in v1): fixed query set, MRR/nDCG with and without rerank.                        | Eval prints both columns side-by-side for the canonical dataset.                                |
| **6** | Dotfiles companion change: provision rerank endpoint in `components/mcsearch/server.yml` (probably TEI).    | `mcsearch status` on a freshly-deployed `main_pc` reports rerank endpoint healthy.              |

Phases 1–4 are the spec. Phase 5 is the justification for tuning
defaults. Phase 6 is a separate PR in the dotfiles repo.

---

## Touched files (rough)

- **new:** `internal/rerank/client.go`, `internal/rerank/client_test.go`
- **`internal/store/store.go`** — `Options.Reranker`, `Search` rerank
  stage, `Hit.RerankScore`, `fetchHits` populates the new field
- **`internal/store/store_test.go`** — stub `Reranker`, two new tests
  (happy path reordering + unreachable fallback)
- **`internal/mcp/server.go`** — `SearchHit.RerankScore`, optional
  `*rerank.Client` field on `Server`, wire into `StoreOpts`
- **`cmd/mcsearch/main.go`** — env-var parsing, `--rerank=off` flag on
  `query`, `status` reports rerank endpoint health (parallel to embed/chat)
- **`README.md`** — env-var table row, one-paragraph rerank section
