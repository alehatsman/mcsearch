# Tuning knobs

The five env vars listed in the main [README](../README.md#environment)
are enough to run mcsearch. This page collects the rest — knobs you
only need when you're tuning latency, capping RAM, or wiring optional
features (rerank, compress, draft).

`mcsearch env --all --doc` prints the same list with the *currently
active* values and where each came from (`env` / `default` /
`unset` / `disabled`), which is usually the fastest way to debug "why
is rerank off."

## Optional features

These are unset by default — set the URL to enable the feature.

| Variable                  | Default | What it does                                                                 |
| ------------------------- | ------- | ---------------------------------------------------------------------------- |
| `MCSEARCH_RERANK_URL`     | unset   | Cross-encoder rerank server. Reorders the fused candidate pool before truncating to `k`. Big quality lift on conceptual queries, ~100 ms latency. |
| `MCSEARCH_COMPRESS_URL`   | unset   | Context-compression `/v1/chat/completions` server. Distils RAG chunks before generation (4–5× token savings). |
| `MCSEARCH_DRAFT_URL`      | unset   | Speculative-draft `/v1/chat/completions` server. Pre-generates code so the main chat model refines instead of writing from scratch (3–10× generation token savings). |

## Rerank tuning

Only relevant when `MCSEARCH_RERANK_URL` is set.

| Variable                       | Default                     | Notes                                                                                              |
| ------------------------------ | --------------------------- | -------------------------------------------------------------------------------------------------- |
| `MCSEARCH_RERANK_STYLE`        | `cohere`                    | `cohere` for Cohere-shape `/rerank` (TEI, Infinity, vLLM cross-encoder); `chat` for decoder-style via `/v1/chat/completions` + logprobs (e.g. Qwen3-Reranker-4B on vLLM). |
| `MCSEARCH_RERANK_MODEL`        | `BAAI/bge-reranker-v2-m3`   | Model name forwarded to the reranker.                                                              |
| `MCSEARCH_RERANK_POOL`         | `40`                        | Fused candidates fed to the reranker. Clamped to `[1, 100]`. Larger pool = better recall, slower. |
| `MCSEARCH_RERANK_TIMEOUT`      | `5s`                        | HTTP timeout per rerank call.                                                                      |
| `MCSEARCH_RERANK_CONCURRENCY`  | `4`                         | Parallel scoring goroutines (`chat` style only). Try 8–16 on a dedicated GPU. Ignored for `cohere`. |
| `MCSEARCH_DISABLE_RERANK`      | unset                       | Set `1` to short-circuit rerank even when URL is set. For A/B comparison.                          |

## Compress / draft model overrides

Each defaults to `MCSEARCH_CHAT_MODEL` — set explicitly when the
compress/draft endpoint serves a different model.

| Variable                  | Default                  | Notes                                              |
| ------------------------- | ------------------------ | -------------------------------------------------- |
| `MCSEARCH_COMPRESS_MODEL` | value of `*_CHAT_MODEL`  | Model for the compress leg.                        |
| `MCSEARCH_COMPRESS_TIMEOUT` | `30s`                  | HTTP timeout per compress call.                    |
| `MCSEARCH_DRAFT_MODEL`    | value of `*_CHAT_MODEL`  | Model for the draft leg.                           |
| `MCSEARCH_DRAFT_TIMEOUT`  | `120s`                   | HTTP timeout per draft call.                       |

## Timeouts & batching

Defaults are sized for local Ollama / vLLM. Bump these only if you see
spurious timeouts (`context deadline exceeded`) or want to push
batch throughput.

| Variable                  | Default | Notes                                       |
| ------------------------- | ------- | ------------------------------------------- |
| `MCSEARCH_EMBED_TIMEOUT`  | `60s`   | HTTP timeout per `/v1/embeddings` call.     |
| `MCSEARCH_EMBED_BATCH`    | `32`    | Max chunks per `/v1/embeddings` call.       |
| `MCSEARCH_CHAT_TIMEOUT`   | `120s`  | HTTP timeout per `/v1/chat/completions` call. |

## Index / search behavior

| Variable                     | Default      | Notes                                                                                                                                                                     |
| ---------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MCSEARCH_DISABLE_BM25`      | unset        | Set `1` to skip the BM25 leg of hybrid search and rank by cosine similarity alone.                                                                                        |
| `MCSEARCH_MAX_HITS_PER_FILE` | unset (no cap) | Positive integer caps how many search hits come from a single file. Promotes result diversity.                                                                          |
| `MCSEARCH_ALLOW_PATHS`       | unset        | Colon-separated path prefixes (`:` on POSIX, `;` on Windows) that `index`/`watch` accept even when the target isn't inside a git work tree. Entries support `~` / `$HOME`. |
