# Architecture

```mermaid
flowchart TB
  subgraph Ingest["Indexing pipeline"]
    src[Source files] --> ts[tree-sitter chunker<br/>functions / types / docs]
    ts --> emb[/Self-hosted /v1/embeddings<br/>ollama · vLLM · TEI/]
    ts --> graph[Go call-graph builder]
    emb --> store[(SQLite vector store)]
    graph --> store
    ts --> bm25[BM25 index]
    bm25 --> store
  end

  subgraph Query["Query path: ask"]
    q[Free-text question] --> router{Intent router<br/>behavior · symbol · callers<br/>callees · architecture · etc.}
    router --> sem[search_semantic<br/>cosine]
    router --> sym[search_symbol]
    router --> gx[graph expansion<br/>incl. calls edges]
    sem --> rrf[RRF fusion<br/>cosine + BM25]
    bm25 -.-> rrf
    rrf --> rerank[Cross-encoder rerank<br/>optional]
    rerank --> bundle
    sym --> bundle
    gx --> bundle
    bundle[Compact bundle<br/>semantic_hits · symbols · graph<br/>suggested_reads · next_action · avoid]
  end

  store --> sem
  store --> sym
  store --> gx

  bundle --> client[Claude / CLI<br/>via MCP stdio]
```

Indexing happens once into SQLite; at query time the router fans out across semantic + lexical + graph legs, fuses with RRF, optionally reranks, and returns reading instructions rather than raw files.
