# ✅ RAG & Retrieval — Interview Checklist

> **TL;DR**: This is a **syllabus, not notes**. Its only job is to make sure nothing falls through the cracks before an interview. Study each item in the linked lesson (or your own source), then tick the box for the depth you've actually reached. Your target for the high-leverage (⭐) topics is the **third** column — *Design/Debug* — not the first.

**How to use this page**

Track every topic at three levels of mastery. Don't mark something "done" just because you *read* it:

| Level | Box | You can honestly say... |
|---|---|---|
| **L1 — Learned** | `☐` | "I've read it and I understand the idea." |
| **L2 — Explain** | `☐` | "I can explain it out loud, unprompted, in an interview." |
| **L3 — Design/Debug** | `☐` | "I can make the tradeoff calls and debug it in a live system." |

Tick a box by replacing `☐` with `✅`. ⭐ = high interview leverage — get these to **L3** first.

**Related chapters**: [Embedding Models](02-embedding-models.md) · [Vector Indexing](03-vector-indexing.md) · [Chunking](05-chunking-strategies.md) · [RAG Evaluation](11-rag-evaluation.md)

---

## 1. Foundations

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| RAG architecture (why retrieve instead of fine-tune) ⭐ | ☐ | ☐ | ☐ | [What is RAG?](what-is-rag.md) · [RAG Fundamentals](01-rag-fundamentals.md) |
| Ingestion → retrieval → generation pipeline ⭐ | ☐ | ☐ | ☐ | [RAG Fundamentals](01-rag-fundamentals.md) |
| Embeddings — what a vector actually encodes | ☐ | ☐ | ☐ | [Embedding Models](02-embedding-models.md) |
| Similarity metrics: cosine vs dot-product vs Euclidean | ☐ | ☐ | ☐ | [Embedding Models](02-embedding-models.md) |
| Vector databases — what they are and when you need one | ☐ | ☐ | ☐ | [Vector Databases](04-vector-databases.md) |
| Dense vs sparse retrieval ⭐ | ☐ | ☐ | ☐ | [Hybrid Search](06-hybrid-search.md) |

## 2. Document Processing

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Parsing (PDF/HTML/docx) and cleaning | ☐ | ☐ | ☐ | [Chunking Strategies](05-chunking-strategies.md) |
| Metadata extraction and attachment | ☐ | ☐ | ☐ | [Chunking Strategies](05-chunking-strategies.md) |
| Chunking: fixed / token / semantic / recursive ⭐ | ☐ | ☐ | ☐ | [Chunking Strategies](05-chunking-strategies.md) |
| Chunk size & overlap tradeoffs ⭐ | ☐ | ☐ | ☐ | [Chunking Strategies](05-chunking-strategies.md) |
| Parent-child chunks | ☐ | ☐ | ☐ | [Chunking Strategies](05-chunking-strategies.md) |
| Contextual retrieval (Anthropic-style context prefixing) | ☐ | ☐ | ☐ | [Chunking Strategies](05-chunking-strategies.md) |
| Handling tables, PDFs, and code | ☐ | ☐ | ☐ | [Multimodal RAG](10-multimodal-rag.md) |

## 3. Retrieval

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Semantic (dense) search | ☐ | ☐ | ☐ | [RAG Fundamentals](01-rag-fundamentals.md) |
| BM25 / keyword retrieval | ☐ | ☐ | ☐ | [Hybrid Search](06-hybrid-search.md) |
| Hybrid search + fusion (RRF) ⭐ | ☐ | ☐ | ☐ | [Hybrid Search](06-hybrid-search.md) |
| Metadata filtering | ☐ | ☐ | ☐ | [Vector Databases](04-vector-databases.md) |
| Query rewriting | ☐ | ☐ | ☐ | [Query Transformation](08-query-transformation.md) |
| Multi-query retrieval | ☐ | ☐ | ☐ | [Query Transformation](08-query-transformation.md) |
| HyDE (hypothetical document embeddings) | ☐ | ☐ | ☐ | [Query Transformation](08-query-transformation.md) |
| Query decomposition | ☐ | ☐ | ☐ | [Query Transformation](08-query-transformation.md) |
| Routing | ☐ | ☐ | ☐ | [Query Transformation](08-query-transformation.md) |
| MMR (maximal marginal relevance) & top-k selection | ☐ | ☐ | ☐ | [Reranking](07-reranking.md) |

## 4. Vector Search Internals

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| ANN vs exact search — why approximate ⭐ | ☐ | ☐ | ☐ | [Vector Indexing](03-vector-indexing.md) |
| HNSW ⭐ | ☐ | ☐ | ☐ | [Vector Indexing](03-vector-indexing.md) |
| IVF | ☐ | ☐ | ☐ | [Vector Indexing](03-vector-indexing.md) |
| Product Quantization (PQ) | ☐ | ☐ | ☐ | [Vector Indexing](03-vector-indexing.md) |
| Indexing tradeoffs: recall vs latency vs memory ⭐ | ☐ | ☐ | ☐ | [Vector Indexing](03-vector-indexing.md) |
| Embedding dimensionality (& Matryoshka) | ☐ | ☐ | ☐ | [Embedding Models](02-embedding-models.md) |
| Index updates / incremental indexing | ☐ | ☐ | ☐ | [Vector Databases](04-vector-databases.md) |

## 5. Advanced RAG

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Cross-encoder reranking ⭐ | ☐ | ☐ | ☐ | [Reranking](07-reranking.md) |
| Late interaction / ColBERT concepts | ☐ | ☐ | ☐ | [Reranking](07-reranking.md) |
| Contextual compression | ☐ | ☐ | ☐ | [Query Transformation](08-query-transformation.md) |
| Parent-document retrieval | ☐ | ☐ | ☐ | [Chunking Strategies](05-chunking-strategies.md) |
| GraphRAG | ☐ | ☐ | ☐ | [Advanced RAG Patterns](09-advanced-rag-patterns.md) |
| Multimodal RAG (ColPali, image+text) | ☐ | ☐ | ☐ | [Multimodal RAG](10-multimodal-rag.md) |
| Agentic RAG | ☐ | ☐ | ☐ | [Advanced RAG Patterns](09-advanced-rag-patterns.md) |
| Corrective RAG (CRAG) & Self-RAG concepts | ☐ | ☐ | ☐ | [Advanced RAG Patterns](09-advanced-rag-patterns.md) |

## 6. Production

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Citations & grounding ⭐ | ☐ | ☐ | ☐ | [RAG Fundamentals](01-rag-fundamentals.md) |
| ACL-aware retrieval (permission-filtered) ⭐ | ☐ | ☐ | ☐ | [Case: Enterprise RAG](../07-system-design-interview/04-case-enterprise-rag.md) |
| Multi-tenancy | ☐ | ☐ | ☐ | [Case: Enterprise RAG](../07-system-design-interview/04-case-enterprise-rag.md) |
| Freshness & re-indexing | ☐ | ☐ | ☐ | [Vector Databases](04-vector-databases.md) |
| Ingestion pipelines | ☐ | ☐ | ☐ | [RAG Fundamentals](01-rag-fundamentals.md) |
| Deduplication | ☐ | ☐ | ☐ | [Chunking Strategies](05-chunking-strategies.md) |
| Caching (embedding & retrieval) | ☐ | ☐ | ☐ | [Caching Strategies](../06-production-and-ops/03-caching-strategies.md) |
| Latency & cost budgets ⭐ | ☐ | ☐ | ☐ | [Cost Optimization](../06-production-and-ops/07-cost-optimization.md) |
| Retrieval debugging | ☐ | ☐ | ☐ | [RAG Evaluation](11-rag-evaluation.md) |

## 7. Evaluation

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| recall@k / precision@k ⭐ | ☐ | ☐ | ☐ | [RAG Evaluation](11-rag-evaluation.md) |
| MRR | ☐ | ☐ | ☐ | [Retrieval & RAG Eval](../05-evaluation/02-retrieval-and-rag-eval.md) |
| NDCG | ☐ | ☐ | ☐ | [Retrieval & RAG Eval](../05-evaluation/02-retrieval-and-rag-eval.md) |
| Context relevance | ☐ | ☐ | ☐ | [RAG Evaluation](11-rag-evaluation.md) |
| Faithfulness / groundedness ⭐ | ☐ | ☐ | ☐ | [RAG Evaluation](11-rag-evaluation.md) |
| Answer correctness | ☐ | ☐ | ☐ | [RAG Evaluation](11-rag-evaluation.md) |
| Golden datasets | ☐ | ☐ | ☐ | [Eval Fundamentals](../05-evaluation/01-eval-fundamentals.md) |

---

**Done with this chapter?** Every ⭐ topic at L3, no empty rows left → you can defend a RAG design end-to-end. Next: [Agents Checklist](../04-agents-and-orchestration/00-checklist.md).
