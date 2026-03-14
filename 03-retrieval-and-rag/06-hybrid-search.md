# Hybrid Search

> **TL;DR**: Hybrid search combines dense vector search with sparse keyword search (BM25). It consistently outperforms either approach alone by 5-15% on recall. The fusion algorithm is RRF (Reciprocal Rank Fusion): combine ranked lists by position, no score calibration needed. Default `alpha=0.5` (equal weight) is a solid starting point. Tune toward BM25 for keyword-heavy queries, toward dense for semantic ones.

**Prerequisites**: [RAG Fundamentals](01-rag-fundamentals.md), [Vector Indexing](03-vector-indexing.md), [Embedding Models](02-embedding-models.md)
**Related**: [Reranking](07-reranking.md), [Advanced RAG Patterns](09-advanced-rag-patterns.md), [Vector Databases](04-vector-databases.md)

---

## Why Hybrid Search Exists

Dense vector search is powerful but has a specific weakness: it can miss exact matches. If a user searches for "RFC 2616" or "Error code 0x80070005" or "model-api-v2", these are precise identifiers that don't benefit from semantic similarity. The relevant document contains "RFC 2616" as a string. Embedding similarity finds documents about HTTP, not the specific RFC.

BM25 (a term frequency-based algorithm) handles exact keywords perfectly. It also handles rare technical terms, acronyms, and proper nouns that might not be well-represented in embedding space.

The combination beats both approaches because different queries have different needs:
- "What is the return policy?" → semantic similarity works well
- "Error ECONNREFUSED 111" → exact keyword matching works better
- "How does OAuth2 handle token refresh?" → both matter

A [study by Pinecone](https://www.pinecone.io/learn/hybrid-search/) found that hybrid search improved recall@10 by an average of 10-15% across diverse query types compared to pure dense search.

---

## BM25: The Sparse Component

BM25 (Best Match 25) is a probabilistic term-frequency ranking algorithm. It scores documents by how often query terms appear, normalized by document length and term rarity across the corpus.

```python
from rank_bm25 import BM25Okapi
import re

def tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())

# Build BM25 index
corpus = ["Machine learning is a subset of AI", "Deep learning uses neural networks", ...]
tokenized_corpus = [tokenize(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Search
query_tokens = tokenize("neural network architectures")
scores = bm25.get_scores(query_tokens)
top_indices = scores.argsort()[::-1][:10]
```

BM25's strengths:
- Zero-shot performance: no training needed, works out of the box
- Exact keyword matching: finds "GDPR Article 17" if that exact string is in the document
- Interpretable: you can see which terms contributed to the score

BM25's weaknesses:
- No semantic understanding: misses synonyms and paraphrases
- Language-dependent: requires language-specific tokenization for best results
- No vector search: can't capture meaning beyond term overlap

---

## SPLADE: Sparse + Semantic

SPLADE (Sparse Lexical and Expansion model) is a learned sparse model that bridges BM25 and dense search. It expands queries and documents with semantically related terms before sparse matching.

Where BM25 represents "machine learning" as `{"machine": 1, "learning": 1}`, SPLADE might represent it as `{"machine": 0.8, "learning": 0.7, "ai": 0.4, "neural": 0.3, "algorithm": 0.3}`.

SPLADE generally outperforms BM25 and in some benchmarks approaches dense model quality. It's supported in Weaviate and can be run locally. If you're investing in the sparse component, consider SPLADE over BM25.

---

## Fusion: Combining Dense and Sparse Results

The two search methods return ranked lists. You need to combine them into a single ranking.

### Reciprocal Rank Fusion (RRF)

RRF is the standard approach. It ranks documents by position across multiple ranked lists, not by raw score values. This avoids the score calibration problem (BM25 scores and cosine similarities are on completely different scales).

```python
def reciprocal_rank_fusion(
    result_lists: list[list[str]],  # each list is ranked doc IDs
    k: int = 60  # RRF parameter, 60 is standard
) -> list[str]:
    """Merge multiple ranked lists using RRF."""
    scores: dict[str, float] = {}
    for ranked_list in result_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

# Usage
dense_results = ["doc-3", "doc-1", "doc-5", "doc-2", "doc-4"]  # ranked by cosine similarity
sparse_results = ["doc-1", "doc-4", "doc-3", "doc-6", "doc-2"]  # ranked by BM25

fused = reciprocal_rank_fusion([dense_results, sparse_results])
# doc-1 appears at rank 2 in dense and rank 1 in sparse -> high fused score
# doc-3 appears at rank 1 in dense and rank 3 in sparse -> high fused score
```

The RRF `k` parameter (default 60) controls how much rank position matters vs absolute rank. Higher k = more weight to absolute position. In practice, k=60 performs well across diverse tasks without tuning.

### Linear Combination (Alpha Weighting)

Instead of RRF, you can normalize scores and linearly combine them:

```python
def alpha_fusion(
    dense_results: list[tuple[str, float]],   # (doc_id, cosine_score)
    sparse_results: list[tuple[str, float]],  # (doc_id, bm25_score)
    alpha: float = 0.5  # 0=pure sparse, 1=pure dense
) -> list[str]:
    # Normalize scores to [0, 1]
    def normalize(results):
        if not results: return {}
        scores = [s for _, s in results]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s: return {id: 1.0 for id, _ in results}
        return {id: (s - min_s) / (max_s - min_s) for id, s in results}

    dense_norm = normalize(dense_results)
    sparse_norm = normalize(sparse_results)

    all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
    combined = {id: alpha * dense_norm.get(id, 0) + (1-alpha) * sparse_norm.get(id, 0)
                for id in all_ids}
    return sorted(combined.keys(), key=lambda x: combined[x], reverse=True)
```

Alpha weighting requires careful normalization. BM25 scores and cosine similarities are on different scales and different distributions. Poor normalization leads to one method dominating regardless of alpha.

**RRF vs Alpha:**
- RRF: simpler, no normalization needed, more robust to score distribution differences
- Alpha: more tunable, can express "I want 70% dense" precisely

Use RRF as the default. Only switch to alpha weighting if you need precise control over the tradeoff and you've validated your normalization.

---

## Full Pipeline: Dense + BM25 + RRF

```python
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, corpus: list[str], top_k: int = 10):
        self.corpus = corpus
        self.top_k = top_k

        # Dense index
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.embed_model.encode(corpus, normalize_embeddings=True)

        # Sparse index
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str) -> list[str]:
        # Dense search
        q_emb = self.embed_model.encode([query], normalize_embeddings=True)[0]
        dense_scores = self.embeddings @ q_emb
        dense_ranked = np.argsort(dense_scores)[::-1][:self.top_k].tolist()

        # Sparse search
        bm25_scores = self.bm25.get_scores(query.lower().split())
        sparse_ranked = np.argsort(bm25_scores)[::-1][:self.top_k].tolist()

        # RRF fusion
        fused = reciprocal_rank_fusion([
            [str(i) for i in dense_ranked],
            [str(i) for i in sparse_ranked]
        ])
        return [self.corpus[int(i)] for i in fused[:self.top_k]]
```

---

## Hybrid Search in Vector Databases

Most production systems use the built-in hybrid search from their vector database rather than implementing it from scratch.

**Weaviate (built-in hybrid):**
```python
results = collection.query.hybrid(
    query="OAuth token refresh",
    alpha=0.5,     # 0=BM25 only, 1=dense only
    limit=5
)
```

**Qdrant (with sparse vectors):**
Qdrant supports sparse vectors natively, letting you store BM25 vectors alongside dense vectors and fuse at query time.

**Pinecone:**
Pinecone's hybrid search requires creating a "hybrid index" that stores both dense and sparse vectors. It uses alpha weighting for fusion.

---

## When to Weight Toward BM25 vs Dense

| Query Type | Weight Toward | Why |
|---|---|---|
| Exact identifiers (error codes, model names, IDs) | BM25 | Exact keyword match critical |
| Technical queries with specific terms | BM25 | Rare terms need exact matching |
| Natural language questions | Dense | Semantic understanding matters |
| Short factual queries | Dense | Meaning matters more than terms |
| Code search | BM25 | Function/variable names are exact |
| Legal/medical terminology | Equal | Both exact terms and context matter |

A simple routing approach: classify the query as "keyword-heavy" or "semantic" and use different alpha values rather than one universal setting.

---

## Concrete Numbers

As of early 2025, on a standard retrieval benchmark:

| Method | Recall@10 | Latency | Cost |
|---|---|---|---|
| BM25 only | ~55-65% | 5-20ms | Minimal |
| Dense only | ~62-70% | 10-30ms | Embedding cost |
| Hybrid (RRF) | ~70-80% | 15-40ms | Embedding + BM25 |
| Hybrid + reranker | ~80-90% | 200-500ms | + Reranker API cost |

*Recall numbers vary significantly by dataset. Test on your specific data.*

The 5-15% recall improvement from hybrid over dense alone is consistent across multiple benchmarks. The latency overhead is minimal (BM25 is fast). The cost overhead is also minimal (BM25 requires no API calls). Hybrid search has one of the best ROI profiles of any retrieval optimization.

---

## Gotchas

**BM25 needs good tokenization.** The default whitespace tokenizer is weak. For technical content, add a tokenizer that handles camelCase, underscores, and punctuation. "getUserById" should tokenize to ["get", "user", "by", "id"].

**Hybrid doesn't help for purely semantic queries.** If your users always ask natural language questions ("how does X work?"), adding BM25 doesn't hurt but doesn't help much either. Measure the recall improvement on your actual query distribution before investing.

**Index freshness synchronization.** Dense and sparse indexes must update together. If you add a document to the dense index but forget to update BM25, documents from that update are invisible to sparse search. Use atomic operations or validate index consistency periodically.

**SPLADE requires more compute than BM25.** SPLADE is a neural model that needs inference at index time and query time. The better quality comes at higher computational cost. Budget accordingly.

---

> **Key Takeaways:**
> 1. Hybrid search (dense + BM25) consistently outperforms either alone by 5-15% recall. The ROI is high: BM25 is cheap to add and RRF requires no score calibration.
> 2. Use RRF as your fusion default: no normalization required, robust across different datasets.
> 3. Weight toward BM25 for keyword-heavy queries (error codes, product names, exact identifiers), toward dense for natural language and conceptual queries.
>
> *"Pure vector search is like a library with only semantic card catalog. Pure BM25 is like ctrl+F on every document. Hybrid is the reference librarian."*

---

## Interview Questions

**Q: Your RAG system works well for natural language questions but fails on queries like product SKUs and error codes. How do you fix it?**

This is the classic case for hybrid search. Dense embeddings capture semantic similarity, but exact identifiers like "SKU-78423" or "Error E_CONN_RESET_0x1234" have low semantic signal. Two documents can be about completely different things but both happen to contain that exact string.

BM25 handles this perfectly because it's term-frequency based: if "SKU-78423" appears in one document and not others, that document gets a very high BM25 score for the query "SKU-78423".

My fix: add BM25 search alongside the existing dense search and combine results with RRF. For the existing dense index (Pinecone or Weaviate), I'd add a BM25 index over the same corpus. At query time, run both searches, combine with RRF.

If the system is using Weaviate, hybrid search is built in and the change is adding `alpha=0.5` to the query call. For Pinecone, I'd either switch to Pinecone's hybrid index type or run BM25 externally with rank_bm25 and do RRF manually.

For queries I know are identifier-heavy (detected by the presence of numbers, special characters, or uppercase sequences), I'd dynamically set alpha lower (0.2-0.3) to weight BM25 more.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| What is BM25? | A term-frequency based ranking algorithm; gives high scores to documents where rare query terms appear frequently |
| What is RRF? | Reciprocal Rank Fusion: merges ranked lists by position, not score; `score = sum(1/(k + rank))` |
| What does the `alpha` parameter control in hybrid search? | 0=pure BM25, 1=pure dense; 0.5 is equal weighting |
| Why is RRF preferred over linear score combination? | No score normalization needed; robust when dense and sparse scores have different distributions |
| What is SPLADE? | A learned sparse model that expands query/document terms semantically; outperforms BM25 |
| What query type benefits most from BM25 in hybrid search? | Exact identifiers, error codes, product SKUs, rare technical terms |
