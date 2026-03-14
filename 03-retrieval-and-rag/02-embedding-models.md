# Embedding Models

> **TL;DR**: Embedding models convert text to dense vectors. The model you pick determines retrieval quality more than the vector database or search algorithm. Start with `text-embedding-3-small` (OpenAI) or `voyage-3-lite` (Voyage AI) for most use cases. Check MTEB for the current leaderboard. The critical rule: always use the same model for indexing and querying.

**Prerequisites**: [RAG Fundamentals](01-rag-fundamentals.md)
**Related**: [Vector Indexing](03-vector-indexing.md), [Chunking Strategies](05-chunking-strategies.md), [Hybrid Search](06-hybrid-search.md)

---

## The Intuition

An embedding model turns text into a point in high-dimensional space. Semantically similar texts land near each other. "What is the return policy?" and "How do I get a refund?" produce vectors that are close together, even though they share no words.

The distance between vectors (usually cosine similarity, ranging from -1 to 1) is the retrieval signal. Vector search finds the documents closest to your query in this space.

The model determines what "similar" means. A general-purpose model might not understand that "myocardial infarction" and "heart attack" are the same thing. A medical embedding model would. Domain-specific retrieval often requires domain-specific models.

---

## MTEB: How to Evaluate Models

[MTEB (Massive Text Embedding Benchmark)](https://huggingface.co/spaces/mteb/leaderboard) is the standard benchmark. It covers 56 tasks across retrieval, clustering, classification, and more. For RAG specifically, look at the **Retrieval** subtask scores.

As of early 2025, top models for retrieval:

| Model | MTEB Retrieval | Dimensions | Context | Cost | Notes |
|---|---|---|---|---|---|
| voyage-3 | ~70+ | 1024 | 32K tokens | $0.06/1M tokens | Best quality for RAG |
| text-embedding-3-large | ~64 | 3072 | 8K tokens | $0.13/1M tokens | OpenAI's best |
| text-embedding-3-small | ~62 | 1536 | 8K tokens | $0.02/1M tokens | Best value |
| bge-large-en-v1.5 | ~60 | 1024 | 512 tokens | Free (self-hosted) | Best open-source |
| e5-large-v2 | ~58 | 1024 | 512 tokens | Free (self-hosted) | Strong open-source |
| all-MiniLM-L6-v2 | ~50 | 384 | 256 tokens | Free (self-hosted) | Fast, good for prototyping |

*Scores are approximate and change as new models release. Always check MTEB for current rankings.*

**My default recommendation:** `text-embedding-3-small` for API-based RAG (great quality/cost ratio), `bge-large-en-v1.5` for self-hosted (best open-source retrieval quality).

---

## Using Embedding Models in Code

```python
# OpenAI (API)
from openai import OpenAI
client = OpenAI()

def embed_openai(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

# Voyage AI (API)
import voyageai
voyage = voyageai.Client()

def embed_voyage(texts: list[str]) -> list[list[float]]:
    result = voyage.embed(texts, model="voyage-3", input_type="document")
    return result.embeddings

# Sentence Transformers (local)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def embed_local(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, normalize_embeddings=True).tolist()
```

**Batch for efficiency.** Single-text embedding calls are expensive in API credits and latency. Always batch: `embed(["text1", "text2", "text3"])` is one API call instead of three.

---

## Query vs Document Asymmetry

Some models are trained with different objectives for queries vs documents. The query "What is X?" is short and general. The document answering it is long and specific. Symmetric models embed both the same way. Asymmetric models optimize for query-to-document matching.

**BGE models** have explicit query/passage modes:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# For queries: add the instruction prefix
query_embedding = model.encode(
    "Represent this sentence for searching relevant passages: What is machine learning?",
    normalize_embeddings=True
)

# For documents: no prefix needed
doc_embedding = model.encode(
    "Machine learning is a subset of artificial intelligence...",
    normalize_embeddings=True
)
```

Using the wrong mode (query prefix on documents, or no prefix on queries) meaningfully degrades retrieval quality. Read the model card for your chosen model.

**Voyage AI's `input_type` parameter:**

```python
# For indexing documents
doc_embeddings = voyage.embed(chunks, model="voyage-3", input_type="document")

# For querying
query_embedding = voyage.embed([query], model="voyage-3", input_type="query")
```

---

## Dimensions: More Is Not Always Better

Higher dimensions mean more expressive embeddings but more storage and slower search.

| Dimensions | Storage per 1M vectors | ANN search latency | Use when |
|---|---|---|---|
| 384 | ~1.5 GB | Very fast | Prototyping, latency-critical |
| 768 | ~3 GB | Fast | Good balance |
| 1024 | ~4 GB | Fast | Production default |
| 1536 | ~6 GB | Medium | High quality requirements |
| 3072 | ~12 GB | Slower | Maximum quality, cost-insensitive |

`text-embedding-3-small` and `text-embedding-3-large` support **Matryoshka representation learning**: you can truncate the embedding to a smaller dimension without retraining. This lets you use 256 dimensions for fast pre-filtering and 1536 for precise reranking.

```python
# text-embedding-3 supports dimension reduction
response = client.embeddings.create(
    input="your text here",
    model="text-embedding-3-large",
    dimensions=512  # truncate to 512 instead of full 3072
)
```

---

## Domain-Specific Embedding Models

General models work well for general text. For specialized domains, consider domain-specific models:

| Domain | Model | Why |
|---|---|---|
| Code | `voyage-code-3`, `text-embedding-3-large` | Code-specific pretraining |
| Medical | `BiomedBERT`, `MedCPT` | Medical vocabulary |
| Legal | `legal-bert-base-uncased` | Legal terminology |
| Multilingual | `multilingual-e5-large`, `voyage-multilingual-2` | Cross-language retrieval |
| Financial | `finance-embeddings-investopedia` | Financial text |

For most enterprise RAG, general models like `text-embedding-3-small` perform surprisingly well even on specialized domains. Test before assuming you need a domain-specific model.

---

## Cost at Scale

For a corpus of 1M chunks at 512 tokens each:

| Model | Indexing Cost | Query Cost (per 1K queries) |
|---|---|---|
| text-embedding-3-small | $10.24 | $0.02 |
| text-embedding-3-large | $66.56 | $0.13 |
| voyage-3 | $30.72 | $0.06 |
| bge-large (self-hosted) | $0 (GPU compute) | $0 |

Self-hosted models are free but require GPU infrastructure. At 10M+ vectors, the economics favor self-hosting. Under 10M vectors, API models are usually cheaper when you factor in GPU costs.

---

## Gotchas

**The same model, always.** Index with `text-embedding-3-small`, query with `text-embedding-3-small`. Mix models and you'll get garbage retrieval because the vector spaces are incompatible.

**Normalize embeddings for cosine similarity.** Most vector databases support cosine similarity natively. But if you're computing similarity manually with dot product, normalize embeddings to unit length first. `normalize_embeddings=True` in sentence-transformers handles this.

**Rate limits on embedding APIs.** OpenAI's embedding endpoint has rate limits (tokens per minute). For large batch jobs, implement exponential backoff and respect the limits. Embedding 1M chunks in a hurry will hit rate limits.

**Don't embed short chunks.** Chunks under 20 tokens embed poorly. A chunk that's just "Q4 2024" or a page number produces an embedding that's not meaningful and pollutes your index.

---

> **Key Takeaways:**
> 1. Use MTEB retrieval scores to compare models. `text-embedding-3-small` is the default choice for most API-based RAG: strong quality, low cost.
> 2. Some models have separate modes for queries vs documents. Using the wrong mode silently degrades retrieval quality.
> 3. Dimensions are a latency/quality tradeoff. Matryoshka embeddings let you use a smaller dimension without retraining.
>
> *"The embedding model determines what 'similar' means in your retrieval system. Choose it more carefully than you choose the vector database."*

---

## Interview Questions

**Q: Your RAG system serves both English and French users. How do you handle multilingual embedding?**

I'd use a multilingual embedding model: `multilingual-e5-large` (open-source) or `voyage-multilingual-2` (API). These are trained on multilingual corpora and can match a French query against an English document if they discuss the same topic.

The alternative is maintaining separate indexes per language, which doubles infrastructure and requires query language detection. That's only worth it if retrieval quality is significantly better with per-language models, which you'd verify by testing.

The tricky edge case: technical jargon and proper nouns often don't translate. "Kubernetes" in English and "Kubernetes" in French embed similarly (they're the same word). But "data center" vs "centre de données" might not embed as closely as you'd want. For critical technical queries, I'd combine multilingual semantic search with BM25 keyword search as a safety net.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| What is MTEB? | Massive Text Embedding Benchmark: the standard leaderboard for comparing embedding models |
| What does "query vs document asymmetry" mean? | Some models use different encoding for short queries vs long documents; use them correctly or retrieval degrades |
| What is Matryoshka representation learning? | Training embeddings that can be truncated to smaller dimensions without retraining |
| What is the critical rule for embedding models? | Use the same model for both indexing (documents) and querying |
| What dimension is a good production default? | 1024 dimensions: good quality/storage balance for most use cases |
| How do you batch embed efficiently? | Pass all texts as a list in one API call; never call one text at a time |
