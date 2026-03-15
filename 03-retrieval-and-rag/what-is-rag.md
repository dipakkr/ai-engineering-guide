# What is RAG (Retrieval-Augmented Generation)?

**Retrieval-Augmented Generation (RAG)** is a technique that gives a language model access to external knowledge at inference time — so it can answer questions using up-to-date or private information it was never trained on.

The core idea: instead of relying on what the model memorized during training, **retrieve the relevant information first, then generate an answer grounded in that retrieved content**.

---

## The Problem RAG Solves

LLMs have two fundamental limitations when used alone:

**1. Knowledge cutoff**
Models are trained on data up to a certain date. GPT-4's training data ends in early 2024. Ask it about something that happened last week and it won't know.

**2. No access to private/proprietary data**
Your company's internal documents, customer records, product manuals, and Slack messages were never in the training data. The model can't answer questions about them.

RAG fixes both problems — without retraining the model.

---

## How RAG Works: The Simple Version

```
User asks: "What is our refund policy for enterprise customers?"
                          ↓
          Search your knowledge base for relevant docs
                          ↓
          Find: "Enterprise Refund Policy v2.pdf" — top 3 sections
                          ↓
          Send to LLM: [Retrieved sections] + "Answer this: " + [User question]
                          ↓
          LLM responds using the retrieved content as its source
```

The model doesn't need to know the answer from memory. The answer is *handed to it* in the prompt, grounded in your actual documents.

---

## How RAG Works: The Technical Version

RAG has two main phases:

### Phase 1: Indexing (done once, updated as data changes)

```
Your documents (PDFs, databases, web pages, etc.)
         ↓
   1. Chunk into smaller pieces (e.g. 512 tokens each)
         ↓
   2. Embed each chunk → convert to a vector (list of numbers)
         ↓
   3. Store vectors in a vector database (Pinecone, Weaviate, pgvector...)
```

Each chunk of text becomes a point in high-dimensional space where **similar meaning = nearby points**.

### Phase 2: Retrieval + Generation (at query time)

```
User query
    ↓
1. Embed the query (same embedding model as indexing)
    ↓
2. Search vector database → find top-K most similar chunks
    ↓
3. Build prompt:
   [System prompt]
   [Retrieved chunks as context]
   [User query]
    ↓
4. LLM generates answer grounded in retrieved context
    ↓
Response to user (optionally with source citations)
```

---

## A Concrete Example

**Setup:** A company builds a customer support bot backed by 5,000 product documentation pages.

**Without RAG:**
> User: "Does the X200 model support USB-C charging?"
> Bot: "I'm not sure — please check the manual." (or worse: confidently wrong)

**With RAG:**
1. Query is embedded → search docs → retrieves: *"X200 Technical Specs, Section 3: The X200 supports USB-C (PD 3.0) charging at up to 65W..."*
2. LLM reads this retrieved section and answers:
> Bot: "Yes, the X200 supports USB-C charging with Power Delivery 3.0 at up to 65W."

The model answered correctly from a document it had never seen during training.

---

## RAG vs. Fine-Tuning: Which Do You Need?

This is one of the most common questions in AI engineering. They solve different problems.

| | RAG | Fine-Tuning |
|---|---|---|
| **Problem solved** | Access to new/private knowledge | Change model behavior or style |
| **Updates** | Real-time (update the index) | Requires retraining |
| **Cost** | Inference + embedding costs | Training compute + time |
| **Transparency** | Can cite sources | Black box |
| **When to use** | Q&A over documents, knowledge bases | Tone, format, domain-specific reasoning |

**General rule:** If the problem is "the model doesn't know X," use RAG. If the problem is "the model doesn't behave like Y," use fine-tuning.

Most production systems use **both**: fine-tune for behavior, RAG for knowledge.

---

## Why Not Just Put Everything in the Context Window?

If models support 128k or 1M token context windows, why not just paste all your documents in?

**It doesn't scale:**
- Your knowledge base might be 100GB of documents — far beyond any context window
- Larger contexts = higher latency + higher cost (every API call costs more)
- Models perform worse with very long contexts (lost-in-the-middle problem)

**Retrieval is smarter:**
- Only the *relevant* 3–5 chunks are fetched — keeping the prompt focused
- A 500-document knowledge base returns the same 3 relevant chunks as a 5,000,000-document one

---

## Key Components of a RAG System

```
┌──────────────────────────────────────────────┐
│              RAG PIPELINE                    │
│                                              │
│  Documents → Chunker → Embedder → VectorDB  │
│                                    ↑         │
│  Query → Embedder → VectorDB Search          │
│                ↓                             │
│          Top-K Chunks                        │
│                ↓                             │
│     Prompt Builder → LLM → Response         │
└──────────────────────────────────────────────┘
```

| Component | Role | Examples |
|---|---|---|
| **Chunker** | Splits documents into retrievable pieces | LangChain splitters, LlamaIndex |
| **Embedding model** | Converts text to vectors | text-embedding-3-small, BGE, E5 |
| **Vector database** | Stores and searches vectors | Pinecone, Weaviate, pgvector, Chroma |
| **Retriever** | Finds relevant chunks for a query | Semantic search, hybrid search |
| **LLM** | Generates answer from context | GPT-4o, Claude, Gemini, Llama |

---

## RAG Failure Modes to Know

RAG seems simple but fails in subtle ways:

**Retrieval fails to find relevant content**
The right document exists but the query doesn't match well. Solution: query rewriting, hybrid search, better chunking.

**Retrieved content doesn't answer the question**
Chunks are relevant topically but miss the specific detail. Solution: smaller chunks, better overlap, re-ranking.

**LLM ignores retrieved content**
The model answers from memory instead of the provided context. Solution: stronger system prompt, chain-of-thought.

**Contradictory retrieved content**
Two retrieved chunks say opposite things. Solution: re-ranking, source filtering, recency weighting.

**Hallucination despite RAG**
The model extrapolates beyond what was retrieved. Solution: constrain with "only answer from the provided context."

---

## Simple RAG in Code

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

# 1. Embed a query
def embed(text):
    res = client.embeddings.create(input=text, model="text-embedding-3-small")
    return np.array(res.data[0].embedding)

# 2. Find top-K similar chunks (simplified — in prod use a vector DB)
def retrieve(query, chunks, chunk_embeddings, k=3):
    q_emb = embed(query)
    scores = [np.dot(q_emb, c) for c in chunk_embeddings]
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_k]

# 3. Generate answer grounded in retrieved context
def rag(query, chunks, chunk_embeddings):
    context = retrieve(query, chunks, chunk_embeddings)
    context_str = "\n\n".join(context)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using only the provided context. "
                                          "If the answer isn't in the context, say so."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content
```

This is the skeleton of every RAG system — production systems add re-ranking, hybrid search, caching, and evaluation on top.

---

## RAG in Practice: Where It's Used

- **Customer support bots** — answer questions from product docs and knowledge bases
- **Internal search** — find answers across company wikis, Slack, Notion, Confluence
- **Legal and compliance** — search contracts, regulations, case law
- **Medical** — retrieve clinical guidelines, drug interaction databases
- **Code assistants** — fetch relevant code snippets, API docs, internal patterns
- **Financial research** — retrieve earnings reports, filings, market data

Any use case where the question is "what does our data say about X?" is a RAG candidate.

---

## What's Next

- **RAG Fundamentals** — deeper dive into each component and design decisions
- **Embedding Models** — how semantic similarity search actually works
- **Vector Indexing** — ANN algorithms (HNSW, IVF) and tradeoffs
- **Vector Databases** — choosing between Pinecone, Weaviate, pgvector, and others
- **Chunking Strategies** — how to split documents for best retrieval quality
- **Hybrid Search** — combining semantic and keyword search for better recall
