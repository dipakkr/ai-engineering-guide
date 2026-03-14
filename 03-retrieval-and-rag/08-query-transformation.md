# Query Transformation

> **TL;DR**: Query transformation rewrites the user's query before retrieval to improve what gets found. HyDE (generate a hypothetical answer, embed that) fixes vocabulary gaps. Multi-query (rephrase the question 3 ways) improves recall. Query decomposition (break compound questions into sub-questions) handles complex multi-part queries. Add these only when you've measured retrieval is failing due to the specific mismatch each pattern fixes.

**Prerequisites**: [RAG Fundamentals](01-rag-fundamentals.md), [Embedding Models](02-embedding-models.md), [Chunking Strategies](05-chunking-strategies.md)
**Related**: [Advanced RAG Patterns](09-advanced-rag-patterns.md), [Hybrid Search](06-hybrid-search.md), [Reranking](07-reranking.md)

---

## When Queries Fail: A Taxonomy

Before adding query transformation, understand which failure mode you're fixing.

| Failure Mode | Symptom | Fix |
|---|---|---|
| Vocabulary mismatch | User asks about "revenue" but docs say "sales" | HyDE, query expansion |
| Short abstract query | "Explain X" doesn't match long document passages | HyDE |
| Multi-part question | Answer spans multiple documents | Query decomposition |
| Ambiguous intent | Query could mean several things | Clarification or multi-query |
| Keyword query on semantic content | "REST API endpoint examples" on prose docs | Query expansion + hybrid search |
| Complex comparative question | "Compare A and B on dimension X" | Decompose into separate retrievals |

Identify the failure mode from production logs before choosing a transformation strategy.

---

## HyDE: Hypothetical Document Embeddings

**The insight:** Short user queries embed differently than long document passages. "What causes inflation?" produces a different vector than a 500-word explanation of inflation causes, even though they're about the same thing.

HyDE generates a hypothetical answer to the query, then embeds that answer instead of the raw query. The hypothetical answer is longer and more specific, so it embeds more similarly to the actual relevant documents.

```python
from anthropic import Anthropic

client = Anthropic()

def hyde_embed(query: str, embed_model) -> list[float]:
    """Generate a hypothetical answer and embed it."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"Write a concise, factual paragraph that would answer this question. Write only the answer, no preamble.\n\nQuestion: {query}"
        }]
    )
    hypothetical_answer = response.content[0].text
    return embed_model.encode([hypothetical_answer])[0].tolist()

# Instead of embedding the query directly:
# query_embedding = embed_model.encode([query])

# Use HyDE:
# query_embedding = hyde_embed(query, embed_model)
```

**When HyDE helps:**
- Short, abstract queries ("explain caching strategies")
- Technical questions where user vocabulary differs from document vocabulary
- Academic or research retrieval where documents are detailed but queries are brief

**When HyDE doesn't help:**
- Simple factual lookups ("what is the price?")
- Queries with specific identifiers (error codes, product SKUs) that need exact matching
- When the hypothetical answer might be wrong, leading you to retrieve documents that match the wrong answer

**The risk:** If the hypothetical answer is factually wrong (the model hallucinating), you embed incorrect information and retrieve documents about the wrong thing. HyDE works best when the LLM can plausibly generate a correct hypothetical answer.

---

## Multi-Query Expansion

Generate multiple rephrasings of the query and retrieve for each. Merge results using RRF.

```python
def multi_query_retrieve(query: str, collection, embed_model, n_queries: int = 3) -> list[str]:
    # Generate rephrasings
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content":
            f"Generate {n_queries} different phrasings of this question that would help find relevant information in a document database. Return only the questions, one per line.\n\nQuestion: {query}"}]
    )

    all_queries = [query] + [q.strip() for q in response.content[0].text.strip().split("\n") if q.strip()][:n_queries]

    # Retrieve for each query
    result_sets = []
    seen_ids = set()

    for q in all_queries:
        q_emb = embed_model.encode([q])[0].tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=5)
        ranked_ids = []
        for doc_id, doc in zip(results["ids"][0], results["documents"][0]):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
            ranked_ids.append(doc_id)
        result_sets.append(ranked_ids)

    # RRF fusion
    fused = reciprocal_rank_fusion(result_sets)
    # Fetch documents for top fused IDs
    return fetch_documents(fused[:10], collection)
```

**When multi-query helps:**
- Queries that could be phrased many ways
- When you want to improve recall for ambiguous intent
- Documents with varied terminology for the same concepts

**The cost:** N LLM calls to generate rephrasings + N retrieval calls. You can parallelize the retrieval calls to reduce latency. The LLM generation step is sequential.

**Empirical result:** Multi-query typically improves recall@10 by 5-10% on diverse query sets. The LLM generation adds 500ms-1s. Worthwhile when recall is the bottleneck.

---

## Query Decomposition

Break a compound question into independent sub-questions, retrieve for each, then synthesize.

```python
def decompose_and_retrieve(query: str, collection, embed_model) -> dict:
    # Decompose
    decomp_response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=400,
        tools=[{
            "name": "decompose_query",
            "description": "Break a complex query into independent sub-questions",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sub_questions": {"type": "array", "items": {"type": "string"}, "description": "List of independent sub-questions"}
                },
                "required": ["sub_questions"]
            }
        }],
        tool_choice={"type": "tool", "name": "decompose_query"},
        messages=[{"role": "user", "content": f"Decompose this into independent sub-questions if it's complex. If it's simple, return just the original question.\n\nQuery: {query}"}]
    )

    sub_questions = parse_tool_result(decomp_response, "decompose_query")["sub_questions"]

    # Retrieve for each sub-question
    results_by_question = {}
    for sq in sub_questions:
        q_emb = embed_model.encode([sq])[0].tolist()
        results = collection.query(query_embeddings=[q_emb], n_results=3)
        results_by_question[sq] = results["documents"][0]

    return results_by_question
```

**When decomposition helps:**
- "Compare the performance and cost of GPT-4o vs Claude Sonnet"
- "What are the advantages of microservices and how do they relate to containerization?"
- Any question with "and" that represents two distinct information needs

**When decomposition hurts:** Simple questions that get unnecessarily split into trivial sub-questions, adding latency for no gain. Build a complexity classifier: only decompose questions with multiple clauses.

---

## Step-Back Prompting

Before retrieving, ask the LLM to formulate a more general "step-back" question that captures the underlying principle.

```python
def step_back_retrieve(query: str, collection, embed_model) -> list[str]:
    # Get the step-back question
    step_back = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content":
            f"Given this specific question, what is the more general question or concept I should understand first?\n\nQuestion: {query}\n\nGeneral question:"}]
    ).content[0].text.strip()

    # Retrieve for both original and step-back
    original_results = retrieve(query, collection, embed_model)
    general_results = retrieve(step_back, collection, embed_model)

    # Combine: original for specific context, general for background
    return list(dict.fromkeys(original_results + general_results))[:10]  # deduplicate, keep order
```

**Example:**
- Query: "Why does Redis use single-threaded event loop?"
- Step-back: "How does event-driven I/O work in network services?"

Retrieving for both gives the LLM both the specific Redis answer and the conceptual background to explain it well.

---

## Query Routing vs Transformation

Sometimes the right answer isn't to transform the query but to route it to a different retrieval strategy:

```python
def smart_retrieve(query: str) -> list[str]:
    # Classify query type
    classification = classify_query(query)

    if classification == "factual_lookup":
        return retrieve_dense(query, top_k=5)  # simple case
    elif classification == "keyword_search":
        return retrieve_bm25(query, top_k=10)  # exact terms matter
    elif classification == "complex_analysis":
        return decompose_and_retrieve(query)  # multi-hop
    elif classification == "abstract_concept":
        return hyde_retrieve(query)  # vocabulary gap
    else:
        return retrieve_hybrid(query, top_k=10)  # safe default
```

This is the routing pattern from agentic patterns applied to retrieval. A fast classifier (Haiku) adds minimal latency but can significantly improve quality by applying the right strategy per query type.

---

## Concrete Numbers

| Transformation | Latency Added | Recall Improvement | Cost Added | When to Use |
|---|---|---|---|---|
| HyDE | +500ms-1s | +5-15% | +1 LLM call | Vocabulary gaps, abstract queries |
| Multi-query (3x) | +500ms-1s | +5-10% | +1 LLM call + 3x retrieval | Low recall on diverse queries |
| Decomposition | +500ms-1s | High for complex Qs | +1-2 LLM calls | Compound questions |
| Step-back | +300ms-700ms | +3-8% | +1 LLM call | Conceptual questions |
| Query routing | +50-150ms | Variable | +1 cheap LLM call | Mixed query types |

---

## Gotchas

**HyDE hallucinates the right answer sometimes.** If your LLM generates a confident hypothetical answer to a question about your proprietary internal data, the hypothetical might be wrong. You'd then retrieve documents that confirm the wrong hypothetical. HyDE works best when the LLM can plausibly generate a relevant hypothetical without needing domain-specific knowledge.

**Multi-query increases cost and latency significantly.** Three rephrasings plus three retrievals is 4-6x the cost of a single retrieve. At high query volume, this compounds. Cache the rephrasings if you see the same query often.

**Decomposition over-splits simple questions.** A well-intentioned query decomposer might split "What's the return policy?" into three sub-questions. Add a check: if the LLM produces only one sub-question, skip the decomposition overhead.

**Query transformations interact.** Running HyDE + multi-query + decomposition on the same query is overkill and expensive. Apply one transformation based on the failure mode you're solving, not all of them.

---

> **Key Takeaways:**
> 1. Query transformation fixes specific retrieval failures. HyDE for vocabulary gaps, multi-query for recall improvement, decomposition for compound questions.
> 2. Each transformation adds 300ms-1s latency and an LLM call. Add them only after measuring the specific failure mode they fix.
> 3. Query routing (classify query type, apply different strategy) is often more efficient than running all transformations on every query.
>
> *"Don't transform every query. Transform the ones that fail."*

---

## Interview Questions

**Q: Users of your RAG system ask complex compound questions that require information from multiple documents. How do you improve retrieval quality?**

This is the decomposition use case. "Compare the authentication approaches of service A and service B, and explain which is more secure in a zero-trust environment" requires: information about service A's auth, information about service B's auth, and principles of zero-trust security. A single vector search on the full compound question might retrieve documents about one service but miss the other.

My approach: add a query complexity classifier (cheap model, one call) that detects compound questions by looking for "and", "compare", "versus", and multi-part structure. For complex queries, decompose into 2-4 independent sub-questions, retrieve for each, merge results with RRF, then send all retrieved context to the LLM along with the original compound question.

The synthesis step is important: the LLM should see all the retrieved context organized by sub-question, not just a flat list of chunks. This helps it structure the answer to actually address each part of the compound question.

For simple questions, bypass decomposition entirely. The classifier overhead is ~50ms with Haiku, worth it to avoid unnecessary decomposition of simple queries.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| What is HyDE? | Hypothetical Document Embeddings: generate a hypothetical answer to the query, embed that instead of the raw query |
| When does HyDE work best? | Abstract queries with vocabulary mismatch between question and document terminology |
| What is query decomposition? | Breaking a compound question into independent sub-questions, retrieving for each separately |
| What is step-back prompting? | Formulating a more general version of the query to retrieve background context before the specific answer |
| What is the main cost of multi-query expansion? | One LLM call to generate rephrasings plus N times the retrieval cost |
| When should you skip query transformation? | For simple factual lookups, keyword searches, or queries already performing well in your eval set |
