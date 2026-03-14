# Vector Databases

> **TL;DR**: For most teams building RAG: use Pinecone (managed, simple) or pgvector (if you already run Postgres). Weaviate and Qdrant are excellent self-hosted options with more features. Chroma is great for development. The "best" vector DB is the one your team can operate reliably. Don't pick based on benchmarks alone; pick based on your operational maturity.

**Prerequisites**: [Vector Indexing](03-vector-indexing.md), [Embedding Models](02-embedding-models.md)
**Related**: [RAG Fundamentals](01-rag-fundamentals.md), [Chunking Strategies](05-chunking-strategies.md)

---

## The Decision Matrix

```mermaid
flowchart TD
    A[I need a vector database] --> B{Existing Postgres infrastructure?}
    B --> |Yes| C[pgvector\nstay in Postgres ecosystem]
    B --> |No| D{Self-hosted or managed?}
    D --> |Managed, minimize ops| E{Scale?}
    D --> |Self-hosted, control matters| F{Features needed?}
    E --> |< 10M vectors| G[Pinecone Starter\nSimplest setup]
    E --> |10M-1B vectors| H[Pinecone Standard\nor Weaviate Cloud]
    F --> |Basic vector search| I[Qdrant\nFast, good Rust performance]
    F --> |Multi-tenancy, GraphQL| J[Weaviate\nRich feature set]
    F --> |Development / prototyping| K[Chroma\nSimplest local setup]
```

---

## Side-by-Side Comparison

| Database | Type | Best For | Cost (1M vectors) | Filtering | Multi-tenancy |
|---|---|---|---|---|---|
| Pinecone | Managed only | Simplicity, fast setup | ~$70/mo | Yes | Namespaces |
| Weaviate | Self-hosted + cloud | Rich features, GraphQL | ~$50/mo cloud | Yes (BM25 built-in) | Multi-tenant classes |
| Qdrant | Self-hosted + cloud | Performance, Rust speed | ~$30/mo cloud | Yes | Collections |
| Chroma | Self-hosted | Development, prototyping | Free | Basic | Collections |
| pgvector | Extension | Postgres teams, SQL joins | Postgres infra | Full SQL | Row-level security |
| Milvus | Self-hosted | Large scale, enterprise | Self-hosted costs | Yes | Partitions |
| Redis VSS | Extension | Low-latency, existing Redis | Redis infra | Yes | Prefixes |

*Costs as of early 2025, approximate. Scale pricing varies significantly.*

---

## Pinecone: The Managed Default

Pinecone is the easiest to get started with. No infrastructure to manage, consistent performance, and a generous free tier.

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-api-key")

# Create index
pc.create_index(
    name="my-rag-index",
    dimension=1536,          # match your embedding model
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("my-rag-index")

# Upsert vectors with metadata
vectors = [
    ("chunk-001", embedding_1, {"document_id": "doc-1", "section": "intro", "date": "2024-01-15"}),
    ("chunk-002", embedding_2, {"document_id": "doc-1", "section": "methods", "date": "2024-01-15"}),
]
index.upsert(vectors=[(id, vec, meta) for id, vec, meta in vectors])

# Query with metadata filtering
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={"document_id": {"$eq": "doc-1"}},  # only retrieve from specific doc
    include_metadata=True
)
```

**Where Pinecone shines:** Teams that want to focus on application logic rather than infrastructure. Startup teams that don't have dedicated infrastructure engineers.

**Where it hurts:** Cost at scale. At 100M vectors, Pinecone's managed pricing becomes expensive. Also: no built-in BM25, limited control over index parameters.

---

## Weaviate: Rich Features

Weaviate has the richest feature set of any vector database. Built-in BM25 hybrid search, multi-tenancy, GraphQL API, and module system for auto-vectorization.

```python
import weaviate

client = weaviate.connect_to_local()  # or weaviate.connect_to_weaviate_cloud()

# Define schema
client.collections.create(
    name="Document",
    properties=[
        weaviate.classes.config.Property(name="content", data_type=weaviate.classes.config.DataType.TEXT),
        weaviate.classes.config.Property(name="document_id", data_type=weaviate.classes.config.DataType.TEXT),
        weaviate.classes.config.Property(name="section", data_type=weaviate.classes.config.DataType.TEXT),
    ],
    vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),  # bring your own vectors
)

collection = client.collections.get("Document")

# Insert with custom vector
collection.data.insert(
    properties={"content": chunk_text, "document_id": "doc-1", "section": "intro"},
    vector=embedding
)

# Hybrid search (BM25 + vector, built-in)
results = collection.query.hybrid(
    query="machine learning techniques",
    alpha=0.5,  # 0=pure BM25, 1=pure vector
    limit=5
)
```

**Where Weaviate shines:** Multi-tenant applications (each tenant gets isolated data), hybrid search without adding a separate BM25 store, complex queries with filters and sorting.

**Where it hurts:** Self-hosted complexity. Running Weaviate in production requires managing Docker/Kubernetes, backups, and upgrades. The GraphQL API has a learning curve.

---

## Qdrant: Performance and Simplicity

Qdrant is written in Rust, which gives it excellent performance characteristics. The Python client is clean and the REST API is intuitive.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Insert points
points = [
    PointStruct(
        id=i,
        vector=embedding,
        payload={"content": chunk, "document_id": doc_id, "section": section}
    )
    for i, (embedding, chunk, doc_id, section) in enumerate(data)
]
client.upsert(collection_name="documents", points=points)

# Search with filter
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value="doc-1"))]
    ),
    limit=5
)
```

**Where Qdrant shines:** Raw search performance (benchmarks consistently near the top), good documentation, and a clean API. Payload filtering is fast even on large corpora.

**Where it hurts:** Fewer built-in integrations than Weaviate. No native BM25 hybrid search (requires external BM25 index).

---

## pgvector: For Postgres Teams

If your application already runs on Postgres, pgvector adds vector search without introducing a new database to manage.

```sql
-- Enable extension
CREATE EXTENSION vector;

-- Add vector column to existing table
ALTER TABLE documents ADD COLUMN embedding vector(1536);

-- Create HNSW index
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 32, ef_construction = 200);

-- Search (cosine similarity)
SELECT content, 1 - (embedding <=> $1) AS similarity
FROM documents
WHERE document_id = 'doc-1'
ORDER BY embedding <=> $1
LIMIT 5;
```

The `<=>` operator is cosine distance. For dot product: `<#>`. For L2: `<->`.

```python
import psycopg2
import numpy as np

conn = psycopg2.connect("postgresql://user:pass@host/db")
cur = conn.cursor()

# Insert
cur.execute(
    "INSERT INTO chunks (content, embedding, doc_id) VALUES (%s, %s, %s)",
    (chunk_text, embedding.tolist(), doc_id)
)

# Search
cur.execute(
    "SELECT content, 1 - (embedding <=> %s) AS sim FROM chunks WHERE doc_id = %s ORDER BY embedding <=> %s LIMIT %s",
    (query_embedding.tolist(), doc_id, query_embedding.tolist(), 5)
)
results = cur.fetchall()
```

**Where pgvector shines:** SQL joins with your application data, existing Postgres operational know-how, exact transactional semantics with your other data, row-level security for access control.

**Where it hurts:** Performance at very large scale (>10M vectors) is slower than dedicated vector databases. HNSW in pgvector is less tunable than FAISS or Qdrant.

---

## Cost at Scale

Rough estimates as of early 2025. Actual costs depend heavily on your query volume and data size.

| Scale | Pinecone | Weaviate Cloud | Qdrant Cloud | pgvector (RDS) |
|---|---|---|---|---|
| 100K vectors | $0 (free tier) | $25/mo | $25/mo | $30/mo (RDS) |
| 1M vectors | $70/mo | $50/mo | $35/mo | $60/mo |
| 10M vectors | $700/mo | $300/mo | $150/mo | $200/mo |
| 100M vectors | $5,000+/mo | $1,500+/mo | $800+/mo | Self-host recommended |
| 1B vectors | Custom | Self-host recommended | Self-host recommended | Not recommended |

*Self-hosting costs depend on hardware. A 10M vector HNSW index needs ~8GB RAM. Plan accordingly.*

---

## The pgvector Decision

Many teams default to pgvector because "we already have Postgres." This is reasonable up to a point.

**Use pgvector if:**
- You have an existing Postgres deployment
- You need SQL joins between vectors and relational data
- You have under 5M vectors
- Your team knows Postgres well

**Graduate to a dedicated vector DB if:**
- Query latency on pgvector exceeds your SLA
- You need built-in BM25 hybrid search
- You're scaling past 10M vectors
- You need multi-tenancy with strict isolation

---

## Metadata Filtering: The Underused Feature

Every serious vector database supports metadata filtering. This is how you scope retrieval to a subset of your corpus without running a separate query.

```python
# Pinecone: only retrieve from documents owned by user "alice"
results = index.query(
    vector=query_embedding,
    filter={"user_id": {"$eq": "alice"}, "created_after": {"$gt": "2024-01-01"}},
    top_k=5
)

# pgvector: use WHERE clause
SELECT content FROM chunks
WHERE user_id = 'alice'
  AND created_at > '2024-01-01'
ORDER BY embedding <=> $1
LIMIT 5;
```

Use metadata filtering for: access control, tenant isolation, date filtering for freshness, document type filtering. It's faster and more reliable than post-filtering retrieved results.

---

## Gotchas

**Pinecone cold starts.** Serverless Pinecone indexes "sleep" after inactivity and have a cold start latency of 1-3 seconds on the first query. For production, use the pod-based offering or keep the index warm with periodic queries.

**pgvector HNSW requires a full table scan to build.** Creating an HNSW index on an existing large table takes significant time and I/O. Do this during off-peak hours and expect query slowdown during the build.

**Weaviate schema migrations are painful.** Changing a collection's schema requires re-creating it and re-importing all data. Design your schema carefully before indexing large datasets.

**Metadata filter performance varies.** All vector databases support metadata filtering, but performance varies significantly. Qdrant and Weaviate have highly optimized filtered search. Some databases do post-filter (retrieve top-K, then apply filter) which is wasteful. Check your database's filter implementation.

**Embedding dimension mismatches silently break retrieval.** Create an index with dimension 1536, insert a 768-dimensional vector: either error or silent corruption depending on the database. Always validate embedding dimensions at insert time.

---

> **Key Takeaways:**
> 1. Start with Pinecone (managed simplicity) or pgvector (existing Postgres). Evaluate Weaviate or Qdrant when you need hybrid search, multi-tenancy, or cost optimization at scale.
> 2. Metadata filtering is as important as vector search. Design your metadata schema for the filter patterns you'll need before indexing.
> 3. The operational cost of self-hosting is real. A managed database at $300/mo may be cheaper than the engineering time to operate a self-hosted one.
>
> *"The best vector database is the one your team can operate at 2am when it breaks. Optimize for operational simplicity before benchmark scores."*

---

## Interview Questions

**Q: Design the vector database architecture for a multi-tenant SaaS application where each customer's data must be strictly isolated. What database do you choose and why?**

Tenant isolation is the critical constraint. The options are: one collection/namespace per tenant, or row-level filtering within a shared collection.

For strong isolation requirements (compliance, contractual data isolation), I'd use either Weaviate's multi-tenant classes or separate Qdrant collections per tenant. Both give complete physical separation. The downside: more objects to manage, and you can't run cross-tenant queries easily.

For softer isolation (trust the application layer), a single collection with `tenant_id` metadata filtering is simpler operationally. Every query adds `filter={"tenant_id": "tenant-xyz"}`. Access control is enforced at the application layer. Faster to set up, but one application bug could expose cross-tenant data.

I'd lean toward physical isolation (Weaviate multi-tenant) for B2B SaaS with enterprise customers who have compliance requirements. For a B2C product with lower data sensitivity, filtering is acceptable with proper application-level validation.

For scale: if you have 10,000 tenants each with 1,000 vectors, that's 10M vectors total. With Weaviate multi-tenant, each tenant's data is sharded separately but managed under one deployment. This scales better than 10,000 separate Qdrant collections.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| What is pgvector? | A Postgres extension that adds vector storage and similarity search operators to Postgres |
| When should you choose pgvector over Pinecone? | When you already run Postgres and need SQL joins between vectors and relational data |
| What is the Pinecone cold start problem? | Serverless indexes pause after inactivity and have 1-3s latency on the first query |
| What does metadata filtering do? | Scopes vector search to a subset of vectors that match attribute conditions, before or during ANN search |
| Which database has built-in BM25 hybrid search? | Weaviate (native hybrid search) |
| At what vector count should you consider moving from pgvector? | Around 5-10M vectors, when query latency or operational complexity becomes a constraint |
