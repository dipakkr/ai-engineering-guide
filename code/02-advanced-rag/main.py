"""
Advanced RAG Pipeline
=====================
Demonstrates:
- Parent-child chunking (index small, return large)
- HyDE (Hypothetical Document Embeddings) query expansion
- Cross-encoder reranking
- Qdrant for vector storage

Run: python main.py
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import anthropic

load_dotenv()

# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    id: str
    text: str           # Small chunk for indexing
    parent_text: str    # Larger parent for returning to LLM
    metadata: dict


# Sample document split into parent paragraphs, each with child sentences
SAMPLE_DOC = """
RAG systems retrieve relevant context before generating answers. The retrieval
component uses embedding similarity to find semantically related passages.
Dense retrieval with bi-encoders is fast but may miss exact keyword matches.

Hybrid search combines dense vector search with sparse BM25 retrieval.
The results from both systems are merged using Reciprocal Rank Fusion.
This approach handles both semantic and keyword queries effectively.

Reranking improves precision by applying a cross-encoder to the top-k results.
Cross-encoders take a query-document pair as input and score their relevance.
They are slower than bi-encoders but significantly more accurate.

Chunking strategy affects both retrieval and generation quality.
Larger chunks provide more context but may include irrelevant text.
Parent-child chunking indexes small chunks and returns their larger parents.
"""


def build_chunks(document: str) -> list[Chunk]:
    """Split document into parent paragraphs, child sentences."""
    chunks = []
    paragraphs = [p.strip() for p in document.strip().split("\n\n") if p.strip()]

    for para_idx, paragraph in enumerate(paragraphs):
        sentences = [s.strip() for s in paragraph.split(".") if s.strip()]
        for sent_idx, sentence in enumerate(sentences):
            if len(sentence) > 20:  # Skip very short fragments
                chunks.append(Chunk(
                    id=f"p{para_idx}_s{sent_idx}",
                    text=sentence,           # Child: indexed for retrieval
                    parent_text=paragraph,   # Parent: returned to LLM
                    metadata={"para_idx": para_idx}
                ))
    return chunks


class AdvancedRAG:
    def __init__(self):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.client = anthropic.Anthropic()

        # In-memory Qdrant (no server required)
        self.qdrant = QdrantClient(":memory:")
        self.collection = "rag_demo"
        self.qdrant.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        self._index_documents(build_chunks(SAMPLE_DOC))

    def _index_documents(self, chunks: list[Chunk]):
        texts = [c.text for c in chunks]
        embeddings = self.embed_model.encode(texts).tolist()

        self.qdrant.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=i,
                    vector=embedding,
                    payload={"text": c.text, "parent_text": c.parent_text}
                )
                for i, (c, embedding) in enumerate(zip(chunks, embeddings))
            ]
        )
        # Store chunks for reranking access
        self.chunks = chunks
        print(f"Indexed {len(chunks)} child chunks from {len(set(c.metadata['para_idx'] for c in chunks))} paragraphs")

    def hyde_retrieve(self, query: str, k: int = 8) -> list[str]:
        """HyDE: generate a hypothetical answer, use its embedding for retrieval."""
        hyde_response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": f"Write a short factual passage that directly answers: {query}"
            }]
        )
        hypothetical_doc = hyde_response.content[0].text

        # Use hypothetical doc embedding for retrieval
        query_embedding = self.embed_model.encode([hypothetical_doc]).tolist()[0]
        results = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_embedding,
            limit=k
        )
        return [r.payload["parent_text"] for r in results]

    def rerank(self, query: str, candidates: list[str], top_n: int = 3) -> list[str]:
        """Cross-encoder reranking: score each candidate against the query."""
        pairs = [(query, doc) for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_n]]

    def generate(self, query: str, context: list[str]) -> str:
        context_str = "\n\n---\n\n".join(context)
        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system="Answer using only the provided context. Be concise and specific.",
            messages=[{"role": "user", "content": f"Context:\n{context_str}\n\nQ: {query}"}]
        )
        return response.content[0].text

    def query(self, question: str) -> dict:
        # Step 1: HyDE retrieval (broader recall)
        candidates = self.hyde_retrieve(question, k=8)

        # Step 2: Deduplicate (parent chunks may repeat across child hits)
        unique_candidates = list(dict.fromkeys(candidates))

        # Step 3: Rerank for precision
        top_context = self.rerank(question, unique_candidates, top_n=3)

        # Step 4: Generate
        answer = self.generate(question, top_context)
        return {"question": question, "answer": answer, "context_used": top_context}


def main():
    rag = AdvancedRAG()

    questions = [
        "What is HyDE and how does it improve retrieval?",
        "How does reranking work in RAG systems?",
        "What is the advantage of parent-child chunking?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        result = rag.query(q)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")


if __name__ == "__main__":
    main()
