"""
Basic RAG Pipeline
==================
A complete RAG system from scratch using:
- Sentence transformers for embeddings
- Cosine similarity for retrieval (no external vector DB)
- Claude for generation

Run: python main.py
"""

import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import anthropic

load_dotenv()

# ─── Document Store ───────────────────────────────────────────────────────────

# Sample knowledge base — replace with your documents
DOCUMENTS = [
    "The Eiffel Tower was built between 1887 and 1889. It stands 330 meters tall.",
    "Claude is an AI assistant made by Anthropic, founded in 2021.",
    "Python was created by Guido van Rossum and first released in 1991.",
    "The Amazon River is the largest river by water discharge in the world.",
    "Photosynthesis is the process plants use to convert sunlight into energy.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "Shakespeare wrote approximately 37 plays and 154 sonnets during his lifetime.",
    "The human genome contains approximately 3 billion base pairs of DNA.",
]


class BasicRAG:
    def __init__(self, documents: list[str]):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = anthropic.Anthropic()
        self.documents = documents
        # Build index at startup
        self.embeddings = self.embedding_model.encode(documents, convert_to_numpy=True)
        print(f"Indexed {len(documents)} documents")

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Find k most similar documents using cosine similarity."""
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)

        # Cosine similarity: dot product of normalized vectors
        doc_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        query_norm = np.linalg.norm(query_embedding)
        normalized_docs = self.embeddings / (doc_norms + 1e-9)
        normalized_query = query_embedding / (query_norm + 1e-9)

        similarities = normalized_docs @ normalized_query.T
        top_k_indices = np.argsort(similarities.flatten())[::-1][:k]

        return [self.documents[i] for i in top_k_indices]

    def generate(self, query: str, context_docs: list[str]) -> str:
        """Generate an answer grounded in the retrieved context."""
        context = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(context_docs))

        message = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=(
                "You are a helpful assistant. Answer questions using only the "
                "provided context. If the context doesn't contain the answer, "
                "say so — do not make up information."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}"
                }
            ]
        )
        return message.content[0].text

    def query(self, question: str, k: int = 3) -> dict:
        """Full RAG pipeline: retrieve then generate."""
        retrieved_docs = self.retrieve(question, k=k)
        answer = self.generate(question, retrieved_docs)
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_docs,
        }


def main():
    rag = BasicRAG(DOCUMENTS)

    questions = [
        "How tall is the Eiffel Tower?",
        "Who created Python and when?",
        "What is photosynthesis?",
        "What is the capital of Mars?",  # Should say it doesn't know
    ]

    for question in questions:
        print(f"\n{'='*60}")
        result = rag.query(question)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"\nSources used:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  [{i}] {source[:80]}...")


if __name__ == "__main__":
    main()
