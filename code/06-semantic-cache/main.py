"""
Semantic Cache
==============
Caches LLM responses by semantic similarity, not exact string match.
"What is ML?" and "Can you explain machine learning?" hit the same cache entry.

Uses:
- Sentence transformers for embeddings
- In-memory vector store (no Redis required to run)
- Cosine similarity threshold for cache hits

Run: python main.py
"""

import os
import time
import hashlib
import json
from dataclasses import dataclass, field
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic

load_dotenv()

# ─── Cache Entry ──────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    query: str
    response: str
    embedding: np.ndarray
    hit_count: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


# ─── Semantic Cache ───────────────────────────────────────────────────────────

class SemanticCache:
    def __init__(
        self,
        similarity_threshold: float = 0.90,
        max_age_seconds: int = 3600,  # 1 hour TTL
        max_entries: int = 1000,
    ):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = similarity_threshold
        self.max_age = max_age_seconds
        self.max_entries = max_entries
        self.entries: list[CacheEntry] = []
        self.stats = {"hits": 0, "misses": 0}

    def _embed(self, text: str) -> np.ndarray:
        return self.embed_model.encode([text], convert_to_numpy=True)[0]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def get(self, query: str) -> str | None:
        """Return cached response if semantically similar query exists."""
        query_embedding = self._embed(query)
        now = time.time()

        best_score = 0.0
        best_entry = None

        for entry in self.entries:
            # Skip expired entries
            if entry.age_seconds > self.max_age:
                continue
            score = self._cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.threshold:
            best_entry.hit_count += 1
            self.stats["hits"] += 1
            print(f"  [CACHE HIT] similarity={best_score:.3f} | original='{best_entry.query[:50]}'")
            return best_entry.response

        self.stats["misses"] += 1
        return None

    def set(self, query: str, response: str):
        """Store a query-response pair."""
        # Evict oldest if at capacity
        if len(self.entries) >= self.max_entries:
            self.entries.sort(key=lambda e: e.created_at)
            self.entries = self.entries[100:]  # Remove oldest 10%

        embedding = self._embed(query)
        self.entries.append(CacheEntry(query=query, response=response, embedding=embedding))

    def invalidate_expired(self):
        """Remove entries older than TTL."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.age_seconds <= self.max_age]
        removed = before - len(self.entries)
        if removed:
            print(f"  [CACHE] Evicted {removed} expired entries")

    def hit_rate(self) -> float:
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total > 0 else 0.0

    def summary(self) -> dict:
        return {
            "entries": len(self.entries),
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{self.hit_rate():.1%}",
        }


# ─── Cached LLM Client ────────────────────────────────────────────────────────

class CachedLLMClient:
    def __init__(self, cache: SemanticCache):
        self.cache = cache
        self.client = anthropic.Anthropic()

    def complete(self, query: str, system_prompt: str = "") -> str:
        # Check cache first
        cached = self.cache.get(query)
        if cached:
            return cached

        # Cache miss — call the LLM
        messages = [{"role": "user", "content": query}]
        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=system_prompt or "You are a helpful assistant. Be concise.",
            messages=messages,
        )
        answer = response.content[0].text

        # Store in cache
        self.cache.set(query, answer)
        return answer


def main():
    cache = SemanticCache(similarity_threshold=0.88)
    llm = CachedLLMClient(cache)

    # These queries are semantically similar — should hit cache after first call
    query_groups = [
        [
            "What is machine learning?",
            "Can you explain machine learning to me?",
            "What does machine learning mean?",
            "Explain what ML is",
        ],
        [
            "How does photosynthesis work?",
            "What is the process of photosynthesis?",
            "Can you explain photosynthesis?",
        ],
        [
            "What is the capital of France?",
            "Which city is the capital of France?",
            "Tell me France's capital city",
        ],
    ]

    for group in query_groups:
        print(f"\n{'='*60}")
        for query in group:
            start = time.time()
            answer = llm.complete(query)
            elapsed = (time.time() - start) * 1000
            print(f"Q: {query}")
            print(f"A: {answer[:80]}... ({elapsed:.0f}ms)")
            print()

    print("\n" + "="*60)
    print("Cache Summary:")
    summary = cache.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
