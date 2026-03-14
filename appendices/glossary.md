# Glossary

Terms used throughout this guide, defined concisely.

---

## A

**Agent**: A system where an LLM drives a loop: observe state, reason about actions, call tools, repeat until task completion. Distinguished from a single LLM call by the presence of a loop and external tool use.

**Attention**: The mechanism in transformers where each token computes relevance weights over all other tokens and aggregates their representations. The core operation that lets tokens "look at" each other.

**AWQ (Activation-Aware Weight Quantization)**: Post-training quantization that identifies and protects the most important weights using activation magnitudes. Generally better quality than GPTQ at the same bit width.

---

## B

**BM25**: A probabilistic text ranking function used in traditional search engines. Based on term frequency and inverse document frequency with length normalization. Still used in hybrid search alongside vector similarity.

**BPE (Byte Pair Encoding)**: Tokenization algorithm that iteratively merges the most common character pairs to build a vocabulary. Used by GPT, Claude, Llama, and most modern LLMs.

**Batch Inference**: Processing multiple requests together for higher GPU utilization. Increases throughput but adds latency for individual requests.

---

## C

**Chain-of-Thought (CoT)**: Prompting technique where the model writes out reasoning steps before producing a final answer. Significantly improves performance on reasoning tasks.

**Chunking**: Splitting documents into smaller pieces for indexing and retrieval. Chunk size affects both what gets retrieved and how much context is needed per chunk.

**Context Window**: The maximum number of tokens a model can process in a single call. Includes system prompt, conversation history, retrieved context, and the response.

**Cross-Encoder**: A model that takes a query and document pair as input and scores their relevance. More accurate than bi-encoder cosine similarity but O(n) in queries × documents — too slow for first-stage retrieval, used for reranking.

---

## D

**Dense Retrieval**: Retrieval using vector similarity between embeddings. Captures semantic similarity but can miss exact keyword matches.

**Distillation**: Training a small "student" model to mimic a large "teacher" model's behavior. Achieves 90-95% of teacher quality at much lower inference cost.

**DPO (Direct Preference Optimization)**: Alternative to RLHF that uses preference data to fine-tune models without training a separate reward model. More stable than PPO-based RLHF.

---

## E

**Embedding**: A dense vector representation of text (or other data) where semantic similarity corresponds to geometric proximity in vector space. Foundation for vector search and RAG.

**Embedding Model**: A model that converts text to embeddings. Separate from generative models — embedding models don't generate text, they produce fixed-size vectors.

---

## F

**Fine-Tuning**: Continuing training of a pre-trained model on task-specific data to adapt its behavior. LoRA is the standard efficient fine-tuning technique.

**FlashAttention**: An implementation of attention that tiles computation in fast GPU SRAM rather than materializing the full attention matrix. Makes long-context inference practical.

**Function Calling**: The ability for an LLM to output structured tool calls that trigger external code execution. Also called "tool use."

---

## G

**GQA (Grouped Query Attention)**: Attention variant where multiple query heads share a single key-value head. Reduces KV cache size by 4-8x. Used in Llama 3, Mistral.

**GPTQ**: Post-training quantization technique that quantizes weights layer-by-layer using calibration data to minimize quantization error.

**GGUF**: File format for quantized models used by llama.cpp. Supports CPU inference with multiple quantization levels (Q2 through Q8).

**Guardrails**: Systems that filter, validate, or modify LLM inputs/outputs to prevent harmful behavior, enforce policies, or ensure output quality.

---

## H

**Hallucination**: When an LLM generates plausible-sounding but false information. More common with closed-book question answering than RAG, where retrieved context grounds the response.

**HyDE (Hypothetical Document Embeddings)**: Query expansion technique where the LLM generates a hypothetical answer to a query, then uses the embedding of that answer for retrieval. Helps bridge the query-document vocabulary gap.

**HNSW (Hierarchical Navigable Small World)**: Graph-based approximate nearest neighbor index. Best recall/speed tradeoff for in-memory vector search. Used by most vector databases.

---

## I

**IVF (Inverted File Index)**: Clustering-based ANN index that divides vectors into clusters and searches only nearby clusters at query time. More scalable than HNSW for very large datasets.

**Instructor**: Python library that uses Pydantic models to enforce structured JSON output from LLMs via function calling. The standard for reliable structured generation.

---

## J

**JSON Mode**: API feature that guarantees a model outputs valid JSON. Less powerful than structured output via function calling — doesn't enforce a specific schema.

---

## K

**KV Cache**: The stored key-value tensors for all previously processed tokens in a generation. Avoids recomputation but consumes VRAM linearly with sequence length.

**k-NN (k-Nearest Neighbors)**: Exact nearest neighbor search. Finds the truly closest k vectors but scales as O(n) per query. Only practical for small datasets (<100K vectors).

---

## L

**LangChain**: Python/JavaScript framework for building LLM applications. Provides chains, agents, and integrations with tools and data sources.

**LangGraph**: Graph-based agent framework (built on LangChain) for stateful multi-step agent workflows. Uses nodes and edges to define agent behavior.

**LLM-as-Judge**: Using an LLM to evaluate other LLM outputs. More scalable than human evaluation; requires calibration against human scores.

**LoRA (Low-Rank Adaptation)**: Fine-tuning technique that trains small adapter matrices alongside frozen base model weights. Updates ~0.5% of parameters while achieving task-specific adaptation.

**Lost-in-the-Middle**: Observed degradation in LLM performance for information positioned in the middle of long contexts. Models recall beginning and end more reliably than middle sections.

---

## M

**MCP (Model Context Protocol)**: Anthropic's open protocol for connecting LLMs to external data sources and tools. Standardizes how agents interact with external systems.

**MMR (Maximal Marginal Relevance)**: Reranking algorithm that balances relevance with diversity. Selects documents that are relevant to the query but not redundant with already-selected documents.

**MoE (Mixture of Experts)**: Architecture where each token is routed to a subset of "expert" FFN layers rather than using all layers. Reduces per-token compute while maintaining total parameter count.

**Multi-Query Retrieval**: Query expansion technique that generates multiple variations of the user's query, retrieves for each, and merges results. Improves recall for ambiguous or complex queries.

---

## N

**NDCG (Normalized Discounted Cumulative Gain)**: Ranking metric that accounts for both relevance and position. Highly relevant documents at the top score more than relevant documents at the bottom.

**NER (Named Entity Recognition)**: Task of identifying and classifying named entities (people, places, organizations) in text.

---

## P

**PagedAttention**: vLLM's KV cache management system that uses virtual memory paging principles to efficiently manage KV cache across concurrent requests.

**Parent-Child Chunking**: Chunking strategy where small child chunks are indexed for precise retrieval but large parent chunks are returned to the LLM for full context.

**Perplexity**: Measure of how well a language model predicts a text sample. Lower perplexity = better prediction. Used to evaluate language model quality during training.

**PII (Personally Identifiable Information)**: Information that can be used to identify individuals. Must be handled carefully (not logged, redacted in prompts) in healthcare, financial, and regulated applications.

**Prompt Caching**: API feature where a stable prefix (system prompt, large context) is cached server-side and reads are billed at a fraction of the write price. 90% cost reduction for cache reads.

**Prompt Injection**: Attack where malicious content in input data overrides the system prompt or causes the LLM to take unintended actions.

---

## Q

**QLoRA**: Combines 4-bit quantization with LoRA to enable fine-tuning large models on consumer GPUs. The base model is stored at 4-bit precision; LoRA adapters are trained at BF16.

**Quantization**: Reducing the numerical precision of model weights to save memory and/or increase inference speed. INT8 is usually safe; INT4 requires quality testing.

---

## R

**RAG (Retrieval-Augmented Generation)**: Architecture that retrieves relevant documents from an external store and includes them in the LLM's context before generation. Reduces hallucination and enables up-to-date knowledge.

**RAGAS**: Evaluation framework for RAG systems with four metrics: context precision, context recall, faithfulness, and answer relevancy.

**RBAC (Role-Based Access Control)**: Access control model where permissions are associated with roles. Relevant in multi-tenant RAG systems where users should only access their organization's documents.

**ReAct**: Reasoning + Acting prompting pattern for agents: reason about what to do, act (call a tool), observe the result, repeat. Standard pattern for LLM agents.

**Reranking**: Second-stage retrieval step using a more accurate (but slower) model to reorder the top-k results from initial retrieval.

**RoPE (Rotary Position Embedding)**: Position encoding that rotates Q and K vectors by position-dependent angles. Encodes relative position information; enables context length extension.

**RRF (Reciprocal Rank Fusion)**: Algorithm for combining ranked lists from multiple retrieval systems. Robust to different score scales; standard for hybrid dense + sparse retrieval fusion.

---

## S

**Semantic Caching**: Caching LLM responses keyed by semantic similarity rather than exact query match. Hits cache for paraphrased versions of previously seen queries.

**SFT (Supervised Fine-Tuning)**: First stage after pre-training: fine-tune on curated (instruction, response) pairs to teach the model to follow instructions.

**SLM (Small Language Model)**: Models in the 1B-13B parameter range that can run on laptops or phones. Trade capability for cost and privacy.

**Sparse Retrieval**: Retrieval using term-based methods (BM25, TF-IDF). Efficient exact match but misses semantic similarity.

**Speculative Decoding**: Inference optimization where a small draft model proposes tokens, verified in parallel by the large model. Increases throughput for long generations.

---

## T

**Temperature**: Sampling parameter that controls output randomness. 0 = deterministic (greedy); 1 = standard sampling; >1 = more random. Set to 0 for deterministic structured tasks.

**TGI (Text Generation Inference)**: HuggingFace's production inference server for LLMs. Alternative to vLLM with different tradeoffs.

**Tool Use**: See Function Calling.

**Top-k Sampling**: Keeps only the k most probable next tokens and samples from them. Truncates the long tail of unlikely tokens.

**Top-p (Nucleus) Sampling**: Keeps the smallest set of tokens whose cumulative probability exceeds p, then samples from them. More dynamic than top-k.

**Trajectory Evaluation**: Evaluating an agent's sequence of reasoning and tool calls, not just the final answer. Checks whether the agent took an efficient and correct path.

---

## V

**Vector Database**: Specialized database for storing and querying high-dimensional vectors. Supports approximate nearest neighbor search with metadata filtering. Examples: Pinecone, Weaviate, Qdrant, Chroma.

**vLLM**: Production LLM inference engine with PagedAttention for efficient KV cache management. Standard for self-hosted serving.

---

## W

**Waterfall Fallback**: Reliability pattern where a primary model failure triggers automatic retry with a backup model. Common pattern: Claude Sonnet → Claude Haiku → cached response.
