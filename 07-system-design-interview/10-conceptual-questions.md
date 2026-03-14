# Conceptual Questions

> **How to use this**: These are rapid-fire questions you should be able to answer in 1-3 sentences. The goal is building fluency. If you can't answer a question confidently in 30 seconds, the corresponding file has a deeper treatment. Read the full answer once, then come back and try again without looking.

**Prerequisites**: [Interview Framework](01-interview-framework.md)
**Related**: [Practice Problems](09-practice-problems.md)

---

## RAG and Retrieval (25 questions)

**Q: What is RAG and why use it instead of fine-tuning?**
RAG retrieves relevant documents at query time and injects them into the prompt, giving the model access to external knowledge without training. Use RAG when knowledge changes frequently, when the corpus is large, or when you need citations. Fine-tune when you want to change the model's behavior or style, not when you want to add knowledge.

**Q: What is the difference between dense and sparse retrieval?**
Dense retrieval (embeddings) captures semantic meaning by mapping text to continuous vectors, finding semantically similar content. Sparse retrieval (BM25, TF-IDF) matches exact keyword overlaps, finding documents with the same terminology. Dense handles paraphrasing; sparse handles exact terms like model numbers or names. Production systems use both.

**Q: What is HyDE?**
Hypothetical Document Embeddings: generate a hypothetical answer to the query, embed that instead of the raw query. The hypothetical answer is longer and more domain-aligned, so it embeds more similarly to the actual relevant documents. Helps with vocabulary mismatch between short queries and long documents.

**Q: What is a reranker and when do you add one?**
A reranker is a cross-encoder that scores query-document relevance jointly (both at the same time). More accurate than embedding cosine similarity but too slow to run on the full corpus. Add it as a second stage: retrieve top-50 with fast ANN search, rerank to top-5. Add when context precision (fraction of retrieved chunks that are relevant) is below 0.70.

**Q: What is the lost-in-the-middle problem?**
LLMs recall information from the beginning and end of the context window better than the middle. Studies show 15-20% accuracy loss for information placed centrally in long contexts. Fix: place critical information at the start or end, not the middle.

**Q: What is chunking and what are the main strategies?**
Chunking splits documents into pieces for indexing. Fixed-size: split at N tokens (simple, loses context at boundaries). Semantic: split at semantic boundaries (better quality, slower). Parent-child: index small chunks for retrieval, inject large parent chunks for context. AST-based: split code at function/class boundaries. Chunk size is the highest-impact parameter to tune in most RAG systems.

**Q: When does hybrid search outperform pure dense search?**
When queries contain specific terms that matter exactly: product names, model numbers, error codes, technical identifiers, names. Dense search finds semantically similar content; BM25 finds exact term matches. For technical documentation, legal text, or product catalogs, hybrid typically improves recall by 10-20%.

**Q: What is context precision vs context recall (RAGAS)?**
Context precision: fraction of retrieved chunks that are actually relevant (measures noise). Context recall: fraction of relevant information that was actually retrieved (measures completeness). Low precision → add reranking or reduce top-K. Low recall → improve chunking, add hybrid search, or increase top-K.

**Q: What is the difference between embedding search and keyword search for FAQs?**
Keyword search requires the user to use the same words as the stored answer. Embedding search handles paraphrasing: "how do I cancel" and "procedure for terminating subscription" return the same result. For FAQ-style applications, embedding search dramatically improves recall.

**Q: What is a knowledge graph and when does it improve RAG?**
A knowledge graph stores entities and relationships (Person → WORKS_AT → Company). It improves RAG for multi-hop queries ("Who manages the team that built the product that John works on?") where the answer requires chaining relationships not captured in any single document. GraphRAG constructs a knowledge graph during indexing and uses it during retrieval.

**Q: How do you keep a RAG index fresh when documents change?**
Webhook-triggered re-index for individual document changes (fast). Scheduled batch re-index nightly (catches missed changes). Tombstone records for deletions (mark as deleted, nightly cleanup). Document freshness metadata to weight recent documents in ranking. The right approach depends on how often documents change and how stale content affects quality.

**Q: What is query decomposition and when is it useful?**
Breaking a compound question into independent sub-questions, retrieving for each, then synthesizing. Useful for questions like "Compare A and B" (requires retrieving information about both separately). Over-use adds latency; add a complexity classifier to only decompose clearly compound queries.

**Q: What is multi-query retrieval?**
Generating 3-5 rephrasings of the query, retrieving for each, and merging results with RRF. Improves recall for ambiguous queries where the right answer might be described with different terminology. Cost: one LLM call + N retrievals. Use when recall is the bottleneck.

---

## Agents and Orchestration (20 questions)

**Q: What is the ReAct pattern?**
Reason-Act: the agent alternates between reasoning (thinking about what to do) and acting (calling a tool). After each tool result, it reasons again before the next action. The loop continues until the task is done or a maximum iteration limit is hit.

**Q: What is the difference between a tool and a resource in MCP?**
Tools are functions the model can call to take actions (create file, query database). Resources are data the model can read (current file contents, database schemas). Tools change state; resources provide context.

**Q: When should you use an agent vs a simple chain?**
Use a chain when the steps are known in advance and don't depend on intermediate results. Use an agent when the task requires decision-making based on what previous steps returned, or when you don't know how many steps will be needed. Most tasks don't need agents; start with chains.

**Q: What is human-in-loop in agent systems?**
Pausing the agent before executing high-risk or irreversible actions to get human approval. In LangGraph, implemented with `interrupt_before=["node_name"]`. The agent generates the action; the human reviews and approves or modifies before execution.

**Q: What is the difference between LangGraph and LangChain?**
LangChain is a collection of components for building LLM applications (document loaders, chains, vector store integrations). LangGraph is a framework for building stateful, multi-step agent workflows as directed graphs, with persistence and human-in-loop support built in. LangGraph is more specialized and better for complex agent workflows.

**Q: What is function calling (tool use)?**
Structured API for LLMs to request external function execution. The model receives a JSON schema of available functions, outputs a structured JSON call when it needs information, and the application executes the function and returns the result. Enables agents to interact with external systems reliably.

**Q: What is the difference between supervisor and peer multi-agent architectures?**
Supervisor: a central orchestrator assigns tasks to specialized worker agents and synthesizes their results. Clear coordination but bottleneck at the supervisor. Peer (pipeline): agents pass results sequentially from one to the next, each transforming the output. Simpler but less flexible for parallel work. Use supervisor for parallel specialization; use pipeline for sequential transformation.

**Q: What is tool bleeding and how do you prevent it?**
When an agent calls tools that are outside the intended scope for the current task. Example: an agent authorized to read order data accidentally calls a tool that modifies orders. Prevention: principle of least privilege (only provide the tools needed for this specific task), scope-check tool inputs before execution.

**Q: What is the plan-and-execute pattern?**
The agent first generates a complete plan (list of steps), then executes each step sequentially. More predictable than pure ReAct (you can review the plan), but less adaptive (the plan doesn't update based on new information). Good for tasks with well-defined structure; less good for open-ended exploration.

**Q: What is DSPy and how does it differ from standard prompting?**
DSPy treats prompts as learnable programs. Instead of hand-writing instructions, you define input/output signatures, and DSPy's optimizers automatically find the best examples and instructions by testing against your eval set. Standard prompting is manual; DSPy is algorithmic optimization.

---

## LLM Foundations (15 questions)

**Q: What is the difference between RLHF and DPO?**
Both align language models with human preferences. RLHF trains a separate reward model from human preference comparisons, then uses PPO to optimize the LLM against it (complex, unstable). DPO (Direct Preference Optimization) directly trains on preference pairs without a separate reward model, making it simpler and more stable. DPO is the modern default.

**Q: What is LoRA?**
Low-Rank Adaptation: adds small trainable rank-decomposition matrices to the frozen base model. Instead of training all model parameters (billions), you train ~1-5% the parameters via the low-rank matrices. Enables fine-tuning on a single consumer GPU. QLoRA adds 4-bit quantization to reduce VRAM further.

**Q: When should you fine-tune instead of using prompting?**
Fine-tune when you need to change the model's style or behavior in ways prompting can't achieve (following a specific response format consistently, specialized domain vocabulary, task-specific reasoning patterns). Do not fine-tune to add knowledge (use RAG). The decision tree: can a good prompt solve this? → Use prompting. Does prompting consistently fail? → Fine-tune.

**Q: What is quantization and what are the tradeoffs?**
Reducing model weight precision (FP32 → INT8 → INT4) to reduce memory and increase speed. INT8 typically loses <1% quality and cuts VRAM in half. INT4 (AWQ, GPTQ) cuts VRAM to ~25% with 2-5% quality loss. GGUF format runs quantized models on CPU via llama.cpp. Trade-off: smaller model, slight quality loss, can't do training.

**Q: What is the KV cache and why does it matter?**
The key-value tensors from the attention mechanism, stored to avoid recomputing attention for previous tokens. Without KV cache, generating token N requires recomputing attention for all N-1 previous tokens. With KV cache, only the new token's attention is computed. This is why inference is fast for sequential generation but VRAM-intensive for long contexts.

**Q: Why is output more expensive than input for most LLM APIs?**
Input tokens can be processed in parallel (the transformer sees all input positions simultaneously). Output tokens must be generated sequentially (each token requires the previous token). The computational cost of generation is higher per token than encoding, and the sequential nature limits throughput.

**Q: What is the difference between temperature and top-p in generation?**
Temperature scales the logits (higher = more uniform distribution = more random). Top-p (nucleus sampling) restricts generation to the smallest set of tokens whose cumulative probability exceeds p. Both control diversity. Temperature is simpler to reason about; top-p adapts to the probability distribution. Low temperature (0-0.2) for factual tasks; medium (0.5-0.7) for reasoning; high (0.8-1.0) for creative tasks.

---

## Production and Operations (15 questions)

**Q: What is prompt caching and how much does it save?**
Anthropic's API can cache the KV attention states of stable prompt prefixes. Subsequent requests that share the same prefix hit the cache at 10% of the normal input price. For systems with a large, stable system prompt used across many requests, typical savings are 60-90% on input token costs.

**Q: What is semantic caching and when does it help?**
Storing LLM responses and serving cached results for queries with high embedding similarity (>0.95 cosine). Handles paraphrasing: "What's your refund policy?" and "How do I get a refund?" return the same cached response. Most effective for FAQ-style applications with repeated similar queries.

**Q: What is drift in LLM applications?**
Quality degradation over time from three sources: model drift (provider updates the model without announcement), data drift (user query distribution changes), concept drift (the world changes, making stored knowledge stale). Detect with weekly automated eval on a fixed golden set.

**Q: What is LLM-as-judge and its main failure modes?**
Using an LLM to score another LLM's output. Main failure modes: position bias (prefers first answer in pairwise), verbosity bias (rates longer answers higher), self-preference bias (models rate their own outputs favorably). Counteract with calibration against human scores (Pearson correlation > 0.70), randomized answer order, cross-provider judging.

**Q: What is a guardrail in LLM production?**
A filter that validates input before sending to the model or output before returning to the user. Input guardrails catch harmful requests, injection attempts, and out-of-scope queries. Output guardrails catch harmful content, format violations, and confidential data leakage. Defense-in-depth: use multiple independent layers.

**Q: What is the Batch API and when do you use it?**
An API for submitting large numbers of LLM requests asynchronously, processed at a 50% cost discount. Responses arrive in minutes to hours rather than seconds. Use for offline workloads: document processing pipelines, nightly summarization, eval runs, training data generation. Never use for real-time user requests.

**Q: What does RAGAS stand for and what does it measure?**
Retrieval Augmented Generation Assessment. Measures four things: context precision (are retrieved chunks relevant?), context recall (did retrieval find everything needed?), faithfulness (does the answer stay within retrieved context?), and answer relevancy (does the answer address the question?). The four metrics diagnose the two main failure modes: retrieval failure and generation failure.

---

## System Design Concepts (10 questions)

**Q: What is the principle of least privilege in agentic systems?**
Give agents only the tools they need for the specific task, not general-purpose access. An agent that answers customer support questions about orders should have read access to orders, not write access, and certainly not access to all customer data. Limits blast radius when an agent is compromised by prompt injection.

**Q: What is a circuit breaker and why use it in LLM pipelines?**
A pattern that stops sending requests to a failing upstream service after N consecutive failures. Prevents cascading failures and queue buildup. After a timeout period, the circuit "closes" again and allows test requests through. Particularly important for LLM applications that depend on external APIs.

**Q: How do you decide between self-hosted and managed API LLMs?**
Managed API is right for most teams. Self-hosting makes sense when: monthly API spend exceeds ~$50K AND you have an ML infrastructure team AND you have strong data residency requirements. The operational cost of running 24/7 GPU infrastructure (model loading, monitoring, on-call) is often underestimated.

**Q: What is the difference between synchronous and asynchronous LLM processing?**
Synchronous: the user waits for the LLM response (chatbots, real-time tools). Asynchronous: the request is queued and processed later, the user is notified when done (batch jobs, background analysis). Async is 50% cheaper via batch API and avoids rate limits, but adds latency measured in minutes not milliseconds.

**Q: What is streaming in LLM APIs and why does it matter for user experience?**
Streaming sends tokens to the client as they're generated rather than waiting for the complete response. Time-to-first-token (TTFT) drops from 2-3 seconds (full generation) to 300-500ms (first token). Users see text appearing progressively, which feels much faster even if total generation time is the same.
