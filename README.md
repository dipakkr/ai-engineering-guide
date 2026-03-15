# 🧠 AI Engineering Guide

Designed for software engineers crossing over into AI, this guide focuses on system architecture, deployment patterns, and operational rigor for **LLMs, RAG, Prompt Engineering, Agents, and Evals**. 

![Demo](guide-demo.gif)  

<p align="center">
  <a href="https://github.com/dipakkr"><img src="https://img.shields.io/badge/GitHub-dipakkr-181717?style=flat-square&logo=github" alt="GitHub dipakkr"></a>
  <a href="https://twitter.com/dipakkr_"><img src="https://img.shields.io/badge/Twitter-@dipakkr_-1DA1F2?style=flat-square&logo=x" alt="Twitter dipakkr_"></a>
  <a href="https://linkedin.com/in/dipakkr"><img src="https://img.shields.io/badge/LinkedIn-dipakkr-0A66C2?style=flat-square&logo=linkedin" alt="LinkedIn dipakkr"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Updated-March%202026-blue?style=flat-square" alt="Updated March 2026">
  <img src="https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square" alt="License MIT">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs Welcome">
</p>


## Who This Is For

**This guide is for you if:**

- You're a senior software engineer (5+ years) moving into AI/ML engineering
- You're preparing for system design interviews at AI-focused companies or big tech AI teams
- You build distributed systems and want to understand how AI components change the design
- You want to go from "I've used the OpenAI API" to "I can design and defend a production AI system"

**This guide is NOT for you if:**

- You're looking for ML theory or math (read Goodfellow's Deep Learning textbook instead)
- You want paper summaries without practical context
- You're a researcher who needs academic rigor over engineering pragmatism

---

## Table of Contents

### [LLM Foundations](01-llm-foundations/)

How transformers work, tokenization, context windows, when to fine-tune vs RAG.

- [01-transformer-intuition](01-llm-foundations/01-transformer-intuition.md) — How transformers work, no math
- [02-tokenization](01-llm-foundations/02-tokenization.md) — Tokens are money
- [03-attention-mechanisms](01-llm-foundations/03-attention-mechanisms.md) — Self-attention, KV cache, Flash Attention
- [04-context-windows](01-llm-foundations/04-context-windows.md) — Long context tradeoffs
- [05-training-pipeline](01-llm-foundations/05-training-pipeline.md) — RAG vs fine-tune vs prompt (THE decision)
- [06-model-landscape](01-llm-foundations/06-model-landscape.md) — Model comparison table
- [07-small-language-models](01-llm-foundations/07-small-language-models.md) — When to use Phi/Gemma instead of GPT-4o
- [08-quantization](01-llm-foundations/08-quantization.md) — INT8/INT4, GGUF, running models on cheap hardware
- [09-fine-tuning](01-llm-foundations/09-fine-tuning.md) — LoRA, QLoRA, when NOT to fine-tune
- [10-distillation-and-pruning](01-llm-foundations/10-distillation-and-pruning.md) — Making models cheaper

### [Prompt Engineering](02-prompt-engineering/)

CoT, structured generation, prompt optimization, injection defense.

- [01-prompting-patterns](02-prompt-engineering/01-prompting-patterns.md) — Zero-shot to Tree of Thought
- [02-context-engineering](02-prompt-engineering/02-context-engineering.md) — The underrated skill that separates good from great
- [03-structured-generation](02-prompt-engineering/03-structured-generation.md) — Instructor, JSON mode, Outlines
- [04-prompt-optimization](02-prompt-engineering/04-prompt-optimization.md) — DSPy, meta-prompting, eval-driven
- [05-prompt-security](02-prompt-engineering/05-prompt-security.md) — Injection attacks and defenses

### [RAG - Retrieval Augmented Generation](03-retrieval-and-rag/)

The complete RAG stack: chunking, embeddings, vector DBs, hybrid search, advanced patterns.

- [01-rag-fundamentals](03-retrieval-and-rag/01-rag-fundamentals.md) — What/why/when, the naive pipeline
- [02-embedding-models](03-retrieval-and-rag/02-embedding-models.md) — MTEB, dimensions, Matryoshka
- [03-vector-indexing](03-retrieval-and-rag/03-vector-indexing.md) — HNSW vs IVF, FAISS
- [04-vector-databases](03-retrieval-and-rag/04-vector-databases.md) — Decision matrix, cost at scale
- [05-chunking-strategies](03-retrieval-and-rag/05-chunking-strategies.md) — THE key lever most teams get wrong
- [06-hybrid-search](03-retrieval-and-rag/06-hybrid-search.md) — Dense + BM25 + RRF
- [07-reranking](03-retrieval-and-rag/07-reranking.md) — Cross-encoders, Cohere, two-stage
- [08-query-transformation](03-retrieval-and-rag/08-query-transformation.md) — HyDE, multi-query, decomposition
- [09-advanced-rag-patterns](03-retrieval-and-rag/09-advanced-rag-patterns.md) — GraphRAG, Agentic RAG, Self-RAG, CRAG
- [10-multimodal-rag](03-retrieval-and-rag/10-multimodal-rag.md) — ColPali, PDFs with tables and images
- [11-rag-evaluation](03-retrieval-and-rag/11-rag-evaluation.md) — RAGAS, debug flowchart

### [Agents and Orchestration](04-agents-and-orchestration/)

ReAct, tool use, MCP, LangGraph, multi-agent systems, memory.

- [01-agent-fundamentals](04-agents-and-orchestration/01-agent-fundamentals.md) — ReAct, perception-action loop, failure modes
- [02-tool-use-and-function-calling](04-agents-and-orchestration/02-tool-use-and-function-calling.md) — OpenAI vs Claude vs Gemini formats
- [03-mcp-protocol](04-agents-and-orchestration/03-mcp-protocol.md) — Full MCP, server code, security
- [04-langchain-overview](04-agents-and-orchestration/04-langchain-overview.md) — What it does well and where it falls short
- [05-langgraph-deep-dive](04-agents-and-orchestration/05-langgraph-deep-dive.md) — Stateful graphs, persistence, human-in-loop
- [06-dspy-framework](04-agents-and-orchestration/06-dspy-framework.md) — Compile don't prompt
- [07-crewai-and-autogen](04-agents-and-orchestration/07-crewai-and-autogen.md) — Honest assessment of multi-agent frameworks
- [08-llamaindex-haystack](04-agents-and-orchestration/08-llamaindex-haystack.md) — Data frameworks vs orchestration frameworks
- [09-multi-agent-systems](04-agents-and-orchestration/09-multi-agent-systems.md) — When you actually need multiple agents
- [10-memory-and-state](04-agents-and-orchestration/10-memory-and-state.md) — Memory tiers, Mem0, Zep, checkpointing
- [11-agentic-patterns](04-agents-and-orchestration/11-agentic-patterns.md) — Reflection, map-reduce, DAG patterns
- [12-browser-and-computer-use](04-agents-and-orchestration/12-browser-and-computer-use.md) — Playwright, Claude Computer Use

### [LLM Evaluation](05-evaluation/)

How to actually measure if your system works: RAGAS, LLM-as-judge, production eval.

- [01-eval-fundamentals](05-evaluation/01-eval-fundamentals.md) — Why eval is hard, the eval pipeline
- [02-retrieval-and-rag-eval](05-evaluation/02-retrieval-and-rag-eval.md) — Precision@K, MRR, NDCG, RAGAS
- [03-llm-as-judge](05-evaluation/03-llm-as-judge.md) — Pointwise vs pairwise, calibration
- [04-agent-and-e2e-eval](05-evaluation/04-agent-and-e2e-eval.md) — Task completion, A/B testing, continuous eval

### [Production Ops](06-production-and-ops/)

Observability, guardrails, caching, inference infra, cost optimization.

- [01-observability-and-tracing](06-production-and-ops/01-observability-and-tracing.md) — LangSmith vs Langfuse, what to log
- [02-guardrails-and-safety](06-production-and-ops/02-guardrails-and-safety.md) — Defense-in-depth, NeMo, LlamaGuard, Presidio
- [03-caching-strategies](06-production-and-ops/03-caching-strategies.md) — Multi-layer caching, semantic cache
- [04-inference-infrastructure](06-production-and-ops/04-inference-infrastructure.md) — GPU table, vLLM vs TGI, auto-scaling
- [05-drift-and-monitoring](06-production-and-ops/05-drift-and-monitoring.md) — Drift types, detection, remediation
- [06-mlops-for-llms](06-production-and-ops/06-mlops-for-llms.md) — CI/CD, prompt versioning, blue-green
- [07-cost-optimization](06-production-and-ops/07-cost-optimization.md) — Token optimization, model routing, batch

### [System Design Interview](07-system-design-interview/)

Interview framework, 5 full case studies, 30 practice problems, 60+ conceptual questions.

- [01-interview-framework](07-system-design-interview/01-interview-framework.md) — The 45-min structure. Worth the whole repo.
- [02-design-patterns-catalog](07-system-design-interview/02-design-patterns-catalog.md) — Full catalog with decision tree
- [03-architecture-templates](07-system-design-interview/03-architecture-templates.md) — 6 reference architectures with cost models
- [04-case-enterprise-rag](07-system-design-interview/04-case-enterprise-rag.md) — Full worked design: enterprise knowledge base
- [05-case-code-assistant](07-system-design-interview/05-case-code-assistant.md) — Full worked design: GitHub Copilot-style
- [06-case-customer-support](07-system-design-interview/06-case-customer-support.md) — Full worked design: support automation
- [07-case-doc-intelligence](07-system-design-interview/07-case-doc-intelligence.md) — Full worked design: document understanding
- [08-case-search-engine](07-system-design-interview/08-case-search-engine.md) — Full worked design: AI-powered search
- [09-practice-problems](07-system-design-interview/09-practice-problems.md) — 30 problems with solution skeletons
- [10-conceptual-questions](07-system-design-interview/10-conceptual-questions.md) — 60+ questions with full conversational answers

### [Appendices](appendices/)

Model pricing, glossary, cost formulas, essential papers.

- [model-pricing-reference](appendices/model-pricing-reference.md) — Current pricing for all major models
- [glossary](appendices/glossary.md) — Terms defined in plain English
- [cost-estimation-formulas](appendices/cost-estimation-formulas.md) — Spreadsheet-ready formulas
- [essential-papers](appendices/essential-papers.md) — The 20 papers worth reading

### [Demo Samples](code/)

Working implementations: RAG pipeline, LangGraph agent, MCP server, eval pipeline.

- [01-basic-rag](code/01-basic-rag/) — Minimal RAG in 100 lines
- [02-advanced-rag](code/02-advanced-rag/) — Hybrid search + reranking
- [03-langgraph-agent](code/03-langgraph-agent/) — Stateful agent with tools
- [04-mcp-server](code/04-mcp-server/) — Working MCP server
- [05-eval-pipeline](code/05-eval-pipeline/) — RAGAS + LLM-as-judge
- [06-semantic-cache](code/06-semantic-cache/) — Semantic caching with Redis
- [07-structured-output](code/07-structured-output/) — Instructor + Pydantic

---

## Contributing

The guide is intentionally opinionated. If you disagree with a recommendation, open an issue with your reasoning and production evidence. PRs welcome for:

- Factual errors or outdated information (especially model specs and pricing)
- Missing failure modes from your production experience

