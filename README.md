<div align="center">

# 🧠 AI Engineering Guide

### From _"I've called the OpenAI API"_ → _"I can design and defend a production AI system."_

The opinionated, interview-ready playbook for engineers crossing into AI — **LLMs · RAG · Prompting · Agents · Evals · LLMOps · AI System Design.**

<p>
  <a href="https://dipakkr.github.io/ai-engineering-guide"><img src="https://img.shields.io/badge/📖_Read_the_live_guide-2563eb?style=for-the-badge" alt="Read the live guide"></a>
</p>

<p>
  <img src="https://img.shields.io/github/stars/dipakkr/ai-engineering-guide?style=flat-square&color=f59e0b" alt="Stars">
  <img src="https://img.shields.io/badge/Updated-August%202026-2563eb?style=flat-square" alt="Updated August 2026">
  <img src="https://img.shields.io/badge/License-MIT-10b981?style=flat-square" alt="License MIT">
  <img src="https://img.shields.io/badge/PRs-welcome-10b981?style=flat-square" alt="PRs Welcome">
  <a href="https://twitter.com/dipakkr_"><img src="https://img.shields.io/badge/Follow-@dipakkr__-1DA1F2?style=flat-square&logo=x" alt="Twitter"></a>
</p>

![Demo](guide-demo.gif)

**9 chapters · 75+ lessons · 400+ checklist items · 100+ diagrams · 5 worked case studies · 30 practice problems**

</div>

---

## ⚡ Why this guide

Most AI content is either academic papers or shallow "build a chatbot in 10 minutes" tutorials. Neither prepares you for the interview question _"design an enterprise RAG system for 10M documents"_ — or for the on-call page when it breaks in production.

This guide is the missing middle: **system architecture, deployment patterns, and operational rigor**, written by an engineer for engineers. Every page is opinionated, includes a diagram, and ends with an interview cheat sheet.

**It's for you if** you're a software engineer (2+ yrs) moving into AI, prepping for AI/ML system-design interviews, or shipping LLM features and want to understand what breaks and why.
**It's *not* for you if** you want ML theory and math (read Goodfellow instead) or academic paper summaries.

---

## ✅ Interview Checklists — start here

Every chapter opens with a **`✅ Chapter Checklist`**: an interview syllabus of **400+ topics**, each tracked at three honest levels of mastery —

> **☐ Learned → ☐ Can explain → ☐ Can design/debug**

Don't mark a topic "done" because you read it. Push the ⭐ high-leverage topics to the third box before an interview. Pick your gap and go:

| | | |
|---|---|---|
| [🐍 Python for AI](00-python-for-ai/00-checklist.md) | [🧬 LLM Internals](01-llm-foundations/00-checklist.md) | [✍️ Prompt & Context](02-prompt-engineering/00-checklist.md) |
| [🔎 RAG & Retrieval](03-retrieval-and-rag/00-checklist.md) | [🤖 Agents & MCP](04-agents-and-orchestration/00-checklist.md) | [🛠️ Agent Harnesses](08-agent-harnesses-and-skills/00-checklist.md) |
| [📏 Evals](05-evaluation/00-checklist.md) | [⚙️ LLMOps & Ops](06-production-and-ops/00-checklist.md) | [🏗️ AI System Design](07-system-design-interview/00-checklist.md) |

---

## 🧭 How to use this guide

- **📖 Read online** — the [live guide](https://dipakkr.github.io/ai-engineering-guide) has full-text search, dark mode, and progress tracking.
- **✅ Study by checklist** — treat the 9 checklists above as your syllabus; drill into the linked lesson for any topic you can't yet explain.
- **💻 Run it locally** — the whole thing is Markdown + a tiny zero-dependency server:

```bash
git clone https://github.com/dipakkr/ai-engineering-guide
cd ai-engineering-guide/docs && python3 server.py
# open http://localhost:8080
```

---

## Table of Contents

> **📋 Every chapter opens with a `✅ Chapter Checklist`** — an interview syllabus that tracks each topic at three levels: **Learned → Can explain → Can design/debug**. Use them to make sure nothing falls through the cracks. Start here: [Python](00-python-for-ai/00-checklist.md) · [LLM](01-llm-foundations/00-checklist.md) · [Prompting](02-prompt-engineering/00-checklist.md) · [RAG](03-retrieval-and-rag/00-checklist.md) · [Agents](04-agents-and-orchestration/00-checklist.md) · [Evals](05-evaluation/00-checklist.md) · [Ops](06-production-and-ops/00-checklist.md) · [Harnesses](08-agent-harnesses-and-skills/00-checklist.md) · [System Design](07-system-design-interview/00-checklist.md).

### [Python for Applied AI](00-python-for-ai/)

The Python + CS foundation AI interviews assume: internals, async, backend, and the DSA screen.

- [00-checklist](00-python-for-ai/00-checklist.md) — ✅ Interview syllabus for this chapter
- [what-is-python-for-ai](00-python-for-ai/what-is-python-for-ai.md) — Why the Python screen still exists for AI roles
- [01-python-fundamentals](00-python-for-ai/01-python-fundamentals.md) — Mutability, generators, decorators, context managers
- [02-internals-and-performance](00-python-for-ai/02-internals-and-performance.md) — The GIL, memory, Big-O of collections
- [03-async-and-concurrency](00-python-for-ai/03-async-and-concurrency.md) — Event loop, gather, semaphores, backpressure
- [04-ai-backend-python](00-python-for-ai/04-ai-backend-python.md) — Pydantic, FastAPI, streaming/SSE, retries
- [05-coding-interviews](00-python-for-ai/05-coding-interviews.md) — The DSA screen: patterns, Big-O, classic problems

### [LLM Foundations](01-llm-foundations/)

How transformers work, tokenization, context windows, when to fine-tune vs RAG.

- [00-checklist](01-llm-foundations/00-checklist.md) — ✅ Interview syllabus for this chapter
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

- [00-checklist](02-prompt-engineering/00-checklist.md) — ✅ Interview syllabus for this chapter
- [01-prompting-patterns](02-prompt-engineering/01-prompting-patterns.md) — Zero-shot to Tree of Thought
- [02-context-engineering](02-prompt-engineering/02-context-engineering.md) — The underrated skill that separates good from great
- [03-structured-generation](02-prompt-engineering/03-structured-generation.md) — Instructor, JSON mode, Outlines
- [04-prompt-optimization](02-prompt-engineering/04-prompt-optimization.md) — DSPy, meta-prompting, eval-driven
- [05-prompt-security](02-prompt-engineering/05-prompt-security.md) — Injection attacks and defenses

### [RAG - Retrieval Augmented Generation](03-retrieval-and-rag/)

The complete RAG stack: chunking, embeddings, vector DBs, hybrid search, advanced patterns.

- [00-checklist](03-retrieval-and-rag/00-checklist.md) — ✅ Interview syllabus for this chapter
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

- [00-checklist](04-agents-and-orchestration/00-checklist.md) — ✅ Interview syllabus for this chapter
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

- [00-checklist](05-evaluation/00-checklist.md) — ✅ Interview syllabus for this chapter
- [01-eval-fundamentals](05-evaluation/01-eval-fundamentals.md) — Why eval is hard, the eval pipeline
- [02-retrieval-and-rag-eval](05-evaluation/02-retrieval-and-rag-eval.md) — Precision@K, MRR, NDCG, RAGAS
- [03-llm-as-judge](05-evaluation/03-llm-as-judge.md) — Pointwise vs pairwise, calibration
- [04-agent-and-e2e-eval](05-evaluation/04-agent-and-e2e-eval.md) — Task completion, A/B testing, continuous eval

### [Production Ops](06-production-and-ops/)

Observability, guardrails, caching, inference infra, cost optimization.

- [00-checklist](06-production-and-ops/00-checklist.md) — ✅ Interview syllabus for this chapter
- [01-observability-and-tracing](06-production-and-ops/01-observability-and-tracing.md) — LangSmith vs Langfuse, what to log
- [02-guardrails-and-safety](06-production-and-ops/02-guardrails-and-safety.md) — Defense-in-depth, NeMo, LlamaGuard, Presidio
- [03-caching-strategies](06-production-and-ops/03-caching-strategies.md) — Multi-layer caching, semantic cache
- [04-inference-infrastructure](06-production-and-ops/04-inference-infrastructure.md) — GPU table, vLLM vs TGI, auto-scaling
- [05-drift-and-monitoring](06-production-and-ops/05-drift-and-monitoring.md) — Drift types, detection, remediation
- [06-mlops-for-llms](06-production-and-ops/06-mlops-for-llms.md) — CI/CD, prompt versioning, blue-green
- [07-cost-optimization](06-production-and-ops/07-cost-optimization.md) — Token optimization, model routing, batch

### [Agent Harnesses and Skills](08-agent-harnesses-and-skills/)

The harness around the model: the loop, tool registry, context management, skills, and permissions that make coding/research agents work.

- [00-checklist](08-agent-harnesses-and-skills/00-checklist.md) — ✅ Interview syllabus for this chapter
- [what-is-an-agent-harness](08-agent-harnesses-and-skills/what-is-an-agent-harness.md) — Model vs harness: why the same model feels different
- [01-harness-fundamentals](08-agent-harnesses-and-skills/01-harness-fundamentals.md) — The model loop and the five pieces
- [02-context-management](08-agent-harnesses-and-skills/02-context-management.md) — Compaction, retrieval, budgeting across a run
- [03-skills](08-agent-harnesses-and-skills/03-skills.md) — Instructions vs tools vs skills, progressive disclosure
- [04-coding-agent-harnesses](08-agent-harnesses-and-skills/04-coding-agent-harnesses.md) — Filesystem, shell, patch/edit, git, sandboxing
- [05-execution](08-agent-harnesses-and-skills/05-execution.md) — Planning, subagents, checkpoints, resumability
- [06-reliability](08-agent-harnesses-and-skills/06-reliability.md) — Verification loops, self-correction, loop detection
- [07-real-systems](08-agent-harnesses-and-skills/07-real-systems.md) — Claude Code, Codex, LangGraph, CrewAI/AutoGen

### [System Design Interview](07-system-design-interview/)

Interview framework, 5 full case studies, 30 practice problems, 60+ conceptual questions.

- [00-checklist](07-system-design-interview/00-checklist.md) — ✅ Design practice log + coverage sequence
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

## 🤝 Contributing

The guide is intentionally opinionated. If you disagree with a recommendation, open an issue with your reasoning and production evidence. PRs welcome for:

- Factual errors or outdated information (especially model specs and pricing)
- Missing failure modes from your production experience
- New diagrams, checklist items, or worked examples

> **Adding a diagram?** Prefer a ` ```mermaid ` block. If you hand-author inline `<svg>`, keep it **blank-line-free** — the Markdown renderer injects `<p>` tags at blank lines, which breaks SVG rendering.

---

<div align="center">

### ⭐ If this helped you, star the repo — it's the #1 way to help other engineers find it.

**Built by [@dipakkr](https://github.com/dipakkr)** · [GitHub](https://github.com/dipakkr) · [Twitter/X](https://twitter.com/dipakkr_) · [LinkedIn](https://linkedin.com/in/dipakkr)

</div>

