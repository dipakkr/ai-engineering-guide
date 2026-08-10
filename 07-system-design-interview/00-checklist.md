# ✅ AI System Design — Interview Checklist

> **TL;DR**: This is not a theory checklist — it's a **practice log**. Design each system below out loud (or on a whiteboard) until you can walk every layer of the coverage sequence without notes. Tick the depth you've reached per design; ⭐ = the designs interviewers reach for most.

**How to use this page**

| Level | Box | You can honestly say... |
|---|---|---|
| **L1 — Learned** | `☐` | "I've read a reference design for this." |
| **L2 — Explain** | `☐` | "I can sketch the architecture unprompted." |
| **L3 — Design/Debug** | `☐` | "I can design it live, defend every tradeoff, and handle follow-ups." |

Tick a box by replacing `☐` with `✅`. ⭐ = highest interview leverage — get these to **L3** first.

**Related chapters**: [Interview Framework](01-interview-framework.md) · [Design Patterns Catalog](02-design-patterns-catalog.md) · [Architecture Templates](03-architecture-templates.md) · [Practice Problems](09-practice-problems.md)

---

## 1. Master the Framework First

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| The 45-minute interview structure ⭐ | ☐ | ☐ | ☐ | [Interview Framework](01-interview-framework.md) |
| Design patterns catalog & decision tree ⭐ | ☐ | ☐ | ☐ | [Design Patterns Catalog](02-design-patterns-catalog.md) |
| Reference architecture templates + cost models ⭐ | ☐ | ☐ | ☐ | [Architecture Templates](03-architecture-templates.md) |
| 60+ conceptual questions (rapid fire) | ☐ | ☐ | ☐ | [Conceptual Questions](10-conceptual-questions.md) |

## 2. The Coverage Sequence (memorize this)

For **every** design, walk these layers in order. Interviewers grade breadth *and* the ability to prioritize under time pressure.

> **Requirements → API → data → ingestion → model → retrieval → prompts → tools → agent architecture → memory → evals → guardrails → observability → caching → scaling → latency → cost → security → failure handling → deployment.**

| Layer | L1 | L2 | L3 | Anchor lesson |
|---|:-:|:-:|:-:|---|
| Requirements (functional + non-functional) ⭐ | ☐ | ☐ | ☐ | [Interview Framework](01-interview-framework.md) |
| API design | ☐ | ☐ | ☐ | [Architecture Templates](03-architecture-templates.md) |
| Data model & storage | ☐ | ☐ | ☐ | [Architecture Templates](03-architecture-templates.md) |
| Ingestion pipeline | ☐ | ☐ | ☐ | [RAG Fundamentals](../03-retrieval-and-rag/01-rag-fundamentals.md) |
| Model choice ⭐ | ☐ | ☐ | ☐ | [Model Landscape](../01-llm-foundations/06-model-landscape.md) |
| Retrieval | ☐ | ☐ | ☐ | [Hybrid Search](../03-retrieval-and-rag/06-hybrid-search.md) |
| Prompts & context | ☐ | ☐ | ☐ | [Context Engineering](../02-prompt-engineering/02-context-engineering.md) |
| Tools & agent architecture ⭐ | ☐ | ☐ | ☐ | [Multi-Agent Systems](../04-agents-and-orchestration/09-multi-agent-systems.md) |
| Memory | ☐ | ☐ | ☐ | [Memory & State](../04-agents-and-orchestration/10-memory-and-state.md) |
| Evals ⭐ | ☐ | ☐ | ☐ | [Eval Fundamentals](../05-evaluation/01-eval-fundamentals.md) |
| Guardrails | ☐ | ☐ | ☐ | [Guardrails & Safety](../06-production-and-ops/02-guardrails-and-safety.md) |
| Observability | ☐ | ☐ | ☐ | [Observability & Tracing](../06-production-and-ops/01-observability-and-tracing.md) |
| Caching | ☐ | ☐ | ☐ | [Caching Strategies](../06-production-and-ops/03-caching-strategies.md) |
| Scaling & latency ⭐ | ☐ | ☐ | ☐ | [Inference Infrastructure](../06-production-and-ops/04-inference-infrastructure.md) |
| Cost ⭐ | ☐ | ☐ | ☐ | [Cost Optimization](../06-production-and-ops/07-cost-optimization.md) |
| Security & tenant isolation | ☐ | ☐ | ☐ | [Case: Enterprise RAG](04-case-enterprise-rag.md) |
| Failure handling | ☐ | ☐ | ☐ | [Agentic Patterns](../04-agents-and-orchestration/11-agentic-patterns.md) |
| Deployment | ☐ | ☐ | ☐ | [MLOps for LLMs](../06-production-and-ops/06-mlops-for-llms.md) |

## 3. Designs to Practice

| System | L1 | L2 | L3 | Reference / anchor |
|---|:-:|:-:|:-:|---|
| ChatGPT-like assistant ⭐ | ☐ | ☐ | ☐ | [Interview Framework](01-interview-framework.md) |
| Enterprise RAG system ⭐ | ☐ | ☐ | ☐ | [Case: Enterprise RAG](04-case-enterprise-rag.md) |
| Perplexity-like research engine ⭐ | ☐ | ☐ | ☐ | [Case: Search Engine](08-case-search-engine.md) |
| Cursor / coding agent ⭐ | ☐ | ☐ | ☐ | [Case: Code Assistant](05-case-code-assistant.md) |
| Customer-support agent ⭐ | ☐ | ☐ | ☐ | [Case: Customer Support](06-case-customer-support.md) |
| Voice AI agent | ☐ | ☐ | ☐ | [Practice Problems](09-practice-problems.md) |
| Deep-research agent | ☐ | ☐ | ☐ | [Practice Problems](09-practice-problems.md) |
| Multi-agent workflow | ☐ | ☐ | ☐ | [Multi-Agent Systems](../04-agents-and-orchestration/09-multi-agent-systems.md) |
| AI data-analysis agent | ☐ | ☐ | ☐ | [Practice Problems](09-practice-problems.md) |
| Document extraction pipeline | ☐ | ☐ | ☐ | [Case: Doc Intelligence](07-case-doc-intelligence.md) |
| Recommendation / personalization AI | ☐ | ☐ | ☐ | [Practice Problems](09-practice-problems.md) |
| Agent with long-term memory | ☐ | ☐ | ☐ | [Memory & State](../04-agents-and-orchestration/10-memory-and-state.md) |
| AI evaluation platform | ☐ | ☐ | ☐ | [Eval Fundamentals](../05-evaluation/01-eval-fundamentals.md) |
| MCP ecosystem / platform | ☐ | ☐ | ☐ | [MCP Protocol](../04-agents-and-orchestration/03-mcp-protocol.md) |
| High-scale LLM gateway ⭐ | ☐ | ☐ | ☐ | [Inference Infrastructure](../06-production-and-ops/04-inference-infrastructure.md) |

---

**Done with this chapter?** Every ⭐ design at L3, coverage sequence memorized cold → you're interview-ready for AI system design. Loop back to any chapter checklist where a layer felt shaky.
