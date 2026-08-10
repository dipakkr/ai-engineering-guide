# ✅ Prompt & Context Engineering — Interview Checklist

> **TL;DR**: A **syllabus, not notes**. Study each item in the linked lesson, then tick the depth you've actually reached. For the high-leverage (⭐) topics, aim for the **third** column — *Design/Debug*.

**How to use this page**

| Level | Box | You can honestly say... |
|---|---|---|
| **L1 — Learned** | `☐` | "I've read it and I understand the idea." |
| **L2 — Explain** | `☐` | "I can explain it out loud, unprompted, in an interview." |
| **L3 — Design/Debug** | `☐` | "I can make the tradeoff calls and debug it in a live system." |

Tick a box by replacing `☐` with `✅`. ⭐ = high interview leverage — get these to **L3** first.

**Related chapters**: [Context Engineering](02-context-engineering.md) · [Structured Generation](03-structured-generation.md) · [Prompt Security](05-prompt-security.md)

---

## 1. Fundamentals

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| System / user / assistant roles ⭐ | ☐ | ☐ | ☐ | [What is Prompt Engineering?](what-is-prompt-engineering.md) |
| Instruction hierarchy | ☐ | ☐ | ☐ | [Prompting Patterns](01-prompting-patterns.md) |
| Zero-shot / few-shot ⭐ | ☐ | ☐ | ☐ | [Prompting Patterns](01-prompting-patterns.md) |
| Delimiters | ☐ | ☐ | ☐ | [Prompting Patterns](01-prompting-patterns.md) |
| Constraints & examples | ☐ | ☐ | ☐ | [Prompting Patterns](01-prompting-patterns.md) |
| Structured outputs & JSON schemas ⭐ | ☐ | ☐ | ☐ | [Structured Generation](03-structured-generation.md) |

## 2. Reasoning & Task Design

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Decomposition | ☐ | ☐ | ☐ | [Prompting Patterns](01-prompting-patterns.md) |
| Planning | ☐ | ☐ | ☐ | [Prompting Patterns](01-prompting-patterns.md) |
| Chain-of-Thought & self-consistency ⭐ | ☐ | ☐ | ☐ | [Prompting Patterns](01-prompting-patterns.md) |
| ReAct | ☐ | ☐ | ☐ | [Agent Fundamentals](../04-agents-and-orchestration/01-agent-fundamentals.md) |
| Tool-use prompting | ☐ | ☐ | ☐ | [Tool Use & Function Calling](../04-agents-and-orchestration/02-tool-use-and-function-calling.md) |
| Reflection / critique patterns | ☐ | ☐ | ☐ | [Agentic Patterns](../04-agents-and-orchestration/11-agentic-patterns.md) |
| Prompt chaining | ☐ | ☐ | ☐ | [Prompting Patterns](01-prompting-patterns.md) |

## 3. Context Engineering

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Context selection ⭐ | ☐ | ☐ | ☐ | [Context Engineering](02-context-engineering.md) |
| Context ordering & relevance | ☐ | ☐ | ☐ | [Context Engineering](02-context-engineering.md) |
| Lost-in-the-middle ⭐ | ☐ | ☐ | ☐ | [Context Engineering](02-context-engineering.md) |
| Context compression & summarization | ☐ | ☐ | ☐ | [Context Engineering](02-context-engineering.md) |
| Dynamic context | ☐ | ☐ | ☐ | [Context Engineering](02-context-engineering.md) |
| Retrieval + prompts | ☐ | ☐ | ☐ | [Context Engineering](02-context-engineering.md) |
| Long-context tradeoffs ⭐ | ☐ | ☐ | ☐ | [Context Windows](../01-llm-foundations/04-context-windows.md) |

## 4. Reliability

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Prompt injection ⭐ | ☐ | ☐ | ☐ | [Prompt Security](05-prompt-security.md) |
| Indirect injection ⭐ | ☐ | ☐ | ☐ | [Prompt Security](05-prompt-security.md) |
| Jailbreak concepts | ☐ | ☐ | ☐ | [Prompt Security](05-prompt-security.md) |
| Instruction conflicts | ☐ | ☐ | ☐ | [Prompt Security](05-prompt-security.md) |
| Grounding & hallucination mitigation ⭐ | ☐ | ☐ | ☐ | [Guardrails & Safety](../06-production-and-ops/02-guardrails-and-safety.md) |
| Output validation | ☐ | ☐ | ☐ | [Structured Generation](03-structured-generation.md) |
| Retries / fallbacks | ☐ | ☐ | ☐ | [Structured Generation](03-structured-generation.md) |

## 5. Optimization

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Prompt versioning | ☐ | ☐ | ☐ | [MLOps for LLMs](../06-production-and-ops/06-mlops-for-llms.md) |
| Prompt evals ⭐ | ☐ | ☐ | ☐ | [Prompt Optimization](04-prompt-optimization.md) |
| A/B testing | ☐ | ☐ | ☐ | [Agent & E2E Eval](../05-evaluation/04-agent-and-e2e-eval.md) |
| Token optimization | ☐ | ☐ | ☐ | [Cost Optimization](../06-production-and-ops/07-cost-optimization.md) |
| Caching | ☐ | ☐ | ☐ | [Caching Strategies](../06-production-and-ops/03-caching-strategies.md) |
| Model-specific prompts | ☐ | ☐ | ☐ | [Prompting Patterns](01-prompting-patterns.md) |
| DSPy concepts ⭐ | ☐ | ☐ | ☐ | [DSPy Framework](../04-agents-and-orchestration/06-dspy-framework.md) |

---

**Done with this chapter?** Every ⭐ topic at L3 → you can design a prompt system and defend it against injection. Next: [RAG Checklist](../03-retrieval-and-rag/00-checklist.md).
