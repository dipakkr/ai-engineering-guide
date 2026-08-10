# ✅ Evals — Interview Checklist

> **TL;DR**: A **syllabus, not notes**. Study each item in the linked lesson, then tick the depth you've actually reached. For the high-leverage (⭐) topics, aim for the **third** column — *Design/Debug*.
>
> **Scope note**: This checklist owns *evaluation*. Its sibling — **observability, tracing, and production monitoring** — lives in the [Production & Ops checklist](../06-production-and-ops/00-checklist.md). Together they cover the full "is it working?" story.

**How to use this page**

| Level | Box | You can honestly say... |
|---|---|---|
| **L1 — Learned** | `☐` | "I've read it and I understand the idea." |
| **L2 — Explain** | `☐` | "I can explain it out loud, unprompted, in an interview." |
| **L3 — Design/Debug** | `☐` | "I can make the tradeoff calls and debug it in a live system." |

Tick a box by replacing `☐` with `✅`. ⭐ = high interview leverage — get these to **L3** first.

**Related chapters**: [Eval Fundamentals](01-eval-fundamentals.md) · [LLM as Judge](03-llm-as-judge.md) · [Agent & E2E Eval](04-agent-and-e2e-eval.md)

---

## 1. Evaluation Fundamentals

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Offline vs online evals ⭐ | ☐ | ☐ | ☐ | [Eval Fundamentals](01-eval-fundamentals.md) |
| Deterministic vs model-based evals ⭐ | ☐ | ☐ | ☐ | [Eval Fundamentals](01-eval-fundamentals.md) |
| Golden datasets ⭐ | ☐ | ☐ | ☐ | [Eval Fundamentals](01-eval-fundamentals.md) |
| Synthetic datasets | ☐ | ☐ | ☐ | [Eval Fundamentals](01-eval-fundamentals.md) |
| Regression suites | ☐ | ☐ | ☐ | [Eval Fundamentals](01-eval-fundamentals.md) |
| Human evaluation | ☐ | ☐ | ☐ | [Eval Fundamentals](01-eval-fundamentals.md) |

## 2. LLM Evals

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Exact match / F1 | ☐ | ☐ | ☐ | [Eval Fundamentals](01-eval-fundamentals.md) |
| BLEU / ROUGE concepts | ☐ | ☐ | ☐ | [Eval Fundamentals](01-eval-fundamentals.md) |
| Semantic similarity | ☐ | ☐ | ☐ | [Eval Fundamentals](01-eval-fundamentals.md) |
| LLM-as-judge ⭐ | ☐ | ☐ | ☐ | [LLM as Judge](03-llm-as-judge.md) |
| Pairwise evaluation ⭐ | ☐ | ☐ | ☐ | [LLM as Judge](03-llm-as-judge.md) |
| Rubric-based scoring | ☐ | ☐ | ☐ | [LLM as Judge](03-llm-as-judge.md) |
| Judge bias & position bias ⭐ | ☐ | ☐ | ☐ | [LLM as Judge](03-llm-as-judge.md) |
| Calibration | ☐ | ☐ | ☐ | [LLM as Judge](03-llm-as-judge.md) |

## 3. RAG Evals

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Retrieval quality ⭐ | ☐ | ☐ | ☐ | [Retrieval & RAG Eval](02-retrieval-and-rag-eval.md) |
| Context precision / recall ⭐ | ☐ | ☐ | ☐ | [Retrieval & RAG Eval](02-retrieval-and-rag-eval.md) |
| Faithfulness / groundedness ⭐ | ☐ | ☐ | ☐ | [Retrieval & RAG Eval](02-retrieval-and-rag-eval.md) |
| Relevance | ☐ | ☐ | ☐ | [Retrieval & RAG Eval](02-retrieval-and-rag-eval.md) |
| Answer correctness | ☐ | ☐ | ☐ | [RAG Evaluation](../03-retrieval-and-rag/11-rag-evaluation.md) |

## 4. Agent Evals

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Task success ⭐ | ☐ | ☐ | ☐ | [Agent & E2E Eval](04-agent-and-e2e-eval.md) |
| Tool selection & tool-call correctness ⭐ | ☐ | ☐ | ☐ | [Agent & E2E Eval](04-agent-and-e2e-eval.md) |
| Trajectory evaluation ⭐ | ☐ | ☐ | ☐ | [Agent & E2E Eval](04-agent-and-e2e-eval.md) |
| Step efficiency | ☐ | ☐ | ☐ | [Agent & E2E Eval](04-agent-and-e2e-eval.md) |
| Planning quality | ☐ | ☐ | ☐ | [Agent & E2E Eval](04-agent-and-e2e-eval.md) |
| Recovery from failures | ☐ | ☐ | ☐ | [Agent & E2E Eval](04-agent-and-e2e-eval.md) |
| Side effects | ☐ | ☐ | ☐ | [Agent & E2E Eval](04-agent-and-e2e-eval.md) |
| Cost / latency in eval | ☐ | ☐ | ☐ | [Agent & E2E Eval](04-agent-and-e2e-eval.md) |

---

**Done with this chapter?** Every ⭐ topic at L3 → you can design an eval harness that actually catches regressions. Next: [Production & Ops Checklist](../06-production-and-ops/00-checklist.md) (observability + LLMOps).
