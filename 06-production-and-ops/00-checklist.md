# ✅ Observability, LLMOps & MLOps — Interview Checklist

> **TL;DR**: A **syllabus, not notes**. Study each item in the linked lesson, then tick the depth you've actually reached. For the high-leverage (⭐) topics, aim for the **third** column — *Design/Debug*.
>
> **Scope note**: This checklist owns *observability, production monitoring, and ops*. Its sibling — *evaluation* (golden datasets, LLM-as-judge, RAG/agent evals) — lives in the [Evaluation checklist](../05-evaluation/00-checklist.md).

**How to use this page**

| Level | Box | You can honestly say... |
|---|---|---|
| **L1 — Learned** | `☐` | "I've read it and I understand the idea." |
| **L2 — Explain** | `☐` | "I can explain it out loud, unprompted, in an interview." |
| **L3 — Design/Debug** | `☐` | "I can make the tradeoff calls and debug it in a live system." |

Tick a box by replacing `☐` with `✅`. ⭐ = high interview leverage — get these to **L3** first.

**Related chapters**: [Observability & Tracing](01-observability-and-tracing.md) · [Inference Infrastructure](04-inference-infrastructure.md) · [MLOps for LLMs](06-mlops-for-llms.md) · [Cost Optimization](07-cost-optimization.md)

---

## 1. Observability

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Traces & spans ⭐ | ☐ | ☐ | ☐ | [Observability & Tracing](01-observability-and-tracing.md) |
| Logging prompts & tool calls ⭐ | ☐ | ☐ | ☐ | [Observability & Tracing](01-observability-and-tracing.md) |
| Token usage & cost per request | ☐ | ☐ | ☐ | [Observability & Tracing](01-observability-and-tracing.md) |
| Latency & TTFT | ☐ | ☐ | ☐ | [Observability & Tracing](01-observability-and-tracing.md) |
| Errors, model & provider tagging | ☐ | ☐ | ☐ | [Observability & Tracing](01-observability-and-tracing.md) |
| Retrieval traces | ☐ | ☐ | ☐ | [Observability & Tracing](01-observability-and-tracing.md) |
| Langfuse / LangSmith / OpenTelemetry ⭐ | ☐ | ☐ | ☐ | [Observability & Tracing](01-observability-and-tracing.md) |

## 2. Production Monitoring

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Dashboards & alerts | ☐ | ☐ | ☐ | [Drift & Monitoring](05-drift-and-monitoring.md) |
| Quality drift ⭐ | ☐ | ☐ | ☐ | [Drift & Monitoring](05-drift-and-monitoring.md) |
| Prompt & model regressions ⭐ | ☐ | ☐ | ☐ | [Drift & Monitoring](05-drift-and-monitoring.md) |
| Feedback loops | ☐ | ☐ | ☐ | [Drift & Monitoring](05-drift-and-monitoring.md) |
| Sampling | ☐ | ☐ | ☐ | [Observability & Tracing](01-observability-and-tracing.md) |
| Red teaming | ☐ | ☐ | ☐ | [Guardrails & Safety](02-guardrails-and-safety.md) |

## 3. Lifecycle (LLMOps)

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Experiment → eval → deploy → monitor loop ⭐ | ☐ | ☐ | ☐ | [MLOps for LLMs](06-mlops-for-llms.md) |
| Model registry | ☐ | ☐ | ☐ | [MLOps for LLMs](06-mlops-for-llms.md) |
| Dataset & prompt versioning ⭐ | ☐ | ☐ | ☐ | [MLOps for LLMs](06-mlops-for-llms.md) |
| Reproducibility | ☐ | ☐ | ☐ | [MLOps for LLMs](06-mlops-for-llms.md) |

## 4. Serving

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| API-based vs self-hosted models ⭐ | ☐ | ☐ | ☐ | [Inference Infrastructure](04-inference-infrastructure.md) |
| vLLM / TGI concepts ⭐ | ☐ | ☐ | ☐ | [Inference Infrastructure](04-inference-infrastructure.md) |
| GPU serving | ☐ | ☐ | ☐ | [Inference Infrastructure](04-inference-infrastructure.md) |
| Autoscaling & load balancing | ☐ | ☐ | ☐ | [Inference Infrastructure](04-inference-infrastructure.md) |
| Batching & streaming | ☐ | ☐ | ☐ | [Inference Infrastructure](04-inference-infrastructure.md) |
| Queues | ☐ | ☐ | ☐ | [Inference Infrastructure](04-inference-infrastructure.md) |

## 5. Reliability

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Retries & exponential backoff ⭐ | ☐ | ☐ | ☐ | [Cost Optimization](07-cost-optimization.md) |
| Timeouts | ☐ | ☐ | ☐ | [Inference Infrastructure](04-inference-infrastructure.md) |
| Circuit breakers | ☐ | ☐ | ☐ | [Inference Infrastructure](04-inference-infrastructure.md) |
| Rate limits | ☐ | ☐ | ☐ | [Cost Optimization](07-cost-optimization.md) |
| Fallbacks & provider failover ⭐ | ☐ | ☐ | ☐ | [Inference Infrastructure](04-inference-infrastructure.md) |
| Graceful degradation | ☐ | ☐ | ☐ | [Inference Infrastructure](04-inference-infrastructure.md) |
| Idempotency | ☐ | ☐ | ☐ | [MLOps for LLMs](06-mlops-for-llms.md) |

## 6. Optimization

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Semantic / prompt caching ⭐ | ☐ | ☐ | ☐ | [Caching Strategies](03-caching-strategies.md) |
| Batching | ☐ | ☐ | ☐ | [Inference Infrastructure](04-inference-infrastructure.md) |
| Model routing ⭐ | ☐ | ☐ | ☐ | [Cost Optimization](07-cost-optimization.md) |
| Small vs large models | ☐ | ☐ | ☐ | [Small Language Models](../01-llm-foundations/07-small-language-models.md) |
| Token budgets | ☐ | ☐ | ☐ | [Cost Optimization](07-cost-optimization.md) |
| Quantization | ☐ | ☐ | ☐ | [Quantization](../01-llm-foundations/08-quantization.md) |
| Latency & cost optimization ⭐ | ☐ | ☐ | ☐ | [Cost Optimization](07-cost-optimization.md) |

## 7. Deployment

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Docker | ☐ | ☐ | ☐ | [MLOps for LLMs](06-mlops-for-llms.md) |
| Kubernetes fundamentals | ☐ | ☐ | ☐ | [MLOps for LLMs](06-mlops-for-llms.md) |
| CI/CD ⭐ | ☐ | ☐ | ☐ | [MLOps for LLMs](06-mlops-for-llms.md) |
| Canary releases & shadow traffic ⭐ | ☐ | ☐ | ☐ | [MLOps for LLMs](06-mlops-for-llms.md) |
| A/B tests | ☐ | ☐ | ☐ | [Agent & E2E Eval](../05-evaluation/04-agent-and-e2e-eval.md) |
| Rollback & feature flags | ☐ | ☐ | ☐ | [MLOps for LLMs](06-mlops-for-llms.md) |

## 8. Security & Governance

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Secrets management | ☐ | ☐ | ☐ | [Guardrails & Safety](02-guardrails-and-safety.md) |
| PII handling & encryption ⭐ | ☐ | ☐ | ☐ | [Guardrails & Safety](02-guardrails-and-safety.md) |
| RBAC & tenant isolation ⭐ | ☐ | ☐ | ☐ | [Case: Enterprise RAG](../07-system-design-interview/04-case-enterprise-rag.md) |
| Audit logs & data retention | ☐ | ☐ | ☐ | [Guardrails & Safety](02-guardrails-and-safety.md) |
| Compliance concepts | ☐ | ☐ | ☐ | [Guardrails & Safety](02-guardrails-and-safety.md) |

## 9. MLOps (Classic)

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Experiment tracking | ☐ | ☐ | ☐ | [MLOps for LLMs](06-mlops-for-llms.md) |
| Feature / data drift ⭐ | ☐ | ☐ | ☐ | [Drift & Monitoring](05-drift-and-monitoring.md) |
| Training pipelines | ☐ | ☐ | ☐ | [MLOps for LLMs](06-mlops-for-llms.md) |
| Retraining triggers | ☐ | ☐ | ☐ | [Drift & Monitoring](05-drift-and-monitoring.md) |
| MLflow concepts | ☐ | ☐ | ☐ | [MLOps for LLMs](06-mlops-for-llms.md) |

---

**Done with this chapter?** Every ⭐ topic at L3 → you can stand up and defend a production LLM platform. Next: [System Design Checklist](../07-system-design-interview/00-checklist.md).
