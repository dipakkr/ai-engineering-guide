# Observability and Tracing

> **TL;DR**: You can't debug what you can't see. Log every LLM call with inputs, outputs, latency, and token counts from day one. LangSmith is the fastest path if you're in the LangChain ecosystem; Langfuse is the open-source alternative that works with any provider. The data you wish you'd logged six months ago is always the data you didn't log.

**Prerequisites**: [Eval Fundamentals](../05-evaluation/01-eval-fundamentals.md), [Agent Fundamentals](../04-agents-and-orchestration/01-agent-fundamentals.md)
**Related**: [MLOps for LLMs](06-mlops-for-llms.md), [Cost Optimization](07-cost-optimization.md), [Drift and Monitoring](05-drift-and-monitoring.md)

---

## What to Log

The minimum viable logging set for every LLM call:

```python
from anthropic import Anthropic
import time
import uuid

client = Anthropic()

def traced_llm_call(
    messages: list[dict],
    system: str = "",
    model: str = "claude-opus-4-6",
    trace_id: str | None = None,
    **kwargs
) -> dict:
    """LLM call with automatic tracing."""
    trace_id = trace_id or str(uuid.uuid4())
    start_time = time.time()

    try:
        response = client.messages.create(
            model=model,
            system=system,
            messages=messages,
            **kwargs
        )

        trace = {
            "trace_id": trace_id,
            "model": model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency_ms": (time.time() - start_time) * 1000,
            "stop_reason": response.stop_reason,
            "status": "success",
            "cost_usd": estimate_cost(model, response.usage),
        }
        log_trace(trace)
        return {"content": response.content[0].text, "trace": trace}

    except Exception as e:
        log_trace({
            "trace_id": trace_id,
            "model": model,
            "latency_ms": (time.time() - start_time) * 1000,
            "status": "error",
            "error": str(e),
        })
        raise

def estimate_cost(model: str, usage) -> float:
    """Rough cost estimate. Update pricing monthly."""
    pricing = {
        "claude-opus-4-6": (0.015, 0.075),       # per 1K tokens: input, output
        "claude-sonnet-4-6": (0.003, 0.015),
        "claude-haiku-4-5-20251001": (0.00025, 0.00125),
    }
    if model not in pricing:
        return 0.0
    input_price, output_price = pricing[model]
    return (usage.input_tokens * input_price + usage.output_tokens * output_price) / 1000
```

**What not to log:** Don't log raw user inputs in shared logs if they contain PII. Log a hashed user ID, not the content. Keep a separate encrypted store for full conversation content that requires explicit access.

---

## LangSmith: Logging for LangChain Ecosystems

[LangSmith](https://smith.langchain.com/) is Anthropic-agnostic tracing built into the LangChain stack. Enable it with environment variables:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "production-rag-v2"

# All LangChain/LangGraph calls now auto-trace to LangSmith
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

llm = ChatAnthropic(model="claude-opus-4-6")
response = llm.invoke([HumanMessage(content="Hello")])
# This call appears in LangSmith dashboard automatically
```

LangSmith gives you a UI to browse traces, compare runs, and annotate examples for your eval set. The killer feature is **datasets**: when you find a conversation in LangSmith that exposed a bug, you can add it to an eval dataset with one click.

**LangSmith pricing (as of 2025):** Free up to 5K traces/month. $40/month for 500K traces. Reasonable for most production systems.

---

## Langfuse: Open-Source Alternative

[Langfuse](https://langfuse.com/) is provider-agnostic, self-hostable, and integrates via a simple SDK. I use it when I want full control or need to run on-premises.

```python
from langfuse import Langfuse
from langfuse.decorators import observe

langfuse = Langfuse(
    public_key="your-public-key",
    secret_key="your-secret-key",
    host="https://cloud.langfuse.com"  # or self-hosted URL
)

@observe()  # Decorator auto-captures inputs, outputs, latency
def rag_pipeline(query: str) -> str:
    # Retrieve
    chunks = retrieve(query)

    # Generate (wrapped in trace context)
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": f"Answer: {query}\n\nContext: {chunks}"}]
    )
    return response.content[0].text

# Manual span creation for nested operations
def complex_agent(task: str) -> str:
    with langfuse.span(name="agent-run", input={"task": task}) as span:
        result = run_agent(task)
        span.update(output={"result": result})
    return result
```

Langfuse captures the full call hierarchy, so multi-step agents show as nested traces with timing at each level.

---

## What the Dashboard Should Show

Build a dashboard (or use LangSmith/Langfuse's built-in one) with these panels:

```
Daily Operations Dashboard:
┌─────────────────────────────────────────────────────────────┐
│ Requests/hour          │ Error rate          │ P95 latency  │
│ ████████████ 1,234     │ 0.3%                │ 2.4s         │
├─────────────────────────────────────────────────────────────┤
│ Cost today: $47.23     │ vs yesterday: +12%  │              │
│ Input tokens: 2.1M     │ Output tokens: 340K │              │
├─────────────────────────────────────────────────────────────┤
│ Model distribution:                                          │
│ claude-opus-4-6 ██████████████ 67%                           │
│ claude-haiku    ████████ 33%                                 │
├─────────────────────────────────────────────────────────────┤
│ Top slow queries (last hour)                                 │
│ [5.2s] "Explain all the regulations for..."                  │
│ [4.8s] "Compare and contrast these 10 documents..."          │
└─────────────────────────────────────────────────────────────┘
```

Alerts to set up:
- Error rate > 1% for 5 minutes: page on-call
- P95 latency > 5 seconds: alert Slack
- Daily cost > 150% of rolling 7-day average: alert
- Cache hit rate drops > 10% from baseline: investigate

---

## Tracing Multi-Step Agents

For agents with multiple LLM calls and tool uses, the trace hierarchy is critical:

```python
from langfuse import Langfuse

langfuse = Langfuse()

def traced_agent(task: str, session_id: str) -> str:
    trace = langfuse.trace(name="customer-support-agent", session_id=session_id)

    # Each LLM call is a span
    reasoning_span = trace.span(name="initial-reasoning")
    initial_analysis = call_llm(task)
    reasoning_span.end(output={"analysis": initial_analysis})

    # Tool calls are separate spans
    tool_span = trace.span(name="lookup-order")
    order_data = lookup_order(extract_order_id(initial_analysis))
    tool_span.end(output=order_data)

    # Final generation
    final_span = trace.span(name="generate-response")
    response = generate_response(task, order_data)
    final_span.end(output={"response": response})

    trace.update(output={"final_response": response})
    return response
```

The trace hierarchy lets you answer: "Why did this specific agent run take 8 seconds?" (The lookup-order span took 6 seconds because the database was slow.) Without the span breakdown, "agent was slow" is the only diagnosis available.

---

## Logging for PII and Compliance

If your application handles PII (names, emails, health data, financial information), logging the raw content of LLM calls can create compliance exposure.

**Strategy: hash IDs, log structure not content**

```python
import hashlib

def log_safely(trace: dict, pii_fields: list[str] = None) -> dict:
    """Remove or hash PII fields before logging."""
    pii_fields = pii_fields or ["email", "phone", "ssn", "credit_card", "name"]
    safe_trace = trace.copy()

    # Hash the actual content, log only the hash
    if "input_text" in safe_trace:
        raw = safe_trace["input_text"]
        safe_trace["input_hash"] = hashlib.sha256(raw.encode()).hexdigest()[:16]
        safe_trace["input_length"] = len(raw)
        del safe_trace["input_text"]

    # Keep full content in encrypted separate store with retention policy
    store_encrypted(trace)  # Separate encrypted store

    return safe_trace
```

**Data retention policy:**
- Raw LLM I/O: 30 days, encrypted, access-controlled
- Aggregated metrics (latency, token counts): 12 months
- Error logs without content: indefinitely
- Eval-annotated examples: indefinitely (these are valuable training data)

---

## Cost Tracking

Track cost at multiple granularities to answer "why is this feature expensive?":

```python
from collections import defaultdict

cost_by_feature = defaultdict(float)
cost_by_model = defaultdict(float)
cost_by_user = defaultdict(float)

def record_cost(feature: str, model: str, user_id: str, cost_usd: float):
    cost_by_feature[feature] += cost_usd
    cost_by_model[model] += cost_usd
    cost_by_user[user_id] += cost_usd

# Weekly report
def weekly_cost_report() -> dict:
    return {
        "top_features": sorted(cost_by_feature.items(), key=lambda x: -x[1])[:5],
        "model_breakdown": dict(cost_by_model),
        "top_users": sorted(cost_by_user.items(), key=lambda x: -x[1])[:10],
    }
```

This makes cost discussions concrete. "Feature X costs $400/day" is more actionable than "we're spending a lot on Claude."

---

## Gotchas

**Logging adds latency if done synchronously.** Write traces asynchronously. Don't let a slow logging call add 200ms to every user request. Use a background queue (Redis, SQS) for trace writes.

**You'll want logs you didn't think to capture.** When you investigate an incident, you'll discover you need the retrieved context, the system prompt version, and the user segment. Log these from day one. Storage is cheap; retroactively adding logs is painful.

**LangSmith and Langfuse send data to third parties.** If you're in a regulated industry (healthcare, finance), check whether your logs can leave your infrastructure. Self-host Langfuse or build a minimal internal tracing solution.

**Cost estimates in traces go stale.** Model pricing changes. Build your cost estimation function to be configurable, not hardcoded. A config file or database table for pricing is better than constants in code.

**Session vs trace distinction matters.** A session is a conversation (multiple turns). A trace is a single LLM call. For multi-turn applications, group traces by session ID so you can view full conversations in your dashboard.

---

> **Key Takeaways:**
> 1. Log every LLM call from day one: inputs, outputs, latency, tokens, cost. You will need this data within a month. Building it after an incident is too late.
> 2. LangSmith integrates automatically if you're using LangChain. Langfuse is the right choice for direct SDK usage or when you need self-hosting.
> 3. Trace multi-step agents with nested spans. "The agent was slow" is not a diagnosis; "the order lookup span took 6 seconds" is.
>
> *"The best time to add observability is before you need it. The second best time is right now."*

---

## Interview Questions

**Q: Your AI application started showing increased latency last week. Walk me through how you'd diagnose it.**

The first thing I'd check is whether I have latency data broken down by component. If I'm logging trace spans (which I should be), I can immediately see whether the latency increase is in the retrieval step, the LLM call, or the post-processing step. Without that breakdown, I'm guessing.

If it's LLM call latency, I'd check: did the input token count increase (longer context = slower), did we change models, did the provider have a service event? Check the provider's status page first — about 40% of unexplained latency increases are provider-side issues.

If it's retrieval latency, check whether the vector database had a config change, whether the index grew significantly, or whether a new query pattern is doing full scans instead of ANN.

If it's a specific query pattern (certain long queries are slow, others are fine), look at the slow query logs and check for context stuffing — someone might have added a new feature that dumps a large document into the prompt.

The tool I'd use: LangSmith or Langfuse trace view, filtered to the last 7 days, sorted by latency. Sample the top-10 slow traces and look for patterns.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| What is the minimum to log for every LLM call? | Model, input/output tokens, latency, cost estimate, status (success/error) |
| What is LangSmith? | LangChain's tracing and observability platform; auto-captures calls when env var is set |
| What is Langfuse? | Open-source observability platform; provider-agnostic, self-hostable |
| What is a trace span? | A timed unit of work within a larger trace; used to break down multi-step agent execution |
| Why track cost by feature, not just total? | "We spent $400 on feature X" is actionable; "we spent $400 total" is not |
| How should raw LLM I/O be stored for GDPR compliance? | Encrypted, access-controlled, with 30-day retention policy and separate from operational logs |
