# Cost Optimization

> **TL;DR**: LLM costs have five levers in order of ROI: caching (free savings, 20-50% reduction), model routing (use cheaper models for simple tasks, 30-60% reduction), prompt caching (90% on stable prefixes), token reduction (trim prompts, 10-30%), and batch processing (non-real-time jobs at 50% discount). Attack them in that order.

**Prerequisites**: [Context Engineering](../02-prompt-engineering/02-context-engineering.md), [Caching Strategies](03-caching-strategies.md)
**Related**: [Inference Infrastructure](04-inference-infrastructure.md), [Observability and Tracing](01-observability-and-tracing.md)

---

## The Cost Breakdown

Before optimizing, understand where money goes. For a typical production RAG application:

```
Cost breakdown (example: 100K queries/day, Claude Opus):

Input tokens (300 avg/query):
  System prompt:     50 tokens  × 100K × $0.015/1K = $75/day
  RAG context:      200 tokens  × 100K × $0.015/1K = $300/day
  User query:        50 tokens  × 100K × $0.015/1K = $75/day

Output tokens (150 avg/query):
  Response:         150 tokens  × 100K × $0.075/1K = $1,125/day

Total:                                                $1,575/day = $47,250/month
```

The output tokens cost 5x more per token than input tokens. Short responses are dramatically cheaper than long ones. This is the first place I look: is the response longer than it needs to be?

---

## Lever 1: Caching

Covered in [Caching Strategies](03-caching-strategies.md). Summary of what to implement first:

| Cache Type | Implementation Effort | Typical Savings |
|---|---|---|
| Exact cache (Redis) | 2 hours | 10-20% of total calls |
| Prompt caching (stable prefix) | 1 hour | 60-90% of input token cost for cached portion |
| Semantic cache | 1 day | 5-15% of exact cache misses |

Implement prompt caching on the system prompt and static context before any other optimization. If your system prompt is 500 tokens and it's sent on every request, that's:
- Without caching: 500 × 100K × $0.015/1K = $750/day
- With caching: 500 × 100K × $0.0015/1K (cache read) = $75/day
- **Savings: $675/day for 1 hour of work**

---

## Lever 2: Model Routing

Not every task needs the most powerful model. Route simple tasks to cheaper models:

```python
from anthropic import Anthropic

client = Anthropic()

# Model pricing (as of early 2025, per 1M tokens input/output)
MODEL_PRICING = {
    "claude-opus-4-6":           (15.00, 75.00),
    "claude-sonnet-4-6":          (3.00, 15.00),
    "claude-haiku-4-5-20251001":  (0.25,  1.25),
}

def route_to_model(task_type: str, complexity: str) -> str:
    """Route to cheapest model that can handle the task."""
    routing_table = {
        # Simple classification: Haiku
        ("classification", "simple"):     "claude-haiku-4-5-20251001",
        ("classification", "complex"):    "claude-haiku-4-5-20251001",

        # Extraction: Haiku or Sonnet
        ("extraction", "simple"):         "claude-haiku-4-5-20251001",
        ("extraction", "complex"):        "claude-sonnet-4-6",

        # Generation: Sonnet default, Opus for critical
        ("generation", "simple"):         "claude-sonnet-4-6",
        ("generation", "complex"):        "claude-sonnet-4-6",
        ("generation", "critical"):       "claude-opus-4-6",

        # Reasoning: Sonnet or Opus
        ("reasoning", "simple"):          "claude-sonnet-4-6",
        ("reasoning", "complex"):         "claude-opus-4-6",
    }
    return routing_table.get((task_type, complexity), "claude-sonnet-4-6")

def classify_task_complexity(user_query: str) -> tuple[str, str]:
    """Fast classification of task type and complexity."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Use cheapest model to classify
        max_tokens=30,
        messages=[{"role": "user", "content":
            f"Classify this query. Return: task_type (classification/extraction/generation/reasoning) "
            f"and complexity (simple/complex/critical). Format: 'task_type,complexity'\n\n{user_query}"}]
    )
    try:
        parts = response.content[0].text.strip().split(",")
        return parts[0].strip(), parts[1].strip()
    except (ValueError, IndexError):
        return "generation", "complex"  # Default to safe routing
```

**ROI of model routing:**

Replacing 50% of Opus calls with Haiku for simple tasks:
- Before: 100K calls × $0.015/1K input = $1,500/day
- After: 50K Opus + 50K Haiku = $750 + $12.50 = $762.50/day
- **Savings: 49%**

---

## Lever 3: Token Reduction

Trim prompts aggressively. Every unnecessary token costs money.

**Audit your prompts for token waste:**

```python
def audit_prompt_tokens(prompt: str) -> dict:
    """Identify token-heavy patterns in a prompt."""
    issues = []

    # Check for verbose instructions
    if len(prompt.split()) > 500:
        issues.append("Prompt is long (>500 words). Consider condensing.")

    # Check for repeated instructions
    sentences = prompt.split(". ")
    if len(sentences) != len(set(sentences)):
        issues.append("Duplicate sentences found. Remove redundancy.")

    # Rough token estimate
    estimated_tokens = len(prompt) // 4

    return {
        "word_count": len(prompt.split()),
        "estimated_tokens": estimated_tokens,
        "daily_cost_at_100k_queries": estimated_tokens * 100_000 * 0.015 / 1_000,
        "issues": issues
    }

# Example: 1000-token system prompt at 100K/day
# Cost: 1000 × 100K × $0.015/1K = $1,500/day
# Trimmed to 500 tokens: $750/day
# Savings: $750/day for half a day of prompt engineering
```

**Common prompt bloat sources:**
- Verbose role descriptions ("You are an extremely helpful, highly knowledgeable, experienced customer service representative for Acme Corp...")
- Repeated constraints ("Always be professional. Maintain a professional tone. Respond professionally.")
- Excessive caveats ("Note that this information may be outdated. Please verify independently. The following is for informational purposes only...")
- Unnecessary few-shot examples that aren't improving quality

---

## Lever 4: Response Length Control

Output tokens cost 5x more than input. Control response length aggressively:

```python
def add_length_constraints(system_prompt: str, use_case: str) -> str:
    constraints = {
        "faq": "Answer in 1-3 sentences. Be direct.",
        "analysis": "Limit to 200 words. Use bullet points.",
        "code": "Return only the code. No explanation unless asked.",
        "chat": "Keep responses under 100 words unless the question requires more.",
    }
    constraint = constraints.get(use_case, "Be concise.")
    return system_prompt + f"\n\n{constraint}"

# Also: set max_tokens appropriately
# FAQ bot: max_tokens=200 (most answers fit in 150 tokens)
# Code assistant: max_tokens=2000
# Document analysis: max_tokens=1000
```

The `max_tokens` parameter is not just a safety limit — it's a cost control. Set it to the 95th percentile of what you actually need, not the theoretical maximum.

---

## Lever 5: Batch Processing

For non-real-time workloads, the Anthropic Batch API offers 50% cost reduction and removes rate limit constraints:

```python
import anthropic
import json

client = anthropic.Anthropic()

def batch_process(items: list[dict]) -> list[dict]:
    """Process items using Message Batches API at 50% cost."""

    # Create batch request
    requests = [
        {
            "custom_id": f"item-{i}",
            "params": {
                "model": "claude-opus-4-6",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": item["prompt"]}]
            }
        }
        for i, item in enumerate(items)
    ]

    batch = client.messages.batches.create(requests=requests)
    print(f"Batch created: {batch.id}")

    # Poll for completion (or use webhook)
    import time
    while True:
        status = client.messages.batches.retrieve(batch.id)
        if status.processing_status == "ended":
            break
        time.sleep(30)
        print(f"Processing... {status.request_counts}")

    # Retrieve results
    results = []
    for result in client.messages.batches.results(batch.id):
        if result.result.type == "succeeded":
            results.append({
                "id": result.custom_id,
                "response": result.result.message.content[0].text
            })
    return results

# Use cases for batch API:
# - Document processing pipelines
# - Nightly report generation
# - Training data generation
# - Offline evaluation runs
# NOT for: real-time user requests (batch processing takes minutes to hours)
```

---

## The Worked Cost Model

Let me show what a full optimization looks like on a real-world example.

**Starting state: E-commerce support bot, 50K queries/day**

```
Current costs:
- Model: claude-opus-4-6
- System prompt: 800 tokens
- RAG context: 1,500 tokens (top-5 chunks)
- Average output: 250 tokens
- No caching

Input cost: (800 + 1,500) × 50K × $0.015/1K = $1,725/day
Output cost: 250 × 50K × $0.075/1K = $937/day
Total: $2,662/day = $79,860/month
```

**After optimization:**

```
Changes made:
1. Prompt caching on system prompt (800 tokens, same for all users)
   System prompt cost: 800 × 50K × $0.0015/1K = $60/day (was $600/day)
   Savings: $540/day

2. Reduce RAG context from top-5 to top-3 chunks (tested, quality same)
   Context tokens: 1,500 → 900 tokens
   Context cost: 900 × 50K × $0.015/1K = $675/day (was $1,125/day)
   Savings: $450/day

3. Route 60% of queries to Sonnet (FAQ questions, simple lookups)
   40% Opus input: 900 tokens × 20K × $0.015/1K = $270/day
   60% Sonnet input: 900 tokens × 30K × $0.003/1K = $81/day
   40% Opus output: 250 × 20K × $0.075/1K = $375/day
   60% Sonnet output: 250 × 30K × $0.015/1K = $112.50/day
   Total generation: $838.50/day (was $1,725/day)
   Savings: $886.50/day

4. Reduce response length constraint (added "3 sentences max for FAQs")
   Average output: 250 → 180 tokens
   Output cost (new): $112.50 × (180/250) = ~$81/day Sonnet, ~$270/day Opus
   Additional savings: ~$136/day

Optimized total: $60 + $675 + $838.50 + savings = ~$1,500/day
Original: $2,662/day
Savings: 44% = $35,000/month
```

This is not theoretical. These are the categories of savings I've seen in production systems.

---

## Monitoring Cost in Real Time

Set up cost alerts before you need them:

```python
from datetime import datetime
import boto3

def calculate_daily_cost(traces: list[dict]) -> float:
    PRICING = {
        "claude-opus-4-6":          (0.015, 0.075),
        "claude-sonnet-4-6":         (0.003, 0.015),
        "claude-haiku-4-5-20251001": (0.00025, 0.00125),
    }

    total = 0.0
    for trace in traces:
        model = trace.get("model", "claude-opus-4-6")
        input_p, output_p = PRICING.get(model, (0.015, 0.075))
        total += (trace["input_tokens"] * input_p +
                  trace["output_tokens"] * output_p) / 1000
    return total

def cost_alert_check(daily_cost: float, budget: float = 2000.0):
    if daily_cost > budget * 1.5:
        send_alert(f"CRITICAL: Daily cost ${daily_cost:.0f} is 50% over budget (${budget:.0f})")
    elif daily_cost > budget:
        send_alert(f"WARNING: Daily cost ${daily_cost:.0f} exceeds budget (${budget:.0f})")
```

---

## Gotchas

**Model routing misclassifications cost more than routing correctly.** If your complexity classifier sends a complex task to Haiku and it fails, you pay for both the Haiku failure and the Opus retry. Test your classifier accuracy on a sample before relying on it.

**Token reduction can hurt quality.** Reducing from top-5 to top-3 RAG chunks saves 40% of context tokens but might drop recall by 10%. Always measure quality after token reduction, not just before.

**Batch API latency is minutes to hours, not seconds.** The 50% cost savings is compelling, but batch processing is asynchronous. Never use it for user-facing real-time requests.

**Cache hit rate depends on query patterns.** If your application has highly unique queries (each user asks something different), exact cache hit rate will be low. Measure cache hit rates before architecting around them.

**Output pricing varies dramatically.** Claude Opus output is $0.075/1K tokens; input is $0.015/1K. Long responses cost 5x more per token. Systems that generate long reports or documents have output costs dominating. For these, response length control is the highest-ROI optimization.

---

> **Key Takeaways:**
> 1. Five levers in ROI order: caching (implement first, highest ROI for minimal effort), model routing (30-60% savings), prompt caching (90% on stable prefixes), token reduction (10-30%), batch processing (50% discount for offline jobs).
> 2. Output tokens cost 5x more than input. For applications generating long text (reports, analysis), response length control is the highest-ROI optimization.
> 3. Measure before optimizing. Build cost tracking by model and feature first, then attack the biggest line items.
>
> *"The most expensive LLM call is the one that generates tokens you didn't need."*

---

## Interview Questions

**Q: Your AI feature costs $50K/month in API calls. The business wants to cut it to $20K without significant quality loss. How?**

I'd start with a cost breakdown before proposing changes. Where is the money going? Is it a few expensive features driving most cost, or is it evenly distributed? What's the current model mix — is everything going to Opus when some tasks could use Haiku?

With a breakdown in hand, I'd apply optimizations in order of ROI. First: prompt caching. If the system prompt and any static context are sent on every request without caching, that's low-hanging fruit. For a system prompt at 1000 tokens, caching alone might save $10-15K/month at this scale.

Second: model routing. I'd audit what tasks we're using Opus for. Classifying support tickets into categories? Haiku handles that at 1/60th the cost. Generating short FAQ answers? Sonnet is fine. I'd estimate 40-50% of calls could route to cheaper models without quality loss.

Third: reduce context size. Most RAG systems retrieve top-5 chunks when top-3 would do. That's a 40% reduction in retrieval context tokens. I'd measure quality at top-3 vs top-5 on the eval set; if there's less than 2% quality difference, reduce.

Together, these three changes typically get to 40-60% cost reduction: from $50K to $20-30K. The quality measurement at each step is non-negotiable — I'm not cutting costs by degrading the product, I'm cutting waste.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| What is the ROI order for cost optimization levers? | Caching → model routing → prompt caching → token reduction → batch processing |
| How much does output cost vs input (Anthropic)? | Output is ~5x more expensive per token than input |
| What discount does the Batch API provide? | 50% cost reduction for async, non-real-time workloads |
| What is model routing? | Sending simple tasks to cheaper models (Haiku) and complex tasks to more capable models (Opus) |
| What is the main risk of aggressive token reduction? | Quality degradation from less context; always measure quality after reducing tokens |
| How do you set max_tokens as a cost control? | Set to the 95th percentile of what you actually need, not the theoretical maximum |
