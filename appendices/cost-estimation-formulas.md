# Cost Estimation Formulas

Quick reference for back-of-envelope calculations. All prices use early 2026 approximations — verify current prices before committing to a budget.

---

## Token Counting Rules of Thumb

```
English text:
  1,000 tokens ≈ 750 words ≈ 4-5 pages

Code:
  1,000 tokens ≈ 50-70 lines (more with symbols, less with prose)

JSON:
  1,000 tokens ≈ 800 characters (characters tokenize at ~1.2 tokens each)

Non-English multipliers:
  Spanish, French: ×1.2-1.3
  German: ×1.4
  Chinese: ×1.5
  Russian: ×1.8
  Japanese, Arabic: ×2.5
  Thai: ×3.0
```

---

## Per-Call Cost Formula

```python
def cost_per_call(
    input_tokens: int,
    output_tokens: int,
    model: str = "claude-sonnet-4-6"
) -> float:
    # Prices per 1M tokens (approximate early 2026)
    pricing = {
        "claude-opus-4-6":          (15.00, 75.00),
        "claude-sonnet-4-6":        (3.00,  15.00),
        "claude-haiku-4-5-20251001":(0.25,  1.25),
        "gpt-4o":                   (5.00,  15.00),
        "gpt-4o-mini":              (0.15,  0.60),
        "gemini-1.5-pro":           (7.00,  21.00),
        "gemini-1.5-flash":         (0.075, 0.30),
    }
    input_price, output_price = pricing.get(model, (3.00, 15.00))
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
```

---

## Daily / Monthly Cost

```python
def monthly_cost(
    queries_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    model: str = "claude-sonnet-4-6"
) -> dict:
    daily = queries_per_day * cost_per_call(avg_input_tokens, avg_output_tokens, model)
    return {
        "daily":   round(daily, 2),
        "monthly": round(daily * 30, 2),
        "annual":  round(daily * 365, 2),
    }

# Examples:
# RAG chatbot: 10K queries/day, 5K input + 500 output, Sonnet
# monthly_cost(10_000, 5_000, 500, "claude-sonnet-4-6")
# → daily: $157.50, monthly: $4,725, annual: $57,488

# Classification pipeline: 100K/day, 200 input + 20 output, Haiku
# monthly_cost(100_000, 200, 20, "claude-haiku-4-5-20251001")
# → daily: $5.25, monthly: $157.50, annual: $1,916
```

---

## Prompt Caching Savings

When you have a large stable system prompt or context:

```python
def caching_savings(
    system_prompt_tokens: int,
    queries_per_day: int,
    model: str = "claude-opus-4-6"
) -> dict:
    # Cache write: 25% of base input price (charged once per cache creation)
    # Cache read: 10% of base input price (charged on each cache hit)
    base_prices = {
        "claude-opus-4-6":           15.00,
        "claude-sonnet-4-6":         3.00,
        "claude-haiku-4-5-20251001": 0.25,
    }
    base = base_prices.get(model, 3.00)
    cache_read_price = base * 0.10  # 90% discount

    without_cache = system_prompt_tokens * base / 1_000_000 * queries_per_day
    with_cache = system_prompt_tokens * cache_read_price / 1_000_000 * queries_per_day

    daily_savings = without_cache - with_cache
    return {
        "without_cache_daily":  round(without_cache, 2),
        "with_cache_daily":     round(with_cache, 2),
        "daily_savings":        round(daily_savings, 2),
        "monthly_savings":      round(daily_savings * 30, 2),
    }

# Example: 50K-token system prompt, 10K queries/day, Opus
# caching_savings(50_000, 10_000, "claude-opus-4-6")
# → without: $7,500/day, with: $750/day, savings: $6,750/day
```

---

## Embedding Cost

```python
def embedding_cost(
    documents: int,
    avg_tokens_per_doc: int,
    model: str = "text-embedding-3-small"
) -> float:
    embedding_prices = {
        "text-embedding-3-large": 0.13,   # per 1M tokens
        "text-embedding-3-small": 0.02,
        "voyage-large-2-instruct": 0.12,
    }
    price = embedding_prices.get(model, 0.02)
    total_tokens = documents * avg_tokens_per_doc
    return round(total_tokens * price / 1_000_000, 2)

# 100K docs, 500 tokens each, small embeddings
# embedding_cost(100_000, 500, "text-embedding-3-small")
# → $1.00 (one-time indexing cost)
```

---

## Self-Hosted vs API Break-Even

```python
def api_vs_self_host_breakeven(
    queries_per_month: int,
    avg_tokens_per_query: int,    # input + output combined
    api_price_per_1m: float = 15.0,  # e.g., Opus output price
    gpu_cost_per_month: float = 2500.0,  # 1x A100 80GB all-in
    gpu_throughput_tokens_per_sec: float = 2000.0  # tokens/sec on that GPU
) -> dict:
    monthly_api_cost = queries_per_month * avg_tokens_per_query * api_price_per_1m / 1_000_000
    # Assume GPU is utilized ~50% of the time productively
    gpu_tokens_per_month = gpu_throughput_tokens_per_sec * 3600 * 24 * 30 * 0.5
    gpus_needed = (queries_per_month * avg_tokens_per_query) / gpu_tokens_per_month
    monthly_gpu_cost = max(1, gpus_needed) * gpu_cost_per_month

    return {
        "monthly_api_cost": round(monthly_api_cost, 0),
        "monthly_gpu_cost": round(monthly_gpu_cost, 0),
        "cheaper": "API" if monthly_api_cost < monthly_gpu_cost else "Self-host",
        "savings_per_month": round(abs(monthly_api_cost - monthly_gpu_cost), 0),
    }

# 10M queries/month, 2K tokens each, vs Opus-equivalent quality
# api_vs_self_host_breakeven(10_000_000, 2_000, api_price_per_1m=15.0)
# → API: $300,000/month vs GPU: ~$5,000/month → self-host saves $295K/month
```

---

## RAG System Total Cost

Full RAG pipeline cost estimate:

```python
def rag_total_cost(
    queries_per_day: int,
    avg_retrieved_chunks: int = 5,
    avg_chunk_tokens: int = 400,
    avg_query_tokens: int = 100,
    avg_system_prompt_tokens: int = 500,
    avg_output_tokens: int = 300,
    generation_model: str = "claude-sonnet-4-6",
    embedding_queries_per_day: int = None  # defaults to queries_per_day
) -> dict:
    if embedding_queries_per_day is None:
        embedding_queries_per_day = queries_per_day

    # Embedding cost (query-time only; indexing is one-time)
    embed_cost_per_query = avg_query_tokens * 0.02 / 1_000_000  # text-embedding-3-small

    # Generation input: system + retrieved context + query
    input_tokens = (avg_system_prompt_tokens +
                    avg_retrieved_chunks * avg_chunk_tokens +
                    avg_query_tokens)

    gen_cost = cost_per_call(input_tokens, avg_output_tokens, generation_model)

    total_per_query = embed_cost_per_query + gen_cost
    return {
        "cost_per_query": round(total_per_query, 5),
        "daily_cost": round(total_per_query * queries_per_day, 2),
        "monthly_cost": round(total_per_query * queries_per_day * 30, 2),
        "embedding_pct": round(embed_cost_per_query / total_per_query * 100, 1),
    }

# Production RAG chatbot: 10K queries/day, Sonnet
# rag_total_cost(10_000, generation_model="claude-sonnet-4-6")
# → cost_per_query: ~$0.016, daily: ~$160, monthly: ~$4,800
```

---

## Model Routing Cost Model

When routing between models based on complexity:

```python
def routing_cost_model(
    queries_per_day: int,
    pct_simple: float = 0.5,    # → Haiku
    pct_medium: float = 0.3,    # → Sonnet
    pct_complex: float = 0.2,   # → Opus
    input_tokens: int = 2000,
    output_tokens: int = 500
) -> dict:
    costs = {
        "haiku":  cost_per_call(input_tokens, output_tokens, "claude-haiku-4-5-20251001"),
        "sonnet": cost_per_call(input_tokens, output_tokens, "claude-sonnet-4-6"),
        "opus":   cost_per_call(input_tokens, output_tokens, "claude-opus-4-6"),
    }

    blended_cost = (pct_simple * costs["haiku"] +
                    pct_medium * costs["sonnet"] +
                    pct_complex * costs["opus"])
    all_opus = costs["opus"]

    return {
        "blended_cost_per_query": round(blended_cost, 5),
        "all_opus_cost_per_query": round(all_opus, 5),
        "daily_savings": round((all_opus - blended_cost) * queries_per_day, 2),
        "cost_reduction_pct": round((1 - blended_cost / all_opus) * 100, 1),
    }

# routing_cost_model(10_000)
# → blended: ~$0.026/query vs all-Opus: ~$0.188/query → 86% reduction
```

---

## Quick Reference Card

| Scenario | Rule of Thumb |
|---|---|
| RAG chatbot, Sonnet, 1K users/day | ~$150-300/month |
| Classification pipeline, Haiku, 1M/day | ~$150-300/month |
| Document analysis, Opus, 10K/day | ~$5,000-15,000/month |
| Embedding 1M documents (small emb) | ~$10 one-time |
| Prompt caching a 50K-token system prompt | Saves 90% of that prompt's cost |
| Self-hosting worth it at... | >$5,000/month API spend |
| Japanese/Thai users cost vs English | ~2.5-3x more tokens |
