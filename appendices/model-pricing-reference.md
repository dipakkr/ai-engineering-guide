# Model Pricing Reference

> Prices are approximate as of early 2026. Verify at provider pricing pages before building cost models. All prices are per 1 million tokens unless noted.

---

## Anthropic (Claude)

| Model | Input $/1M | Output $/1M | Context | Notes |
|---|---|---|---|---|
| Claude Opus 4.6 | $15 | $75 | 200K | Highest quality, extended thinking available |
| Claude Sonnet 4.6 | $3 | $15 | 200K | Best cost/quality for most tasks |
| Claude Haiku 4.5 | $0.25 | $1.25 | 200K | Classification, fast responses |
| Prompt caching (write) | $3.75 (Opus) | — | — | 25% of base input price |
| Prompt caching (read) | $0.30 (Opus) | — | — | 2% of base input price |

Cache write prices: Opus $3.75/1M, Sonnet $0.30/1M, Haiku $0.03/1M
Cache read prices: Opus $1.50/1M, Sonnet $0.30/1M, Haiku $0.03/1M

---

## OpenAI (GPT)

| Model | Input $/1M | Output $/1M | Context | Notes |
|---|---|---|---|---|
| GPT-4o | $5 | $15 | 128K | Versatile, vision, function calling |
| GPT-4o-mini | $0.15 | $0.60 | 128K | Simple tasks, very low cost |
| o1 | $15 | $60 | 200K | Extended reasoning, slow |
| o3-mini | $1.10 | $4.40 | 200K | Reasoning at lower cost |
| text-embedding-3-large | $0.13 | — | — | Per 1M tokens |
| text-embedding-3-small | $0.02 | — | — | Per 1M tokens |

Batch API: 50% discount on eligible models for async jobs.

---

## Google (Gemini)

| Model | Input $/1M | Output $/1M | Context | Notes |
|---|---|---|---|---|
| Gemini 1.5 Pro | $7 (>128K: $14) | $21 (>128K: $42) | 1M | Best long-context |
| Gemini 1.5 Flash | $0.075 (>128K: $0.15) | $0.30 (>128K: $0.60) | 1M | High-volume, fast |
| Gemini 1.5 Flash-8B | $0.0375 | $0.15 | 1M | Ultra low cost |
| text-multilingual-embedding-002 | $0.001 | — | — | Per 1K characters |

---

## Other Providers

| Model | Provider | Input $/1M | Output $/1M | Notes |
|---|---|---|---|---|
| Mistral Large 2 | Mistral | $3 | $9 | EU data residency |
| Mistral 7B Instruct | Mistral | $0.25 | $0.25 | Self-hostable |
| Command R+ | Cohere | $3 | $15 | RAG-optimized |
| Command R | Cohere | $0.50 | $1.50 | Lower cost RAG |
| Llama 3.1 405B | Fireworks AI | $3 | $3 | Best OSS via API |
| Llama 3.1 70B | Together AI | $0.88 | $0.88 | Open-source quality |
| voyage-large-2-instruct | Voyage AI | $0.12 | — | Embeddings, Anthropic recommended |

---

## Self-Hosted GPU Reference

For teams running their own inference infrastructure:

| GPU | VRAM | On-demand $/hr | Spot $/hr | Provider | Best For |
|---|---|---|---|---|---|
| A100 80GB | 80 GB | $3.50-4.00 | $1.50-2.00 | AWS, GCP, CoreWeave | 70B models, production |
| H100 80GB | 80 GB | $5.00-7.00 | $2.50-3.50 | CoreWeave, Lambda | Highest throughput |
| L40S | 48 GB | $2.50-3.00 | $1.00-1.50 | AWS g6, CoreWeave | 13B-34B models |
| A10G | 24 GB | $1.00-1.50 | $0.40-0.70 | AWS g5 | 7B-13B models |
| L4 | 24 GB | $0.50-0.80 | $0.20-0.40 | GCP g2 | 7B models, budget |
| T4 | 16 GB | $0.35-0.50 | $0.15-0.25 | AWS g4dn | Small models only |

---

## Cost Estimation Formulas

### Per-call cost
```
cost = (input_tokens / 1_000_000 × input_price) + (output_tokens / 1_000_000 × output_price)
```

### Daily cost
```
daily_cost = cost_per_call × queries_per_day
```

### With prompt caching (stable system prompt)
```
cache_write_cost = system_prompt_tokens / 1_000_000 × cache_write_price  # First call only
cache_read_cost = system_prompt_tokens / 1_000_000 × cache_read_price    # Subsequent calls
```

### Token estimates
- 1 page of English text ≈ 500 tokens
- 1,000 words ≈ 750 tokens
- 1 line of code ≈ 10-15 tokens
- 1KB of JSON ≈ 300-400 tokens

---

## When Self-Hosting Beats APIs

Monthly API cost that justifies self-hosting (assuming 1 A100 80GB at $2,500/month all-in):

| Model | Break-even monthly API spend |
|---|---|
| Claude Haiku tier equivalent | ~$5,000/month |
| Claude Sonnet tier equivalent | ~$2,500/month |
| Claude Opus tier equivalent | ~$1,000/month |

These are rough estimates — self-hosting adds engineering overhead (reliability, updates, monitoring). Factor in 0.5-1 FTE to maintain a self-hosted inference stack.

---

*Last updated: early 2026. Verify current prices at:*
- *Anthropic: console.anthropic.com/pricing*
- *OpenAI: openai.com/pricing*
- *Google: cloud.google.com/vertex-ai/pricing*
