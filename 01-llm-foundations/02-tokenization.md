# Tokenization

> **TL;DR**: Tokens are the atomic units of LLM processing, roughly 0.75 words in English. They're money: every token in, every token out costs you. English is efficient (1 token ≈ 1 word). Code, JSON, and non-Latin languages are expensive (1 token ≈ 0.25-0.5 characters). Use tiktoken to count tokens before sending.

**Prerequisites**: [Transformer Intuition](01-transformer-intuition.md)
**Related**: [Context Windows](04-context-windows.md), [Cost Optimization](../06-production-and-ops/07-cost-optimization.md), [Context Engineering](../02-prompt-engineering/02-context-engineering.md)

---

## What Is a Token?

A token is the smallest unit a language model processes. It's not a word, not a character — it's somewhere in between.

Modern LLMs use Byte Pair Encoding (BPE), a data-driven tokenization that merges common character sequences into single tokens. The tokenizer learns from the training corpus which character combinations appear frequently enough to merit their own token.

**Common English tokenizations:**
```
"Hello" → ["Hello"]          (1 token)
"running" → ["running"]      (1 token)
"unbelievable" → ["unbel", "ievable"]  (2 tokens)
"tokenization" → ["token", "ization"] (2 tokens)
"!@#$%" → ["!", "@", "#", "$", "%"]   (5 tokens)
```

The rough rule: 1 token ≈ 4 characters ≈ 0.75 words in English. 750 words ≈ 1,000 tokens.

---

## Token Counting in Practice

Use [tiktoken](https://github.com/openai/tiktoken) to count tokens accurately before sending:

```python
import tiktoken

# Get the tokenizer for a specific model
enc = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def estimate_cost(text: str, model: str = "claude-opus-4-6") -> float:
    """Estimate input token cost."""
    n_tokens = count_tokens(text)
    price_per_1k = {
        "claude-opus-4-6": 0.015,
        "claude-sonnet-4-6": 0.003,
        "claude-haiku-4-5-20251001": 0.00025,
    }
    return n_tokens * price_per_1k.get(model, 0.003) / 1000

# Examples
print(count_tokens("The quick brown fox jumps over the lazy dog."))  # ~10 tokens
print(count_tokens("The quick brown fox jumps over the lazy dog." * 100))  # ~1000 tokens

# Real usage: check if context fits before API call
MAX_CONTEXT = 180_000  # Leave 20K for response
if count_tokens(system_prompt + user_message) > MAX_CONTEXT:
    # Truncate or compress before calling API
    pass
```

Anthropic uses a different tokenizer than OpenAI, but for estimation purposes, tiktoken's counts are within 5-10% for Claude models.

---

## Why Non-English Languages Are Expensive

This is a practical consequence of BPE tokenization: the vocabulary is trained primarily on English and common programming languages. Other languages have less efficient tokenizations.

**Tokens per word comparison:**

| Language | Avg tokens per word | Relative cost |
|---|---|---|
| English | 1.0 | 1x |
| Spanish | 1.2 | 1.2x |
| French | 1.3 | 1.3x |
| German | 1.4 | 1.4x |
| Russian | 1.8 | 1.8x |
| Japanese | 2.5 | 2.5x |
| Chinese | 1.5 | 1.5x |
| Arabic | 2.5 | 2.5x |
| Thai | 3.0 | 3.0x |

The reason: Japanese characters often map to 2-3 tokens each because BPE didn't see enough Japanese to learn single-token representations for common kanji. Same text in Japanese costs 2-3x more than the English equivalent.

**Example:**
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

english = "Hello, how are you today?"
japanese = "こんにちは、今日のお調子はいかがですか？"  # Same meaning

print(f"English: {len(enc.encode(english))} tokens")   # ~6 tokens
print(f"Japanese: {len(enc.encode(japanese))} tokens") # ~18 tokens
```

**Implication for multilingual products:** Budget 2-3x token costs for Japanese, Arabic, and Thai support compared to English. Monitor token usage by language in production.

---

## Why Code Is Token-Inefficient

Code has special characters, indentation, and structure that tokenize poorly:

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

prose = "The function takes a list of integers and returns their sum."
code = "def sum_integers(nums: list[int]) -> int: return sum(nums)"

print(f"Prose: {len(enc.encode(prose))} tokens")  # ~12 tokens
print(f"Code: {len(enc.encode(code))} tokens")    # ~18 tokens
```

JSON is especially bad. Those curly braces, quotes, and colons are each separate tokens:

```python
import json

data = {"name": "John", "age": 30, "city": "New York"}

json_str = json.dumps(data)
natural = f"Person named John, age 30, lives in New York"

print(f"JSON: {len(enc.encode(json_str))} tokens")    # ~18 tokens
print(f"Natural: {len(enc.encode(natural))} tokens")  # ~12 tokens
```

**Implication:** When injecting structured data into prompts, consider whether natural language descriptions are cheaper than JSON. For large datasets, the savings are significant.

---

## Tokenization Edge Cases

**Whitespace matters:**
```python
# Leading spaces create different tokens
enc.encode("hello")        # ['hello'] — 1 token
enc.encode(" hello")       # [' hello'] — 1 token (different token)
enc.encode("  hello")      # [' ', ' hello'] — 2 tokens
```

This is why consistent formatting in prompts matters. Inconsistent spacing can add tokens unnecessarily.

**Numbers tokenize unpredictably:**
```python
enc.encode("42")     # ['42']   — 1 token
enc.encode("1234")   # ['1234'] — 1 token
enc.encode("12345")  # ['123', '45'] — 2 tokens
enc.encode("123456") # ['123', '456'] — 2 tokens
```

Large numbers split in non-obvious ways. This is why LLMs sometimes make arithmetic errors: "12345 + 67890" doesn't operate on whole numbers; it operates on the tokens ["123", "45", "+", "678", "90"], which isn't how humans do arithmetic.

**Special tokens:**
```python
# Start/end of text tokens are implicit
# Some models use explicit separator tokens between system/user/assistant
# These add to your token count even though you don't write them
```

When counting tokens for budget purposes, include 3-5% buffer for special tokens that the API adds automatically.

---

## Practical Token Budget Management

```python
from anthropic import Anthropic

client = Anthropic()

def safe_api_call(system: str, messages: list[dict], model: str = "claude-opus-4-6",
                  max_response_tokens: int = 1024) -> str:
    """Make API call with token budget validation."""

    # Exact token count via API (most accurate)
    count_response = client.messages.count_tokens(
        model=model,
        system=system,
        messages=messages
    )
    input_tokens = count_response.input_tokens

    # Model context limits (as of 2025)
    context_limits = {
        "claude-opus-4-6": 200_000,
        "claude-sonnet-4-6": 200_000,
        "claude-haiku-4-5-20251001": 200_000,
    }

    limit = context_limits.get(model, 200_000)
    available = limit - input_tokens

    if available < max_response_tokens:
        raise ValueError(
            f"Not enough context: {input_tokens} input tokens + {max_response_tokens} "
            f"response tokens exceeds {limit} limit"
        )

    return client.messages.create(
        model=model,
        system=system,
        messages=messages,
        max_tokens=min(max_response_tokens, available)
    ).content[0].text
```

---

## The Vocabulary Size Tradeoff

Modern LLMs have vocabularies of 32K-200K tokens. Larger vocabulary = more efficient tokenization (common phrases as single tokens) but larger embedding tables.

| Model Family | Vocabulary Size |
|---|---|
| GPT-2 | 50,257 |
| GPT-3/3.5/4 | 100,256 (cl100k_base) |
| Claude (estimated) | 100K+ |
| Llama 3 | 128,256 |
| Gemma | 256,000 |

Llama 3's move to 128K vocabulary (vs Llama 2's 32K) specifically improved non-English efficiency. Gemma 2's 256K vocabulary further improves multilingual and code tokenization.

---

## Gotchas

**Token counting before the API call is an estimate.** The exact token count from the API might differ by 1-3% due to special tokens added automatically. Always leave headroom when setting context limits.

**Prompt compression is real.** Replacing verbose instructions with concise ones saves tokens and often improves performance (less noise for the model to parse). Audit prompts longer than 500 words for redundancy.

**Repeating content in the prompt is expensive.** If your prompt repeats the same context section twice for emphasis, you're paying twice. Use explicit instructions instead: "The following is the most critical constraint, pay close attention:"

**Token limits include system prompt.** The context window limit applies to the sum of system prompt + conversation history + retrieved context + user message + response. Teams often forget that a 2000-token system prompt plus 3 rounds of history can consume 20% of the budget before the user even asks their question.

---

> **Key Takeaways:**
> 1. Tokens are money. Count them before sending. Use tiktoken for estimates; use the API's count_tokens endpoint for exactness.
> 2. Non-English languages cost 1.5-3x more than English for the same semantic content. Factor this into international product pricing and budgets.
> 3. Code and JSON tokenize inefficiently. Consider natural language descriptions for structured data when the token savings matter at scale.
>
> *"The fastest way to waste money on LLMs is to send tokens you don't need. Count before you send."*

---

## Interview Questions

**Q: Your international expansion requires supporting Japanese users. The product manager wants to know if costs will increase significantly. How do you answer?**

Yes, costs will increase significantly. Japanese text tokenizes at roughly 2.5x the rate of English — the same sentence costs about 2.5x as many tokens. This applies to both user messages coming in and model responses going out.

For input, there's not much you can do — that's the user's message. For output, if your application generates responses and then translates, you're paying for translation tokens on top. It's better to generate directly in Japanese if the model supports it (Claude and GPT-4 do), rather than translate from English.

My estimate: Japanese users will cost roughly 2-2.5x what English users cost. Build this into the pricing model for the Japanese market. At scale, you can partially offset this with semantic caching (Japanese users asking the same FAQ questions) and prompt caching (stable Japanese system prompts).

One thing to measure in production: compare the actual token-per-query ratio for Japanese vs English users against this estimate. Language models sometimes generate wordier responses in certain languages.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| What is a token? | The atomic unit of LLM processing; roughly 4 characters or 0.75 English words |
| What is BPE? | Byte Pair Encoding: tokenization that learns to merge common character sequences into single tokens |
| Why is Japanese expensive in tokens? | BPE was trained primarily on English; Japanese characters often map to multiple tokens |
| How many tokens is 750 words? | About 1,000 tokens |
| What tool counts tokens for OpenAI models? | tiktoken (also a good approximation for Claude models) |
| Why do numbers tokenize oddly? | BPE doesn't know arithmetic; large numbers split at arbitrary character boundaries |
