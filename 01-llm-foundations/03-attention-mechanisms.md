# Attention Mechanisms

> **TL;DR**: Attention lets each token look at all other tokens and weight their influence on its representation. Multi-head attention runs this in parallel from multiple "perspectives." The KV cache stores computed key-value pairs to avoid recomputation during generation. FlashAttention is the engineering trick that makes long-context models practical. You need this to understand context window limits and inference costs.

**Prerequisites**: [Transformer Intuition](01-transformer-intuition.md), [Tokenization](02-tokenization.md)
**Related**: [Context Windows](04-context-windows.md), [Inference Infrastructure](../06-production-and-ops/04-inference-infrastructure.md)

---

## Self-Attention: The Core Mechanism

The intuition: every token in the sequence gets to "ask" about every other token, and the answers determine how the token's representation is updated.

For each token, attention computes three things:
- **Query (Q):** "What kind of information am I looking for?"
- **Key (K):** "What information do I advertise about myself?"
- **Value (V):** "What information do I actually transmit?"

The mechanism:
1. Compute dot product of Q with all Ks: "How relevant is each token to my query?"
2. Scale by sqrt(d_k) to prevent gradient issues with large dot products
3. Softmax to get a probability distribution: "How much weight to give each token?"
4. Multiply weights by V: "Collect the weighted sum of all values"

The result: each token gets a new representation that's a weighted combination of all other tokens' values, weighted by how relevant they were to the query.

**Why it works:** "Bank" with a high dot product against "river"'s key updates its representation using "river"'s value information. The result captures the "geographical bank" meaning, not the "financial bank" meaning.

---

## Multi-Head Attention: Different Types of Relationships

A single attention operation looks for one type of relationship. Multi-head attention runs several attention operations in parallel:

```python
import torch
import torch.nn.functional as F

def multi_head_attention(Q, K, V, n_heads: int = 8):
    """Simplified multi-head attention."""
    d_model = Q.shape[-1]
    d_head = d_model // n_heads

    # Split Q, K, V into n_heads each
    # Shape: (batch, seq_len, n_heads, d_head)
    Q_heads = Q.view(*Q.shape[:-1], n_heads, d_head)
    K_heads = K.view(*K.shape[:-1], n_heads, d_head)
    V_heads = V.view(*V.shape[:-1], n_heads, d_head)

    # Compute attention for each head in parallel
    scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / (d_head ** 0.5)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, V_heads)

    # Concatenate heads and project back
    output = output.view(*output.shape[:-2], d_model)
    return output
```

Different heads learn to capture different relationship types:
- One head might learn syntactic dependencies (subject-verb agreement)
- Another might track coreferences ("she" → who "she" refers to)
- Another might capture semantic relationships (synonyms, antonyms)
- Another might focus on nearby tokens (local context)

A 70B parameter model typically has 64 attention heads. Each sees the full sequence and contributes a different "view" of the relationships.

---

## The KV Cache: Why Generation Is Fast

During text generation, the model produces one token at a time. Naively, generating token N would require recomputing attention over all N-1 previous tokens. That's O(N²) work per token.

The KV cache avoids this: after computing attention for each token, store the K and V tensors. When generating the next token, only compute Q for the new token and attend to the cached K and V from previous tokens.

```python
class TransformerWithKVCache:
    def __init__(self):
        self.kv_cache = {}  # layer_id -> (K_cached, V_cached)

    def generate_next_token(self, new_token_embedding, layer_id: int):
        """Generate next token using KV cache."""
        Q = self.compute_query(new_token_embedding)
        K_new = self.compute_key(new_token_embedding)
        V_new = self.compute_value(new_token_embedding)

        if layer_id in self.kv_cache:
            K_cached, V_cached = self.kv_cache[layer_id]
            # Append new K, V to cache
            K = torch.cat([K_cached, K_new], dim=1)  # Seq dimension
            V = torch.cat([V_cached, V_new], dim=1)
        else:
            K, V = K_new, V_new

        self.kv_cache[layer_id] = (K, V)

        # Attend: new token attends to all cached positions
        return self.attention(Q, K, V)
```

**The catch:** KV cache memory grows linearly with sequence length. For a 70B model with 128K context:

```
KV cache memory = 2 × n_layers × n_heads × d_head × seq_len × dtype_bytes
= 2 × 80 × 64 × 128 × 128,000 × 2 bytes  (FP16)
= 2 × 80 × 64 × 128 × 128,000 × 2
≈ 42 GB
```

A 70B model's weights are ~140 GB. At maximum context (128K), the KV cache adds another 42 GB. You need 180+ GB total — 3+ A100 80GB GPUs.

---

## The Quadratic Scaling Problem

Standard self-attention scales as O(n²) in both time and memory with sequence length n. Every token attends to every other token — that's n² attention scores.

For long sequences:
- 1K tokens: 1M attention scores (manageable)
- 10K tokens: 100M attention scores (getting expensive)
- 100K tokens: 10B attention scores (very expensive)
- 1M tokens: 1T attention scores (impractical without optimizations)

**FlashAttention** ([Dao et al. 2022](https://arxiv.org/abs/2205.14135)) is the key engineering innovation. It doesn't change the mathematical result of attention — it changes how the computation is done on GPU hardware.

Traditional attention: compute the full n×n attention matrix, then multiply by V. This requires reading/writing the full matrix to/from GPU memory.

FlashAttention: tile the computation to fit in GPU SRAM (on-chip memory), process in blocks, accumulate the result without ever materializing the full n×n matrix in slower HBM memory.

```
Standard attention:  3 HBM accesses per element of attention matrix
FlashAttention:      much fewer HBM accesses (tiled computation in SRAM)
Speedup:             2-4x for typical sequence lengths
Memory:              O(n) instead of O(n²) memory footprint
```

FlashAttention is now standard in all major implementations. It's what makes 200K-context Claude models practical to run.

---

## Grouped Query Attention (GQA)

Standard multi-head attention has separate K and V matrices for each head. This is expensive in the KV cache.

**Grouped Query Attention** ([Ainslie et al. 2023](https://arxiv.org/abs/2305.13245)) groups multiple query heads to share a single K/V head:

```
Standard MHA:  32 Q heads, 32 K heads, 32 V heads  → large KV cache
GQA:           32 Q heads, 8 K heads, 8 V heads    → 4x smaller KV cache
MQA:           32 Q heads, 1 K head,  1 V head     → 32x smaller KV cache
```

Llama 3 uses GQA with 8 K/V heads. This reduces the KV cache by 4x compared to standard MHA with minimal quality loss — critical for enabling long-context inference on fewer GPUs.

---

## Positional Encoding: How the Model Knows Order

Attention doesn't inherently capture position — the mechanism is permutation-invariant. To know that "cat" comes before "sat," the model needs positional information explicitly added.

**Rotary Position Embedding (RoPE)** is the current standard. Instead of adding a position vector, it rotates the Q and K vectors by an angle that depends on their position. When you take the dot product Q·K, the relative position information is naturally encoded in the result.

**Why RoPE is better:**
- **Relative positions generalize better:** The model learns about "token X is 5 positions before token Y" rather than "token is at absolute position 42"
- **Length extrapolation:** Models can sometimes generalize to longer contexts than they saw in training (with techniques like YaRN, RoPE scaling)
- **No positional vector to store:** The rotation is computed on the fly, not a learned lookup table

---

## Attention Visualization

You can inspect what the model is attending to with tools like [BertViz](https://github.com/jessevig/bertviz):

```python
from transformers import AutoTokenizer, AutoModel
from bertviz import head_view

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

inputs = tokenizer("The animal didn't cross the street because it was too tired", return_tensors="pt")
outputs = model(**inputs)
attention = outputs.attentions  # Tuple: one tensor per layer

# Visualize which tokens "it" attends to
# You'll see "it" strongly attending to "animal" — coreference resolution
head_view(attention, tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
```

Attention visualization is useful for debugging: if a model is making a wrong inference, looking at what it's attending to can reveal whether the issue is in attention patterns or somewhere else.

---

## Gotchas

**Long context doesn't mean equally good recall at all positions.** Even with FlashAttention and 200K context, the model's effective use of information at the start of a very long context is weaker than at the end. The lost-in-middle problem is real. Position matters.

**KV cache VRAM limits effective batch size.** If you have 80GB VRAM and the model weights take 35GB, you have 45GB for KV cache. At 200K context, the KV cache per sequence is ~42GB. That means you can only serve ONE request at maximum context. For high-throughput serving, you trade context length for batch size.

**Attention is not interpretable in a simple sense.** High attention weight from token A to token B doesn't necessarily mean A "uses" B for its prediction. Research on attention interpretability ([Jain & Wallace 2019](https://arxiv.org/abs/1902.10186)) shows attention weights don't always correspond to feature importance. Don't over-interpret attention visualizations.

---

> **Key Takeaways:**
> 1. The KV cache is why generation is fast but why long-context inference is VRAM-hungry. Each token caches K and V tensors for all layers; at 128K context, this adds 40+ GB to VRAM requirements.
> 2. FlashAttention makes long-context practical by tiling computations in fast SRAM instead of materializing the full n×n attention matrix in slower HBM. It's a memory access optimization, not a mathematical change.
> 3. Grouped Query Attention (used in Llama 3, Mistral) reduces KV cache by 4x with minimal quality loss, enabling longer effective contexts on the same hardware.
>
> *"The KV cache is the hidden constraint. When someone asks 'why can't I run 100 parallel sessions at 200K context?', the answer is always KV cache memory."*

---

## Interview Questions

**Q: Why does serving LLMs at long context require so much more GPU memory than short context?**

The KV cache is the primary reason. During generation, the model stores the key and value tensors for every previously processed token. For a 70B parameter model, each transformer layer has K and V matrices with dimensions proportional to the sequence length. At 128K context, the KV cache alone requires ~40-50 GB of VRAM.

This means that for long-context serving, the VRAM is split between: model weights (fixed, ~140 GB for 70B FP16), KV cache (grows linearly with sequence length and number of concurrent sessions), and working memory for the current computation.

For high-throughput inference at long context, you're often constrained by KV cache memory rather than compute. Techniques like Grouped Query Attention (4x smaller KV cache) and KV cache quantization (store at INT8 instead of FP16) help, but the fundamental constraint remains.

vLLM's PagedAttention specifically addresses this by managing KV cache like virtual memory — allocating pages on demand and sharing pages across sequences that share a common prefix.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| What does the KV cache store? | The Key and Value tensors for all previously processed tokens, reused to avoid recomputation |
| What is FlashAttention? | An implementation of attention that tiles computations in fast SRAM, reducing memory bandwidth and enabling longer contexts |
| What is GQA? | Grouped Query Attention: multiple query heads share K/V heads, reducing KV cache by 4x |
| Why does attention scale quadratically? | Every token attends to every other token: n tokens × n tokens = n² attention scores |
| What is RoPE? | Rotary Position Embedding: encodes position by rotating Q and K vectors, better than absolute position encodings |
| What is multi-head attention? | Running attention in parallel across multiple "heads," each learning different relationship types |
