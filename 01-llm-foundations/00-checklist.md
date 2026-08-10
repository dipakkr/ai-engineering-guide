# ✅ LLM Internals, Training & Fine-Tuning — Interview Checklist

> **TL;DR**: A **syllabus, not notes**. Study each item in the linked lesson, then tick the depth you've actually reached. For the high-leverage (⭐) topics, aim for the **third** column — *Design/Debug* — not just "I read it."

**How to use this page**

| Level | Box | You can honestly say... |
|---|---|---|
| **L1 — Learned** | `☐` | "I've read it and I understand the idea." |
| **L2 — Explain** | `☐` | "I can explain it out loud, unprompted, in an interview." |
| **L3 — Design/Debug** | `☐` | "I can make the tradeoff calls and debug it in a live system." |

Tick a box by replacing `☐` with `✅`. ⭐ = high interview leverage — get these to **L3** first.

**Related chapters**: [Attention Mechanisms](03-attention-mechanisms.md) · [Training Pipeline](05-training-pipeline.md) · [Fine-Tuning](09-fine-tuning.md) · [Quantization](08-quantization.md)

---

## 1. Foundations

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Tokenization & BPE ⭐ | ☐ | ☐ | ☐ | [Tokenization](02-tokenization.md) |
| Embeddings (token → vector) | ☐ | ☐ | ☐ | [Transformer Intuition](01-transformer-intuition.md) |
| Transformer architecture ⭐ | ☐ | ☐ | ☐ | [Transformer Intuition](01-transformer-intuition.md) |
| Self-attention & Q/K/V ⭐ | ☐ | ☐ | ☐ | [Attention Mechanisms](03-attention-mechanisms.md) |
| Multi-head attention | ☐ | ☐ | ☐ | [Attention Mechanisms](03-attention-mechanisms.md) |
| Positional encoding / RoPE | ☐ | ☐ | ☐ | [Attention Mechanisms](03-attention-mechanisms.md) |
| Residual connections & normalization | ☐ | ☐ | ☐ | [Transformer Intuition](01-transformer-intuition.md) |
| Feed-forward network (FFN) | ☐ | ☐ | ☐ | [Transformer Intuition](01-transformer-intuition.md) |
| Logits & softmax | ☐ | ☐ | ☐ | [What is an LLM?](what-is-an-llm.md) |

## 2. Generation / Decoding

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Autoregressive decoding ⭐ | ☐ | ☐ | ☐ | [What is an LLM?](what-is-an-llm.md) |
| Temperature | ☐ | ☐ | ☐ | [What is an LLM?](what-is-an-llm.md) |
| Top-k / top-p (nucleus) sampling ⭐ | ☐ | ☐ | ☐ | [What is an LLM?](what-is-an-llm.md) |
| Greedy & beam search concepts | ☐ | ☐ | ☐ | [What is an LLM?](what-is-an-llm.md) |
| Repetition penalties | ☐ | ☐ | ☐ | [What is an LLM?](what-is-an-llm.md) |
| Stop tokens | ☐ | ☐ | ☐ | [What is an LLM?](what-is-an-llm.md) |
| Structured outputs | ☐ | ☐ | ☐ | [Structured Generation](../02-prompt-engineering/03-structured-generation.md) |

## 3. Architecture

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Encoder vs decoder vs encoder-decoder | ☐ | ☐ | ☐ | [Transformer Intuition](01-transformer-intuition.md) |
| Causal masking | ☐ | ☐ | ☐ | [Attention Mechanisms](03-attention-mechanisms.md) |
| KV cache ⭐ | ☐ | ☐ | ☐ | [Attention Mechanisms](03-attention-mechanisms.md) |
| Context windows ⭐ | ☐ | ☐ | ☐ | [Context Windows](04-context-windows.md) |
| FlashAttention concepts | ☐ | ☐ | ☐ | [Attention Mechanisms](03-attention-mechanisms.md) |
| Mixture of Experts (MoE) | ☐ | ☐ | ☐ | [Model Landscape](06-model-landscape.md) |
| Quantization ⭐ | ☐ | ☐ | ☐ | [Quantization](08-quantization.md) |
| Speculative decoding | ☐ | ☐ | ☐ | [Inference Infrastructure](../06-production-and-ops/04-inference-infrastructure.md) |

## 4. Training

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Pretraining objective (next-token) ⭐ | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |
| Cross-entropy loss | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |
| Gradient descent & backprop basics | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |
| AdamW | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |
| Learning rate & warmup | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |
| Batch size / epochs | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |
| Overfitting | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |
| Distributed training concepts | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |

## 5. Post-Training

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| SFT (supervised fine-tuning) ⭐ | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |
| Instruction tuning | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |
| RLHF ⭐ | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |
| Reward models | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |
| PPO concepts | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |
| DPO ⭐ | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |
| Preference datasets | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |
| Reasoning-model training concepts | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |

## 6. Fine-Tuning

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Full fine-tuning vs PEFT ⭐ | ☐ | ☐ | ☐ | [Fine-Tuning](09-fine-tuning.md) |
| LoRA ⭐ | ☐ | ☐ | ☐ | [Fine-Tuning](09-fine-tuning.md) |
| QLoRA ⭐ | ☐ | ☐ | ☐ | [Fine-Tuning](09-fine-tuning.md) |
| Adapters | ☐ | ☐ | ☐ | [Fine-Tuning](09-fine-tuning.md) |
| Rank / alpha | ☐ | ☐ | ☐ | [Fine-Tuning](09-fine-tuning.md) |
| Dataset preparation | ☐ | ☐ | ☐ | [Fine-Tuning](09-fine-tuning.md) |
| Train / validation split | ☐ | ☐ | ☐ | [Fine-Tuning](09-fine-tuning.md) |
| Catastrophic forgetting | ☐ | ☐ | ☐ | [Fine-Tuning](09-fine-tuning.md) |
| Fine-tuning evaluation | ☐ | ☐ | ☐ | [Fine-Tuning](09-fine-tuning.md) |
| When NOT to fine-tune ⭐ | ☐ | ☐ | ☐ | [Training Pipeline](05-training-pipeline.md) |

## 7. Inference

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Precision: FP32 / FP16 / BF16 | ☐ | ☐ | ☐ | [Quantization](08-quantization.md) |
| INT8 / INT4 | ☐ | ☐ | ☐ | [Quantization](08-quantization.md) |
| Batching & continuous batching ⭐ | ☐ | ☐ | ☐ | [Inference Infrastructure](../06-production-and-ops/04-inference-infrastructure.md) |
| Throughput vs latency ⭐ | ☐ | ☐ | ☐ | [Inference Infrastructure](../06-production-and-ops/04-inference-infrastructure.md) |
| TTFT (time to first token) | ☐ | ☐ | ☐ | [Inference Infrastructure](../06-production-and-ops/04-inference-infrastructure.md) |
| Tokens/sec | ☐ | ☐ | ☐ | [Inference Infrastructure](../06-production-and-ops/04-inference-infrastructure.md) |
| GPU memory math ⭐ | ☐ | ☐ | ☐ | [Inference Infrastructure](../06-production-and-ops/04-inference-infrastructure.md) |
| Model serving | ☐ | ☐ | ☐ | [Inference Infrastructure](../06-production-and-ops/04-inference-infrastructure.md) |

---

**Done with this chapter?** Every ⭐ topic at L3 → you can whiteboard a transformer and defend a fine-tune-vs-RAG call. Next: [Prompt Engineering Checklist](../02-prompt-engineering/00-checklist.md).
