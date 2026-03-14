# Essential Papers

Organized by topic. For each paper I've noted the key contribution and whether it's worth reading in full vs just the abstract and introduction.

---

## Foundations

**Attention Is All You Need** (Vaswani et al., 2017)
[arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
The original transformer paper. Multi-head attention, positional encoding, encoder-decoder architecture for translation. Historical importance is high; many implementation details have been superseded.
*Read:* Abstract + Section 3 (Model Architecture). The rest is specific to translation.

**BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)
[arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
Masked language modeling + next sentence prediction for bidirectional encoders. Foundation for embedding models and classification. Encoder-only; predates GPT-style decoder models.
*Read:* Abstract + Sections 3-4 for the pre-training approach.

**Language Models are Few-Shot Learners (GPT-3)** (Brown et al., 2020)
[arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
Demonstrated that scale alone enables few-shot learning without fine-tuning. Introduced the prompt engineering paradigm. 175B parameter model.
*Read:* Abstract + Section 2 (approach) + the few-shot results tables.

---

## Attention and Efficiency

**FlashAttention: Fast and Memory-Efficient Exact Attention** (Dao et al., 2022)
[arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
IO-aware attention implementation using CUDA tiling. Makes long-context practical. Every production LLM inference stack uses this.
*Read:* Abstract + Section 3. The CUDA kernels are optional unless you're writing inference code.

**GQA: Training Generalized Multi-Query Transformer Models** (Ainslie et al., 2023)
[arxiv.org/abs/2305.13245](https://arxiv.org/abs/2305.13245)
Grouped query attention reduces KV cache by 4-8x with minimal quality loss. Used in Llama 3, Mistral, most modern models.
*Read:* Abstract + Section 2. Short and practical.

**RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021)
[arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
Rotary position embeddings encode relative position by rotating Q/K vectors. Better length generalization than absolute position encodings. Now standard.
*Read:* Abstract + Section 3.2 (the RoPE formulation).

---

## RAG and Retrieval

**Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (Lewis et al., 2020)
[arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
Original RAG paper. DPR retrieval + BART generation. Most production RAG systems use different components but the same architectural pattern.
*Read:* Abstract + Sections 2-3 (model) + Section 4 (experiments).

**Lost in the Middle: How Language Models Use Long Contexts** (Liu et al., 2023)
[arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172)
Demonstrates that LLMs recall information at beginning and end of long contexts better than the middle. The U-shaped recall curve. Essential for designing RAG context placement.
*Read:* Abstract + Section 3 (experiments). The finding is simple; the graphs tell the story.

**RAGAS: Automated Evaluation of Retrieval Augmented Generation** (Es et al., 2023)
[arxiv.org/abs/2309.15217](https://arxiv.org/abs/2309.15217)
Introduces four automatic metrics: faithfulness, answer relevancy, context precision, context recall. Now the standard RAG evaluation framework.
*Read:* Abstract + Section 3 (metrics definitions). Essential for RAG practitioners.

**Dense Passage Retrieval for Open-Domain Question Answering** (Karpukhin et al., 2020)
[arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)
DPR: using separate BERT encoders for queries and documents. Foundation of dense retrieval. Shows dense retrieval beats BM25 on QA tasks.
*Read:* Abstract + Sections 2-3. The training approach is the important part.

---

## Fine-Tuning and Adaptation

**LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2022)
[arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
Trains low-rank matrices alongside frozen base model weights. 0.5% of parameters, full fine-tuning quality. The standard for efficient fine-tuning.
*Read:* Full paper. Short (8 pages) and practically essential.

**QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023)
[arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
4-bit NF4 quantization + double quantization + LoRA enables fine-tuning 65B models on single GPUs. Practical guide to consumer-hardware fine-tuning.
*Read:* Abstract + Sections 2-3 (approach) + Section 4 (results).

**Direct Preference Optimization** (Rafailov et al., 2023)
[arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
Trains on preference pairs directly without a reward model. More stable than PPO-based RLHF. Increasingly used as RLHF replacement.
*Read:* Abstract + Section 4 (method). The math is approachable.

**RLHF: Learning to summarize with human feedback** (Stiennon et al., 2020)
[arxiv.org/abs/2009.01325](https://arxiv.org/abs/2009.01325)
Foundational RLHF paper on summarization. Reward model + PPO training. Pre-InstructGPT but demonstrates the approach.
*Read:* Abstract + Sections 2-3.

**InstructGPT: Training language models to follow instructions** (Ouyang et al., 2022)
[arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
Applies RLHF to GPT-3 to create GPT-3.5. Demonstrates that alignment via RLHF produces more helpful models even at smaller scale.
*Read:* Abstract + Sections 2-3 (methods) + Section 4 (results).

---

## Quantization

**GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** (Frantar et al., 2022)
[arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
Layer-by-layer quantization using Hessian second-order information. 3-4 bit quantization with minimal quality loss.
*Read:* Abstract + Section 3 (method). Technical but accessible.

**AWQ: Activation-aware Weight Quantization for LLM Compression** (Lin et al., 2023)
[arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)
Identifies 1% of salient weights using activation magnitudes and protects them. Consistently better than GPTQ at the same bit width.
*Read:* Abstract + Section 3.

---

## Agents and Reasoning

**ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2022)
[arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
Reasoning + Acting loop: think about what to do, act, observe. The foundational agent pattern used everywhere.
*Read:* Full paper. Short and essential for understanding agents.

**Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** (Wei et al., 2022)
[arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)
Showing reasoning steps before answering dramatically improves performance on multi-step problems. The paper behind CoT prompting.
*Read:* Abstract + Section 2 (method) + the few-shot examples in the appendix.

**Tree of Thoughts: Deliberate Problem Solving with Large Language Models** (Yao et al., 2023)
[arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601)
Extends CoT to explore multiple reasoning paths in a tree structure. Better on hard planning problems.
*Read:* Abstract + Section 2 (method). Useful context for understanding reasoning strategies.

**Toolformer: Language Models Can Teach Themselves to Use Tools** (Schick et al., 2023)
[arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761)
Training LLMs to self-generate API calls. Pre-function-calling but foundational for understanding why tool use works.
*Read:* Abstract + Section 3.

---

## Evaluation

**Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** (Zheng et al., 2023)
[arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)
Systematic evaluation of using GPT-4 as a judge. Position bias, verbosity bias, agreement with humans. Essential reading for anyone building LLM-based evaluation.
*Read:* Full paper. Short and directly applicable.

**Large Language Models are not Fair Evaluators** (Wang et al., 2023)
[arxiv.org/abs/2305.17926](https://arxiv.org/abs/2305.17926)
Documents position bias in LLM-as-judge: judges prefer the first presented option. Calibration techniques.
*Read:* Abstract + Sections 3-4. Informs how to design robust LLM judges.

---

## Inference and Scaling

**Scaling Laws for Neural Language Models** (Kaplan et al., 2020)
[arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)
Power law relationships between model size, data, compute, and loss. Foundation for understanding why bigger models are better.
*Read:* Abstract + Section 4 (empirical scaling laws). The derivations are optional.

**Efficient Large Scale Language Modeling with Mixtures of Experts** (Artetxe et al., 2021)
[arxiv.org/abs/2112.10684](https://arxiv.org/abs/2112.10684)
Mixture of experts for LLMs. Foundation for understanding Mixtral and similar models.
*Read:* Abstract + Section 2.

**vLLM: Efficient Memory Management for LLM Serving with PagedAttention** (Kwon et al., 2023)
[arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
PagedAttention: manages KV cache like OS virtual memory. 2-4x throughput improvement over standard serving.
*Read:* Abstract + Sections 2-3. Essential for infrastructure engineers.

---

## Small Language Models

**Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone** (Abdin et al., 2024)
[arxiv.org/abs/2404.14219](https://arxiv.org/abs/2404.14219)
Microsoft's Phi-3-mini (3.8B) outperforms Llama 8B through heavy data curation. Demonstrates quality > quantity for training data.
*Read:* Abstract + Section 2 (data) + benchmark results.

---

## How to Read Papers Efficiently

For most papers, you don't need to read every section:

1. **Abstract** (2 min): What problem, what approach, what result?
2. **Introduction** (5 min): Context, motivation, main contributions
3. **Method/Model section** (10 min): How does it actually work?
4. **Key results table or figure** (5 min): Does it work?
5. **Conclusion** (2 min): What does it mean?

Skip: Related work (unless you're writing a paper), proofs, full experimental appendices.

For code-oriented engineers: many papers have companion GitHub repos with implementations that are faster to understand than the paper itself.
