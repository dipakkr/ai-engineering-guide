# Fine-Tuning

> **TL;DR**: Fine-tuning adapts a pre-trained model to your specific task by continuing training on your data. LoRA (Low-Rank Adaptation) makes this practical: instead of updating 7B parameters, you train ~10M adapter weights. QLoRA combines 4-bit quantization with LoRA to fine-tune on a single consumer GPU. The biggest mistake is fine-tuning when you should be prompting. The second biggest mistake is fine-tuning when you should be using RAG.

**Prerequisites**: [Training Pipeline](05-training-pipeline.md), [Small Language Models](07-small-language-models.md), [Quantization](08-quantization.md)
**Related**: [Prompt Optimization](../02-prompt-engineering/05-prompt-optimization.md), [Model Selection](06-model-landscape.md)

---

## When to Fine-Tune (and When Not To)

Fine-tuning takes time, money, and expertise. Before starting, exhaust these alternatives:

**Try first:**
1. Better prompting — system prompt, few-shot examples, chain-of-thought
2. RAG — if the problem is missing knowledge, not behavior
3. Structured output with Instructor/Pydantic — if the problem is output format
4. A different base model — sometimes Haiku already does what you need

**Fine-tune when:**
- The task has a specific style or format that's hard to prompt reliably
- You have 500+ labeled examples of the target behavior
- Latency matters and you can't afford the token overhead of few-shot prompting
- You need consistent behavior across thousands of calls at low cost
- Data privacy requires not sending examples to an API (fine-tune a local model)

**Don't fine-tune when:**
- You have fewer than 200 examples
- You're trying to add new knowledge (use RAG)
- The base model already works with good prompting
- You don't have an eval set to measure improvement

The classic failure mode: spend 3 weeks fine-tuning, get 2% improvement over a well-crafted prompt. If you can get to 95% of the quality with prompting in a day, fine-tuning for the remaining 5% is rarely worth it.

---

## LoRA: The Standard Approach

Full fine-tuning updates all 7 billion parameters of a 7B model. LoRA ([Hu et al. 2022](https://arxiv.org/abs/2106.09685)) inserts small trainable weight matrices alongside the frozen base model weights.

**The math:** For a weight matrix W of shape (d_in, d_out), LoRA approximates the update as:
```
W_new = W + B × A
```
Where B is (d_in, r) and A is (r, d_out), and r is the rank (typically 8-64). Instead of updating d_in × d_out parameters, you update (d_in + d_out) × r parameters — 100-1000x fewer.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset
import torch

# Base model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# LoRA configuration
lora_config = LoraConfig(
    r=16,                          # Rank: 8-64, higher = more capacity
    lora_alpha=32,                 # Scaling factor (typically 2*r)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 41,943,040 || all params: 8,072,384,512 || 0.52%

# Training
training_args = TrainingArguments(
    output_dir="./fine-tuned-lora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
    warmup_steps=100,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
)
trainer.train()
```

After training, you can merge the LoRA weights into the base model for faster inference, or keep them separate and load/unload adapters at runtime.

---

## QLoRA: Fine-Tuning on a Consumer GPU

QLoRA ([Dettmers et al. 2023](https://arxiv.org/abs/2305.14314)) adds 4-bit quantization to LoRA, enabling fine-tuning of large models on consumer hardware:

- Fine-tune a 7B model on a single RTX 4090 (24GB)
- Fine-tune a 13B model on a single A100 40GB
- Fine-tune a 70B model on 2 A100 80GB GPUs

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import torch

# 4-bit quantization for training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # Nested quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

# Required step for training a quantized model
model = prepare_model_for_kbit_training(model)

# LoRA on top of the quantized model
lora_config = LoraConfig(r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
```

The key insight: the base model weights are stored at 4-bit precision (frozen), but the LoRA adapter weights are trained at BF16 precision. The compute happens in BF16; you only pay for 4-bit storage.

**QLoRA hardware requirements (approximate):**

| Model | Full FT | LoRA (BF16) | QLoRA (4-bit) |
|---|---|---|---|
| 7B | 8× A100 80GB | 2× A100 40GB | 1× RTX 4090 24GB |
| 13B | 16× A100 | 4× A100 40GB | 1× A100 40GB |
| 70B | 80× A100 | 8× A100 80GB | 2× A100 80GB |

---

## Data Preparation

The training data format matters as much as the training code. For instruction fine-tuning, use a chat template consistent with the base model's training:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

def format_example(example: dict) -> str:
    """Format a training example using the model's chat template."""
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]
    # apply_chat_template handles <|begin_of_text|>, <|eot_id|>, etc.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

# Example training data structure
training_examples = [
    {
        "system": "You are a customer support agent for Acme Corp.",
        "instruction": "My order hasn't arrived after 2 weeks.",
        "output": "I'm sorry to hear that. Let me look up your order..."
    }
]
```

**Data quality checklist:**
- Minimum: 200 examples. Recommended: 1,000+. More with diversity is better than more with repetition.
- Label quality matters more than quantity. 500 carefully written examples beat 5,000 inconsistent ones.
- Include negative examples (wrong behaviors to avoid) if the model currently does the wrong thing.
- Hold out 10-20% as a test set. Don't use the test set for any decision-making during training.
- Check for contamination: if your test set appears in training data, your eval numbers are meaningless.

---

## Catastrophic Forgetting

Fine-tuning on a narrow task can degrade performance on other tasks. A model fine-tuned to be a customer support agent might start failing at general reasoning.

**Why it happens:** Gradient updates push weights toward the new task distribution. If the new data doesn't represent the full distribution the model was trained on, the weights lose their general-purpose representations.

**Mitigations:**

*LoRA naturally reduces forgetting.* Because only the adapter weights are updated (0.5% of parameters), the base model's representations are frozen. This is one of LoRA's biggest practical advantages over full fine-tuning.

*Replay:* Mix general-purpose training data (from the original pre-training distribution) into your fine-tuning dataset. A 90:10 split of task data to general data often preserves capabilities well.

*Eval the capabilities you care about:* Run your fine-tuned model through a general benchmark (MMLU, HellaSwag) alongside your task-specific eval. If the general benchmark drops >5%, you have a forgetting problem.

---

## Training Hyperparameters

For LoRA fine-tuning, these defaults work for most tasks:

| Hyperparameter | Default | When to Change |
|---|---|---|
| Rank (r) | 16 | Increase to 32-64 for complex tasks |
| lora_alpha | 32 (= 2×r) | Keep at 2×r |
| Learning rate | 2e-4 | Decrease to 1e-4 if training is unstable |
| Batch size | 8-16 effective | Increase with gradient accumulation if you can |
| Epochs | 3 | Reduce to 1-2 if overfitting; check eval loss |
| Warmup | 5-10% of steps | Helps with stability at start |
| Gradient clipping | 1.0 | Standard; prevents gradient explosions |

**Signs of overfitting:**
- Training loss continues decreasing but eval loss starts increasing
- Model outputs are formulaic and repetitive
- Model works well on training examples but fails on new variations

**Signs of underfitting:**
- Training and eval loss are both high and barely decreasing
- Model hasn't learned the target format after 3 epochs

---

## Evaluating Fine-Tuned Models

The eval process is the most important part. Without a rigorous eval, you don't know if fine-tuning helped.

```python
import json
from anthropic import Anthropic

client = Anthropic()

def evaluate_fine_tuned_model(
    test_cases: list[dict],
    model_fn,           # Function that takes input and returns output
    rubric: str
) -> dict:
    """
    Evaluate fine-tuned model using LLM-as-judge against a rubric.
    """
    scores = []
    for case in test_cases:
        output = model_fn(case["input"])

        # LLM judge
        judge_response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"""Rate this response 1-5 based on: {rubric}

Input: {case['input']}
Expected: {case['expected_output']}
Actual: {output}

Score (1-5) and one-sentence reason:"""
            }]
        )
        # Parse score from response
        text = judge_response.content[0].text
        score = int(text[0]) if text[0].isdigit() else 3
        scores.append(score)

    return {
        "mean_score": sum(scores) / len(scores),
        "pass_rate": sum(1 for s in scores if s >= 4) / len(scores),
        "n_evaluated": len(scores)
    }
```

Run this eval before fine-tuning (to establish baseline), after fine-tuning, and after merging to catch regressions. The pass_rate (% scoring 4+) is often more useful than mean score.

---

## Deployment: Merged vs Adapter

After training, you have two options:

**Merge adapter into base model:**
```python
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16
)
peft_model = PeftModel.from_pretrained(base_model, "./fine-tuned-lora")
merged_model = peft_model.merge_and_unload()  # Weights are now baked in
merged_model.save_pretrained("./merged-model")
```

**Keep adapter separate:**
```python
# Load base + adapter at runtime
base_model = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(base_model, "./fine-tuned-lora")
```

**Merged** is better for production: faster inference, simpler deployment, compatible with vLLM and llama.cpp.

**Separate adapter** is better when: you have multiple fine-tuned variants of the same base model (load different adapters for different tasks), or you're still iterating on the adapter.

---

## Gotchas

**Format mismatch kills models.** The most common fine-tuning failure is using the wrong chat template. Each model family has a specific template (`<|system|>`, `[INST]`, `<|begin_of_text|>`, etc.). If you don't use `apply_chat_template()`, you're training the model on malformed data and it will produce garbage at inference.

**Learning rate too high causes loss spikes.** If your training loss spikes up suddenly then comes down, your learning rate is too high. Start at 1e-4 instead of 2e-4 and use a cosine schedule with warmup.

**Fine-tuning doesn't add knowledge.** If the model doesn't know something in its base weights, fine-tuning won't teach it that fact. Fine-tuning adjusts behavior and style; knowledge comes from pre-training or RAG. I've seen teams fine-tune on their product documentation and wonder why the model still doesn't know their product — because fine-tuning on a few hundred examples can't compete with the pre-training signal.

**Rank too high wastes parameters.** r=64 rarely performs meaningfully better than r=16 for most tasks, but it's 4x more parameters to train and store. Start at r=16; increase only if you have evidence it helps.

**Test on distribution shifts.** Your eval set should include examples that are slightly different from your training data — different phrasing, edge cases, topics adjacent to the training domain. If your eval set is drawn from the same distribution as your training data, you're measuring memorization, not generalization.

---

> **Key Takeaways:**
> 1. Fine-tune with LoRA (r=16, target attention and MLP projections), not full fine-tuning. 0.5% of parameters is enough for most task adaptation. Use QLoRA to do it on a single GPU.
> 2. Data quality beats data quantity. 500 carefully curated examples outperform 5,000 noisy ones. Use `apply_chat_template()` — format errors are the most common cause of poor fine-tuning results.
> 3. Establish a rigorous eval before starting. If you can't measure improvement, you can't know if fine-tuning worked. LLM-as-judge with a rubric is a fast way to get a consistent eval signal.
>
> *"Fine-tuning is not a substitute for a good prompt. It's a substitute for an expensive frontier model when you have enough examples of the right behavior."*

---

## Interview Questions

**Q: A team wants to fine-tune GPT-4 to respond in their brand voice. Is this a good idea? What would you recommend?**

Fine-tuning GPT-4 for brand voice is likely overkill and may not even be the right approach. Brand voice is a style problem, and style can usually be captured with a well-crafted system prompt and a few dozen few-shot examples. I'd start there.

If prompting doesn't get close enough, I'd reach for a smaller open-source model + LoRA fine-tuning rather than fine-tuning GPT-4. Reasons: cost (LoRA fine-tuning a 7B model costs ~$5-50 on a cloud GPU; GPT-4 fine-tuning is hundreds of dollars plus ongoing premium costs), control (you own the weights), and the fact that brand voice doesn't require GPT-4-level capability.

The cases where fine-tuning GPT-4 specifically makes sense: you need GPT-4-level capability AND specific consistent formatting AND you don't want to maintain your own infrastructure. That's a narrow set of requirements.

My actual recommendation: write a strong system prompt with 3-5 examples of ideal brand voice responses. Evaluate against 50 test cases. If you're at 85%+ pass rate, ship it. If not, investigate why — is it knowledge, reasoning, or style? Style problems are fine-tuning candidates; knowledge problems are RAG candidates; reasoning problems usually mean you need a better base model.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| What is LoRA? | Low-Rank Adaptation: trains small matrices alongside frozen base model weights, updating ~0.5% of parameters |
| What is QLoRA? | LoRA on a 4-bit quantized model, enabling fine-tuning on consumer GPUs |
| What rank should I use for LoRA? | Start at r=16; increase to 32-64 only if you have evidence it helps |
| What is catastrophic forgetting? | When fine-tuning on a narrow task degrades performance on the original capabilities |
| How many examples do I need? | Minimum 200; recommended 1,000+ with high quality |
| What is `apply_chat_template`? | HuggingFace function that formats training data with the model's specific chat tokens |
