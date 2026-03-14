# LLM-as-Judge

> **TL;DR**: LLM-as-judge is the only scalable way to evaluate free-form text quality. The key decisions are pointwise vs pairwise (pairwise is more reliable but costs 2x), which model to use as judge (the strongest model you can afford), and how to handle bias (calibrate against human labels, randomize answer order). The biggest failure mode is using LLM judgment without calibrating it against human judgment first.

**Prerequisites**: [Eval Fundamentals](01-eval-fundamentals.md)
**Related**: [Retrieval and RAG Eval](02-retrieval-and-rag-eval.md), [Agent and E2E Eval](04-agent-and-e2e-eval.md), [Prompting Patterns](../02-prompt-engineering/01-prompting-patterns.md)

---

## Why LLM-as-Judge Exists

You have 10,000 generated answers. You need to know which are good and which aren't. Your options:

1. **Human review:** 10 minutes per answer × 10,000 = 1,666 hours. Not feasible for iteration.
2. **String matching:** Checks if "Paris" appears in the answer to "What is the capital of France?" Useless for open-ended generation.
3. **BLEU/ROUGE:** Measures n-gram overlap with a reference. Misses paraphrasing, penalizes valid alternative wordings.
4. **Embedding similarity:** Better, but still fails for subtle quality differences.
5. **LLM-as-judge:** Another LLM evaluates the answer. Scales, handles nuance, catches subtle errors.

LLM-as-judge doesn't replace human judgment. It approximates it at scale. The calibration question ("how close is the LLM's judgment to a human's?") is always the critical one.

---

## Pointwise vs Pairwise

**Pointwise:** Ask the judge to score a single response on a scale (1-5, 0-10, or pass/fail).

**Pairwise:** Show the judge two responses and ask which is better.

```python
from anthropic import Anthropic

client = Anthropic()

# Pointwise: score one response
def pointwise_judge(question: str, answer: str, rubric: str) -> float:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content":
            f"Rate this answer on a scale of 1-5.\n\n"
            f"Rubric: {rubric}\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Score (1-5, return number only):"}]
    )
    try:
        return float(response.content[0].text.strip())
    except ValueError:
        return 3.0  # Default if parsing fails

# Pairwise: compare two responses
def pairwise_judge(question: str, answer_a: str, answer_b: str) -> str:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content":
            f"Which response better answers the question? Consider accuracy, completeness, and clarity.\n\n"
            f"Question: {question}\n\n"
            f"Response A: {answer_a}\n\n"
            f"Response B: {answer_b}\n\n"
            f"Return 'A', 'B', or 'tie' with one sentence explaining why."}]
    )
    text = response.content[0].text.strip()
    if text.upper().startswith("A"):
        return "A"
    elif text.upper().startswith("B"):
        return "B"
    else:
        return "tie"
```

**When to use each:**

| Evaluation Need | Approach | Why |
|---|---|---|
| Track quality over time | Pointwise | Gives you a trend line |
| Compare two system versions | Pairwise | More reliable for detecting small differences |
| Ranking multiple systems | Pairwise (ELO) | Consistent relative ranking |
| Pass/fail threshold | Pointwise | Simple binary decision |
| Detecting regressions | Pairwise vs previous version | Higher sensitivity to quality changes |

Pairwise is more reliable because it's easier for a judge to say "A is better than B" than to assign a number that means something consistent. The numeric scale is subjective; relative comparison is more stable.

---

## Writing a Good Judging Rubric

The rubric determines what the judge evaluates. Vague rubrics produce noisy scores; specific rubrics produce consistent scores.

```python
# Bad rubric: too vague
bad_rubric = "Rate the quality of the answer."

# Better rubric: specific criteria
good_rubric = """Rate the answer on these criteria (1-5 each):
1. Accuracy: Is the information factually correct?
2. Completeness: Does it answer all parts of the question?
3. Clarity: Is it easy to understand?
4. Conciseness: Does it avoid unnecessary length?

Return format: {"accuracy": X, "completeness": X, "clarity": X, "conciseness": X, "overall": X}"""

def structured_judge(question: str, answer: str) -> dict:
    import json
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content":
            f"{good_rubric}\n\nQuestion: {question}\nAnswer: {answer}"}]
    )
    try:
        text = response.content[0].text
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"overall": 3.0}
```

Include examples of 1, 3, and 5 ratings in the rubric. This anchors the judge's scale and produces more consistent scores.

---

## Handling Judge Bias

LLM judges have predictable biases you need to account for:

**Position bias:** Judges prefer the first answer shown in pairwise comparisons. Counteract by randomizing order and averaging both orderings:

```python
def debiased_pairwise(question: str, answer_a: str, answer_b: str) -> str:
    """Run pairwise comparison in both orderings, average results."""
    result_1 = pairwise_judge(question, answer_a, answer_b)  # A first
    result_2 = pairwise_judge(question, answer_b, answer_a)  # B first

    # Flip result_2 (B in position 1 winning = A losing)
    result_2_flipped = "A" if result_2 == "B" else "B" if result_2 == "A" else "tie"

    if result_1 == result_2_flipped:
        return result_1
    return "tie"  # Inconsistent results = too close to call
```

**Verbosity bias:** Judges tend to rate longer answers higher, even when longer isn't better. Counteract with explicit rubric instructions: "Penalize unnecessary length. A 3-sentence answer that fully addresses the question is better than a 10-sentence answer that repeats itself."

**Self-preference bias:** Claude used as a judge tends to rate Claude-generated answers higher than GPT-4 answers, and vice versa. For cross-model comparison, use a third-party judge (GPT-4 as judge when comparing Claude vs Llama) or use human judges as the reference.

**Sycophancy bias:** If the judge sees which system generated the answer, it may favor answers from "prestigious" systems. Always blind the judge to the source model.

---

## Calibration: The Critical Step

An LLM judge is only useful if its scores correlate with human judgments. Calibration measures this correlation.

```python
from scipy.stats import pearsonr, spearmanr

def calibrate_judge(
    questions: list[str],
    answers: list[str],
    human_scores: list[float],
    judge_fn
) -> dict:
    """Measure correlation between LLM judge and human scores."""
    llm_scores = [judge_fn(q, a) for q, a in zip(questions, answers)]

    pearson_corr, pearson_p = pearsonr(human_scores, llm_scores)
    spearman_corr, spearman_p = spearmanr(human_scores, llm_scores)

    return {
        "pearson_correlation": pearson_corr,
        "spearman_correlation": spearman_corr,
        "mean_absolute_error": sum(abs(h - l) for h, l in
                                   zip(human_scores, llm_scores)) / len(human_scores),
        "sample_size": len(human_scores)
    }

# Minimum acceptable calibration:
# Pearson correlation > 0.7
# Spearman correlation > 0.6
# MAE < 0.8 (on a 1-5 scale)
```

Run calibration on 50-100 examples that have both human and LLM scores. If Pearson correlation is below 0.7, your judge is unreliable for that task. Common fixes:
- Better rubric (more specific criteria)
- Better judge model (stronger models correlate better with humans)
- Task-specific few-shot examples in the judge prompt

---

## Choosing the Judge Model

The judge model should be the strongest, most instruction-following model you can use:

| Judge Model | Human Correlation | Cost per 1K evals | Notes |
|---|---|---|---|
| Claude Opus 4.6 | ~0.80-0.88 | ~$15 | Best for nuanced quality |
| Claude Sonnet 4.6 | ~0.75-0.83 | ~$3 | Good balance |
| GPT-4o | ~0.78-0.85 | ~$5 | Strong, use when comparing Claude |
| Claude Haiku 4.5 | ~0.65-0.72 | ~$0.25 | Only for simple pass/fail |
| Llama 3.1 70B | ~0.60-0.70 | ~$0.50 | Self-hosted option |

Numbers are rough approximations. Measure on your specific task. The right choice depends on your quality bar and budget.

**Key rule:** Don't use the same model as judge and generator for the same system. Self-evaluation is systematically biased. Use a different provider as judge when possible.

---

## Structured Judging for Specific Tasks

Different tasks need different rubrics:

**For RAG answers:**
```python
RAG_JUDGE_PROMPT = """Evaluate this RAG system response.

Question: {question}
Retrieved Context: {context}
System Response: {response}

Score each dimension (1-5):
- Faithfulness: Does the response only use information from the context? (5=no hallucinations)
- Completeness: Does it answer all parts of the question using available context?
- Clarity: Is the response well-organized and easy to understand?

Return JSON: {{"faithfulness": X, "completeness": X, "clarity": X}}"""
```

**For code generation:**
```python
CODE_JUDGE_PROMPT = """Evaluate this generated code.

Task: {task}
Generated Code:
```python
{code}
```

Score (1-5):
- Correctness: Does it solve the stated task? (Run mentally through 2-3 test cases)
- Code quality: Is it readable, well-structured, follows Python conventions?
- Edge cases: Does it handle obvious edge cases?
- Efficiency: Is the approach reasonable for the problem size?

Return JSON: {{"correctness": X, "quality": X, "edge_cases": X, "efficiency": X}}"""
```

---

## LLM Judge vs Programmatic Checks

Don't use LLM judgment for things you can check programmatically:

| Check | Use LLM Judge | Use Code |
|---|---|---|
| Code executes without errors | No | Yes: run the code |
| JSON is valid | No | Yes: json.loads() |
| Response length | No | Yes: len(text.split()) |
| Contains prohibited words | No | Yes: regex match |
| Semantic accuracy | Yes | No |
| Tone is appropriate | Yes | No |
| Answer is complete | Yes | Partially |
| Factual correctness | Yes (but verify key facts with retrieval) | No |

A hybrid pipeline uses code for what code can verify and LLM judgment for what requires understanding.

---

## Running Evals at Scale

For 10K+ evaluations, parallelize to control cost and time:

```python
import asyncio
from anthropic import AsyncAnthropic

async_client = AsyncAnthropic()

async def async_judge(question: str, answer: str, rubric: str) -> float:
    response = await async_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=50,
        messages=[{"role": "user", "content":
            f"Rate 1-5:\n{rubric}\n\nQ: {question}\nA: {answer}\n\nScore:"}]
    )
    try:
        return float(response.content[0].text.strip()[0])
    except (ValueError, IndexError):
        return 3.0

async def batch_evaluate(qa_pairs: list[tuple], rubric: str, concurrency: int = 20) -> list[float]:
    semaphore = asyncio.Semaphore(concurrency)
    async def rate_limited_judge(q, a):
        async with semaphore:
            return await async_judge(q, a, rubric)
    tasks = [rate_limited_judge(q, a) for q, a in qa_pairs]
    return await asyncio.gather(*tasks)
```

At 20 concurrent requests, 1000 evaluations take ~30 seconds with Haiku vs ~10 minutes sequentially.

---

## Gotchas

**Judge scores drift with prompt changes.** If you change the judge prompt between eval runs, scores become incomparable. Treat the judge prompt as part of your eval configuration and version it.

**High scores don't mean users are happy.** I've seen systems with LLM judge scores >4/5 where users still gave thumbs down. The judge wasn't measuring what users actually cared about. Always validate against behavioral signals from production.

**Calibration degrades on out-of-distribution queries.** You calibrate against 50 examples. The calibration holds for queries similar to those 50. When the query distribution shifts (new user segment, new use case), recalibrate.

**Judge models have knowledge cutoffs.** If your system generates answers about events after the judge model's training cutoff, the judge can't evaluate factual accuracy. Use retrieval-augmented judging for time-sensitive facts.

**Cost adds up fast.** 10K evals at $0.005 each = $50. 10K evals daily = $1,500/month. Choose judge model quality based on your actual quality bar, not the most powerful model by default.

---

> **Key Takeaways:**
> 1. Calibrate the judge before trusting it. If LLM scores don't correlate >0.70 with human scores (Pearson), fix the rubric or use a stronger judge model.
> 2. Pairwise comparison is more reliable than pointwise scoring for detecting small quality differences. Use it for A/B testing system versions.
> 3. Bias is real and predictable: counteract position bias with double-blind ordering, verbosity bias with explicit rubric instructions, self-preference bias with cross-provider judging.
>
> *"LLM-as-judge is a measuring instrument. Use it after calibrating it, not before."*

---

## Interview Questions

**Q: You need to evaluate whether your customer support bot is giving accurate answers. How do you set up the eval pipeline?**

The core question is what "accurate" means for customer support. I'd define it as: the answer correctly addresses the customer's question based on the company's actual policies, without hallucinating. That breaks into faithfulness (did the bot stick to what's in the knowledge base?) and correctness (is the answer technically right?).

For faithfulness, RAGAS handles this: it compares the answer to the retrieved context and flags claims not supported by the context. That's automated and doesn't need human judgment.

For correctness, I'd use LLM-as-judge with a task-specific rubric. The rubric would have the judge compare the bot's answer to a ground-truth answer (written by a support expert) and score completeness and accuracy. I'd calibrate this by getting human evaluators to score 100 examples, then measuring correlation with the LLM scores. If Pearson is above 0.75, I'll trust the LLM judge for ongoing evaluation.

The ongoing pipeline: every day, sample 200 support conversations from production. Run them through the RAGAS + LLM judge pipeline. Track trends over time. If faithfulness drops below 0.80 or judge scores drop by more than 0.3 points, trigger a manual review and potentially a rollback.

I'd also add behavioral signals from production: was the ticket reopened within 24 hours? Did the customer escalate to a human agent? These aren't perfect quality signals but they're ground truth at scale.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| What is pointwise judging? | Scoring a single response on a numeric scale (1-5) |
| What is pairwise judging? | Comparing two responses to determine which is better |
| What is position bias in LLM judging? | The judge's tendency to prefer the first answer shown in pairwise comparisons |
| How do you counteract position bias? | Run pairwise comparison in both orderings and require consistent results |
| What is calibration for LLM judges? | Measuring correlation between LLM judge scores and human scores; target Pearson > 0.70 |
| Why avoid using the same model as judge and generator? | Self-evaluation is systematically biased toward the model's own outputs |
| What is verbosity bias? | LLM judges tend to rate longer answers higher even when length doesn't mean quality |
