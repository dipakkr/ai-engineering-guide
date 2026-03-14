# DSPy Framework

> **TL;DR**: DSPy replaces hand-written prompts with compiled programs. You define the task signature (input fields, output fields), write a few examples, and DSPy optimizes the prompts for you. The paradigm shift: instead of "write a better prompt," you "run the optimizer." Best when you have labeled data and a clear eval metric. Not worth the overhead for simple one-off tasks.

**Prerequisites**: [Agent Fundamentals](01-agent-fundamentals.md), [Prompting Patterns](../02-prompt-engineering/01-prompting-patterns.md)
**Related**: [Prompt Optimization](../02-prompt-engineering/04-prompt-optimization.md), [Eval Fundamentals](../05-evaluation/01-eval-fundamentals.md)

---

## The Paradigm Shift

The traditional approach to improving LLM outputs: write a prompt, test it, adjust the wording, test again, add few-shot examples, adjust the examples. This is manual, brittle, and doesn't transfer well between models.

DSPy's insight: if you have a clear definition of what "good output" looks like (a metric function), you can treat prompt writing as an optimization problem. The optimizer tries many prompt variants, evaluates them against your metric, and keeps the best. You write less prompt engineering, the system finds better prompts than you would by hand.

It's the difference between hand-tuning hyperparameters and running a hyperparameter search. Both can get good results; automation scales better.

---

## Core Concepts

### Signatures

A Signature defines the task: what goes in, what comes out. It's not a prompt, it's a typed contract:

```python
import dspy

class QuestionAnswer(dspy.Signature):
    """Answer questions based on provided context."""
    context: str = dspy.InputField(desc="Relevant background information")
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="Concise, factual answer")

class SentimentClassifier(dspy.Signature):
    """Classify the sentiment of customer feedback."""
    feedback: str = dspy.InputField()
    sentiment: str = dspy.OutputField(desc="One of: positive, negative, neutral")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")
```

The docstring becomes the task description. The field descriptions guide the LLM. No prompt template, no manual few-shot examples.

### Modules

Modules are components that use signatures. `dspy.Predict` is the basic one:

```python
# Configure the LLM
lm = dspy.LM("anthropic/claude-opus-4-6")
dspy.configure(lm=lm)

# Create a predictor from a signature
qa = dspy.Predict(QuestionAnswer)

# Call it
result = qa(
    context="The Eiffel Tower was built in 1889 for the World's Fair.",
    question="When was the Eiffel Tower built?"
)
print(result.answer)  # "1889"
```

`dspy.ChainOfThought` adds a reasoning step before the answer:

```python
qa_with_reasoning = dspy.ChainOfThought(QuestionAnswer)
result = qa_with_reasoning(context=ctx, question=q)
print(result.reasoning)  # "The context states the tower was built in 1889..."
print(result.answer)     # "1889"
```

### Programs

Chain modules into programs:

```python
class RAGProgram(dspy.Module):
    def __init__(self, retriever):
        self.retriever = retriever
        self.answer = dspy.ChainOfThought(QuestionAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        context = self.retriever(question)
        return self.answer(context=context, question=question)
```

This is your full RAG pipeline expressed as a DSPy program. The key: every LLM call inside `self.answer` is now optimizable.

---

## Optimization: Where DSPy Pays Off

The point of writing DSPy programs rather than direct API calls is that DSPy can optimize them.

```python
from dspy.teleprompt import BootstrapFewShot

# Your training examples
trainset = [
    dspy.Example(
        question="What is the capital of France?",
        context="France is a country in Western Europe. Paris is its capital and largest city.",
        answer="Paris"
    ).with_inputs("question", "context"),
    # ... 20-50 more examples
]

# Define your metric
def exact_match(example, prediction, trace=None) -> bool:
    return example.answer.lower().strip() == prediction.answer.lower().strip()

# Optimize
optimizer = BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=4)
optimized_rag = optimizer.compile(RAGProgram(retriever), trainset=trainset)
```

`BootstrapFewShot` automatically:
1. Generates candidate few-shot examples by running the program on your training set
2. Selects examples that cause the program to produce correct answers
3. Injects them as demonstrations into the prompts

The result: prompts with automatically selected few-shot examples that maximize your metric.

---

## The Optimizer Zoo

| Optimizer | Use When | Requirement |
|---|---|---|
| `BootstrapFewShot` | Simple programs, fast iteration | 10-50 labeled examples |
| `BootstrapFewShotWithRandomSearch` | More thorough optimization | 50-200 labeled examples |
| `MIPRO` | Complex programs, best quality | 100+ examples, slower |
| `BayesianSignatureOptimizer` | Instruction optimization | Labeled examples + LLM judge |
| `BootstrapFinetune` | Distilling to smaller models | 500+ examples, fine-tuning access |

Start with `BootstrapFewShot`. It's the simplest and works well for most use cases.

---

## When DSPy Beats Hand-Written Prompts

The scenarios where DSPy delivers clear ROI:

**Multi-step pipelines with interdependencies.** If step 3 depends on step 2's output format, hand-crafting prompts for each step while keeping them compatible is tedious. DSPy optimizes the full pipeline jointly.

**When you're switching models.** Prompts optimized for Claude often don't transfer to GPT-4o or vice versa. With DSPy, recompile for the new model. The optimizer handles the adaptation.

**Classification and extraction tasks with labeled data.** If you have 100 labeled examples and a clear metric, DSPy will find better few-shot examples than you'd pick by hand.

**Teams without prompt engineering expertise.** DSPy lets engineers define task logic without deep prompt engineering knowledge. The optimizer fills in the craft.

---

## When to Stick With Direct Prompting

| Situation | Skip DSPy | Why |
|---|---|---|
| Simple one-off task | Yes | Setup overhead not worth it |
| No labeled examples | Yes | Optimizer needs training data |
| No clear metric function | Yes | Can't optimize without a score |
| Prompt needs domain-specific framing | Maybe | DSPy-generated prompts can sound generic |
| Single LLM call in production | Yes | Direct call is simpler and debuggable |

The honest trade-off: DSPy adds a compilation step and requires labeled data and metric functions. For simple tasks, the overhead exceeds the benefit.

---

## Concrete Comparison: Hand-Written vs DSPy

```python
# Hand-written approach
SYSTEM_PROMPT = """You are an expert at classifying customer support tickets.
Classify each ticket into: billing, technical, account, general.
Consider the urgency and department.

Examples:
- "My payment failed" -> billing
- "App crashes on login" -> technical
[10 more carefully chosen examples...]
"""

def classify_ticket(ticket: str) -> str:
    response = client.messages.create(
        model="claude-opus-4-6",
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": ticket}]
    )
    return response.content[0].text

# DSPy approach
class TicketClassifier(dspy.Signature):
    """Classify customer support ticket into appropriate department."""
    ticket: str = dspy.InputField(desc="Customer support ticket text")
    category: str = dspy.OutputField(desc="One of: billing, technical, account, general")

classifier = dspy.Predict(TicketClassifier)
# After optimization with 50 labeled examples, DSPy auto-selects better few-shot examples
# and rewrites the instruction to maximize accuracy on your validation set
```

In practice, the DSPy approach often achieves 2-5% higher accuracy on classification tasks after optimization, and the examples it selects are usually more diverse and edge-case-covering than hand-picked ones.

---

## Gotchas

**Compilation is expensive.** Running `BootstrapFewShot` with 50 examples and 4 bootstrap trials makes hundreds of LLM calls. Budget $5-50 in API costs per optimization run depending on model and dataset size.

**Compiled programs aren't human-readable.** The optimized prompts are stored as program state. You can inspect them (`program.dump_state()`) but you can't easily read a single "prompt file." This is a cognitive shift.

**Your metric IS your optimization target.** If you optimize for exact match on a classification task but what users actually want is "correct category AND helpful explanation," you'll get a model that nails the category but ignores the explanation. Define metrics carefully.

**DSPy programs are harder to debug than direct calls.** When a DSPy prediction is wrong, you need to inspect which few-shot examples were injected, what the compiled instruction says, and whether the issue is in the signature or the optimizer. LangSmith tracing helps.

---

> **Key Takeaways:**
> 1. DSPy treats prompt writing as optimization. Define a signature and metric, provide labeled data, run the optimizer. The framework finds better prompts than manual iteration.
> 2. The ROI is highest for multi-step pipelines, model portability, and classification tasks with labeled data. The overhead isn't worth it for simple one-off tasks.
> 3. Compilation is expensive but runs once. The optimized program is cheap to run.
>
> *"DSPy is the difference between tuning a guitar by ear and using a chromatic tuner. Both work; automation is more reliable."*

---

## Interview Questions

**Q: When would you recommend DSPy over traditional prompt engineering for an enterprise RAG system?**

DSPy makes most sense when you have two things: labeled evaluation data and a measurable metric. For an enterprise RAG system where you have a set of question-answer pairs and you can write a metric function (exact match on factual questions, or LLM-as-judge for open-ended ones), DSPy can optimize the retrieval and generation steps jointly.

The specific case where I'd recommend it: if the RAG system needs to work across multiple LLM providers (say, Claude for most queries but a smaller model for cost optimization), DSPy handles the cross-model adaptation by recompiling. Hand-written prompts would need to be rewritten separately for each model.

The case where I'd skip it: if the team is just starting out, doesn't have a labeled eval set yet, or the prompt engineering work is more about domain-specific framing than few-shot selection. DSPy is an optimization tool, not a substitute for understanding the domain.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| What is a DSPy Signature? | A typed definition of a task's inputs and outputs, without specifying the prompt |
| What does DSPy's optimizer do? | Automatically selects few-shot examples and rewrites instructions to maximize a metric |
| What is `BootstrapFewShot`? | The simplest DSPy optimizer: generates and selects few-shot examples from a training set |
| What does "compile" mean in DSPy? | Running the optimizer to find the best prompts for a given program and metric |
| When is DSPy not worth the overhead? | Simple tasks with no labeled data, no clear metric, or where one-off direct prompting suffices |
| What is `ChainOfThought` in DSPy? | A module that adds an explicit reasoning step before the final answer field |
