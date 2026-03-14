"""
LLM Evaluation Pipeline
========================
Demonstrates:
- Pointwise LLM-as-judge evaluation
- Pairwise comparison (A vs B)
- Structured rubric scoring
- Results summary with pass/fail

Run: python main.py
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

# ─── Test Cases ───────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "question": "What is photosynthesis?",
        "context": "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen.",
        "expected_topics": ["sunlight", "plants", "glucose or sugar", "oxygen"],
    },
    {
        "question": "How does the Eiffel Tower stay up?",
        "context": "The Eiffel Tower uses a lattice structure of puddled iron. The open lattice design reduces wind resistance and distributes weight efficiently.",
        "expected_topics": ["iron or metal", "lattice or structure", "wind"],
    },
    {
        "question": "What causes rainbows?",
        "context": "Rainbows form when sunlight enters water droplets, refracts (bends), reflects off the back, and refracts again on exit. Different wavelengths bend at different angles, separating colors.",
        "expected_topics": ["light", "water", "refraction or bending", "colors or wavelengths"],
    },
]


# ─── System Under Test ────────────────────────────────────────────────────────

def generate_answer(question: str, context: str, model: str = "claude-haiku-4-5-20251001") -> str:
    """The system being evaluated."""
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Context: {context}\n\nAnswer this question concisely: {question}"
        }]
    )
    return response.content[0].text


# ─── Pointwise Judge ──────────────────────────────────────────────────────────

POINTWISE_RUBRIC = """
Rate the response on a scale of 1-5 based on:
- Accuracy: Does it correctly reflect the context?
- Completeness: Does it cover the key points?
- Conciseness: Is it appropriately brief without being too terse?

Return JSON only: {"score": <1-5>, "reason": "<one sentence>"}
"""


@dataclass
class EvalResult:
    question: str
    answer: str
    score: int
    reason: str
    passed: bool


def judge_pointwise(question: str, context: str, answer: str) -> EvalResult:
    """Use Claude to score a response on a 1-5 rubric."""
    judge_prompt = f"""Question: {question}
Context used: {context}
Response to evaluate: {answer}

{POINTWISE_RUBRIC}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": judge_prompt}]
    )

    try:
        data = json.loads(response.content[0].text.strip())
        score = int(data.get("score", 3))
        reason = data.get("reason", "")
    except (json.JSONDecodeError, KeyError):
        score, reason = 3, "Could not parse judge response"

    return EvalResult(
        question=question,
        answer=answer,
        score=score,
        reason=reason,
        passed=score >= 4
    )


# ─── Pairwise Judge ───────────────────────────────────────────────────────────

def judge_pairwise(question: str, answer_a: str, answer_b: str) -> dict:
    """Compare two answers and determine which is better."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"""Compare these two answers to: "{question}"

Answer A: {answer_a}
Answer B: {answer_b}

Which is better? Return JSON only:
{{"winner": "A" or "B" or "tie", "reason": "<one sentence>"}}"""
        }]
    )

    try:
        return json.loads(response.content[0].text.strip())
    except json.JSONDecodeError:
        return {"winner": "tie", "reason": "Could not parse"}


# ─── Eval Runner ──────────────────────────────────────────────────────────────

def run_eval_suite(test_cases: list[dict]) -> list[EvalResult]:
    results = []
    for i, case in enumerate(test_cases):
        print(f"  Running test {i+1}/{len(test_cases)}: {case['question'][:50]}...")

        answer = generate_answer(case["question"], case["context"])
        result = judge_pointwise(case["question"], case["context"], answer)
        results.append(result)

        time.sleep(0.5)  # Rate limiting

    return results


def print_summary(results: list[EvalResult]):
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"\n[{status}] Score: {r.score}/5")
        print(f"Q: {r.question}")
        print(f"A: {r.answer[:100]}...")
        print(f"Reason: {r.reason}")

    pass_rate = sum(1 for r in results if r.passed) / len(results)
    avg_score = sum(r.score for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"Pass rate: {pass_rate:.0%} ({sum(1 for r in results if r.passed)}/{len(results)})")
    print(f"Average score: {avg_score:.1f}/5")


def run_pairwise_comparison(test_cases: list[dict]):
    """Compare two model variants head-to-head."""
    print("\n" + "="*60)
    print("PAIRWISE COMPARISON: Haiku vs Haiku (different prompts)")
    print("="*60)

    wins_a, wins_b, ties = 0, 0, 0

    for case in test_cases[:2]:  # Just first 2 for demo
        answer_a = generate_answer(case["question"], case["context"])
        # Variant B: more explicit about using context only
        response_b = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"Using ONLY this context: {case['context']}\n\nAnswer in one sentence: {case['question']}"
            }]
        )
        answer_b = response_b.content[0].text

        result = judge_pairwise(case["question"], answer_a, answer_b)
        winner = result["winner"]
        if winner == "A": wins_a += 1
        elif winner == "B": wins_b += 1
        else: ties += 1

        print(f"\nQ: {case['question']}")
        print(f"A: {answer_a[:80]}...")
        print(f"B: {answer_b[:80]}...")
        print(f"Winner: {winner} — {result['reason']}")

    print(f"\nOverall: A wins {wins_a}, B wins {wins_b}, Ties {ties}")


def main():
    print("Running pointwise evaluation...")
    results = run_eval_suite(TEST_CASES)
    print_summary(results)

    print("\nRunning pairwise comparison...")
    run_pairwise_comparison(TEST_CASES)


if __name__ == "__main__":
    main()
