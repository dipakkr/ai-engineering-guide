# MLOps for LLMs

> **TL;DR**: LLM MLOps is mostly standard software engineering: version control, CI/CD, blue-green deployments. The LLM-specific parts are prompt versioning (prompts are code, not config), eval gates in CI (a prompt change that drops quality by >3% fails the build), and model version pinning (don't let silent provider updates break you). Start simple: git for prompts, CI eval check, done.

**Prerequisites**: [Eval Fundamentals](../05-evaluation/01-eval-fundamentals.md), [Drift and Monitoring](05-drift-and-monitoring.md)
**Related**: [Observability and Tracing](01-observability-and-tracing.md), [Cost Optimization](07-cost-optimization.md)

---

## LLM MLOps vs Traditional MLOps

Traditional ML MLOps focuses on model versioning, training pipelines, and feature stores. LLM MLOps is different:

| Traditional ML | LLM MLOps |
|---|---|
| Model retraining takes hours/days | Prompt changes take minutes |
| Model versioning = artifact versioning | Model versioning = API endpoint + provider version |
| Feature drift = retrain | Prompt drift = re-tune prompt |
| CI tests model accuracy | CI runs LLM eval with judge |
| Deployment = new model artifact | Deployment = new prompt version |

The key insight: for teams using managed API LLMs, the "model" is mostly the system prompt. Treat prompt engineering as the equivalent of model development. That means: version control, code review, test coverage, and staged rollouts.

---

## Prompt Versioning

Prompts belong in git, not in a database config table.

**Directory structure:**
```
prompts/
├── customer-service/
│   ├── v1.0.txt          # Initial version
│   ├── v1.1.txt          # Added format constraints
│   ├── v2.0.txt          # Major rewrite for new model
│   └── production.txt    # Symlink → v2.0.txt (current production)
├── rag-synthesis/
│   ├── v1.0.txt
│   └── production.txt
└── eval/
    ├── golden-set-v1.json    # Never modify
    ├── golden-set-v2.json    # Added 30 new examples from production
    └── judge-prompt.txt
```

Every prompt change goes through a pull request. The PR description includes: what changed, why, and the eval score delta.

```python
# prompts/loader.py
import os
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(name: str, version: str = "production") -> str:
    """Load a prompt by name. version='production' loads current prod version."""
    prompt_path = PROMPTS_DIR / name / f"{version}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {name}/{version}")
    return prompt_path.read_text(strip=True)

def load_production_prompt(name: str) -> str:
    return load_prompt(name, version="production")
```

---

## CI Eval Gate

Every prompt change must pass an eval gate before merging:

```yaml
# .github/workflows/eval-gate.yml
name: Eval Gate

on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'src/**'

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run eval
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          python scripts/run_eval.py \
            --prompt prompts/customer-service/production.txt \
            --golden-set eval/golden-set-v2.json \
            --baseline-score 0.847 \
            --threshold 0.03 \
            --output eval-results.json

      - name: Comment results
        uses: actions/github-script@v6
        with:
          script: |
            const results = require('./eval-results.json');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              body: `## Eval Results\n- Score: ${results.score.toFixed(3)}\n- Baseline: ${results.baseline.toFixed(3)}\n- Delta: ${results.delta.toFixed(3)}\n- Status: ${results.passed ? '✅ PASS' : '❌ FAIL'}`
            });
```

```python
# scripts/run_eval.py
import argparse
import json
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--golden-set", required=True)
    parser.add_argument("--baseline-score", type=float, required=True)
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prompt = open(args.prompt).read()
    golden_set = json.load(open(args.golden_set))

    scores = evaluate_prompt(prompt, golden_set)
    current_score = sum(scores) / len(scores)

    result = {
        "score": current_score,
        "baseline": args.baseline_score,
        "delta": current_score - args.baseline_score,
        "passed": current_score >= args.baseline_score - args.threshold,
    }

    json.dump(result, open(args.output, "w"), indent=2)

    if not result["passed"]:
        print(f"FAIL: Score {current_score:.3f} is {result['delta']:.3f} below baseline")
        sys.exit(1)

    print(f"PASS: Score {current_score:.3f} (baseline: {args.baseline_score:.3f})")

if __name__ == "__main__":
    main()
```

The eval gate prevents prompt regressions from reaching production. It runs on every PR that touches prompts or application code.

---

## Blue-Green Deployment for Prompts

For high-traffic systems, rolling out a new prompt version to all users at once is risky. Blue-green deployment keeps the old version running while gradually shifting traffic:

```python
import os

CURRENT_PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v2.0")
CANARY_PROMPT_VERSION = os.environ.get("CANARY_PROMPT_VERSION", None)
CANARY_TRAFFIC_PCT = float(os.environ.get("CANARY_TRAFFIC_PCT", "0"))

def get_prompt_for_request(request_id: str) -> tuple[str, str]:
    """Returns (prompt, version) for this request."""
    if CANARY_PROMPT_VERSION and CANARY_TRAFFIC_PCT > 0:
        # Deterministic assignment by request_id
        if hash(request_id) % 100 < CANARY_TRAFFIC_PCT:
            return load_prompt("customer-service", CANARY_PROMPT_VERSION), CANARY_PROMPT_VERSION

    return load_prompt("customer-service", CURRENT_PROMPT_VERSION), CURRENT_PROMPT_VERSION
```

Rollout schedule:
1. Deploy with CANARY_TRAFFIC_PCT=5 for 24 hours
2. Check metrics: thumbs down, error rate, eval scores
3. If no regressions: CANARY_TRAFFIC_PCT=50 for 24 hours
4. If no regressions: promote canary to production (update CURRENT_PROMPT_VERSION)

---

## Environment Management

Separate environments prevent production contamination:

```
dev → staging → production

dev:
  - Use cheaper/faster models (Haiku instead of Opus)
  - Short eval sets (20 examples, not 200)
  - No rate limiting
  - Log everything

staging:
  - Same models as production
  - Full eval set runs
  - Rate limiting matching production
  - Shadow production traffic (5%)

production:
  - Pinned model versions
  - Full observability
  - Alerting enabled
  - All guardrails active
```

```python
import os

ENV = os.environ.get("ENV", "dev")

MODEL_CONFIG = {
    "dev": {
        "model": "claude-haiku-4-5-20251001",
        "eval_set_size": 20,
        "log_all": True
    },
    "staging": {
        "model": "claude-opus-4-6",
        "eval_set_size": 200,
        "log_all": True
    },
    "production": {
        "model": "claude-opus-4-6",
        "eval_set_size": 200,
        "log_all": False  # Sample 10%
    }
}

config = MODEL_CONFIG[ENV]
```

---

## Knowledge Base Versioning

For RAG systems, the knowledge base (indexed documents) is also a versioned artifact:

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class KnowledgeBaseVersion:
    version_id: str
    created_at: datetime
    document_count: int
    embedding_model: str
    chunk_settings: dict
    quality_score: float  # RAGAS context precision on golden set

def promote_knowledge_base(new_version: KnowledgeBaseVersion) -> bool:
    """Promote new KB version after quality check."""
    current = get_production_kb_version()

    # Quality gate: new KB must not decrease context precision by >5%
    if new_version.quality_score < current.quality_score - 0.05:
        print(f"KB promotion blocked: {new_version.quality_score:.3f} < {current.quality_score:.3f} - 0.05")
        return False

    set_production_kb_version(new_version.version_id)
    return True
```

When you update the knowledge base (new documents, re-chunking, embedding model change), run RAGAS eval on it before promoting to production.

---

## Incident Playbook

When something breaks in production:

```markdown
## P1: Quality Regression
1. Check eval score (automated monitoring should have caught this)
2. Identify what changed: prompt version? Model update? Knowledge base?
3. Rollback: revert to last known-good prompt version (< 5 minutes)
4. Diagnose: compare eval scores on previous vs current version
5. Fix: update prompt for new model behavior, rerun eval
6. Validate: eval gate passes, canary at 5% for 24 hours
7. Post-mortem: why wasn't this caught before production?

## P2: High Error Rate
1. Check API provider status page (50% of the time it's a provider incident)
2. Implement exponential backoff if not already present
3. Consider failover to secondary provider if latency > SLA
4. Check recent code changes for new error sources
5. If self-hosted: check GPU memory, restart inference server

## P3: Cost Spike
1. Check for traffic spike (legitimate) vs runaway loop (agent stuck)
2. Look at token count distribution: is anyone sending huge inputs?
3. Check cache hit rate: did caching break?
4. Kill switch: rate limiting or circuit breaker if needed
```

---

## Gotchas

**Prompts in databases create configuration drift.** Storing prompts in a database config table rather than git means: no code review, no history, no easy rollback. Even if prompts are loaded from a database, the source of truth should be git with a deploy process that writes to the database.

**Eval gate costs money.** Running 200 LLM eval calls on every PR costs ~$1-5 per PR. At 20 PRs/day, that's $20-100/day just for CI evals. Use Haiku for the judge when possible. Budget for it explicitly.

**Staging doesn't always predict production behavior.** Model caching, production traffic patterns, and real user queries create conditions that staging doesn't replicate. The canary deployment is the real production test.

**Knowledge base updates can silently degrade RAG quality.** A team adds a large batch of low-quality documents to the knowledge base. Retrieval precision drops because the new documents compete with the good ones. Without a quality gate on KB changes, this goes undetected.

---

> **Key Takeaways:**
> 1. Prompts are code. Version them in git, review them in PRs, test them with eval gates, deploy them with blue-green rollouts.
> 2. The eval gate in CI is the highest-ROI automation: it catches prompt regressions before they reach users, at $1-5 per run.
> 3. Have a rollback ready. The time to plan rollback is before you need it, not during an incident at 2am.
>
> *"If you wouldn't deploy application code without tests, don't deploy prompts without eval gates."*

---

## Interview Questions

**Q: How do you build a deployment pipeline for a production AI assistant that gets prompt updates weekly?**

The core is treating prompts as code. Every prompt update goes through a PR with an automated eval gate: the PR runs the current prompt against a golden set of 200 examples, compares to the baseline score, and fails the PR if quality drops by more than 3%. This prevents regressions from merging.

After merging, the deployment pipeline deploys to staging with a shadow traffic setup: 5% of production traffic flows through the new prompt, we compare quality metrics between staging (new prompt) and production (current prompt). If staging metrics are within bounds after 24 hours, we canary to 10% production traffic, then 50%, then 100%.

For the knowledge base (the indexed documents), same approach: any KB update runs RAGAS context precision evaluation against a fixed query set. Promotion requires context precision to not drop by more than 5% from the current production KB.

Observability: every request logs which prompt version and KB version was used. This means if quality degrades, I can immediately query "which version was this user on?" and confirm whether it correlates with a recent deployment.

Rollback: keep the last 3 prompt versions and KB versions deployable. A one-command rollback to the previous prompt takes 30 seconds. This is the most important thing to have ready before you ever need it.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| Where should prompts be stored? | Git, treated as code; deployed to runtime via a versioned deploy process |
| What is a CI eval gate? | A check in CI that runs the prompt against a golden set and fails the build if quality drops below threshold |
| What is blue-green deployment for prompts? | Running the new prompt on a small traffic percentage (canary) before full rollout |
| What quality gate should KB updates require? | Context precision must not drop by more than 5% on a fixed RAGAS eval set |
| Why use Haiku for the eval judge in CI? | Cost: Haiku is 10x cheaper than Opus for judge calls; good enough for binary pass/fail |
| What's the first step in a quality regression incident? | Rollback to the previous known-good prompt version (< 5 minutes), then diagnose |
