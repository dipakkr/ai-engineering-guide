# ✅ Agent Harnesses & Skills — Interview Checklist

> **TL;DR**: A **syllabus, not notes**. Modern coding/research agents increasingly depend on the **harness** — the loop, tools, context management, and permissions *around* the model — not just the model itself. Study each item, then tick the depth you've reached. ⭐ = high interview leverage — get these to **L3** first.
>
> Every item links to the lesson that teaches it in the **Where** column. This chapter is the practical companion to [Agents & Orchestration](../04-agents-and-orchestration/00-checklist.md).

**How to use this page**

| Level | Box | You can honestly say... |
|---|---|---|
| **L1 — Learned** | `☐` | "I've read it and I understand the idea." |
| **L2 — Explain** | `☐` | "I can explain it out loud, unprompted, in an interview." |
| **L3 — Design/Debug** | `☐` | "I can design one and debug it in a live system." |

Tick a box by replacing `☐` with `✅`.

---

## 1. Harness Fundamentals

| Topic | L1 | L2 | L3 | Where |
|---|:-:|:-:|:-:|---|
| The model loop ⭐ | ☐ | ☐ | ☐ | [Harness Fundamentals](01-harness-fundamentals.md) |
| System prompt | ☐ | ☐ | ☐ | [Harness Fundamentals](01-harness-fundamentals.md) |
| Tool registry ⭐ | ☐ | ☐ | ☐ | [Harness Fundamentals](01-harness-fundamentals.md) |
| Context management | ☐ | ☐ | ☐ | [Harness Fundamentals](01-harness-fundamentals.md) |
| State | ☐ | ☐ | ☐ | [Harness Fundamentals](01-harness-fundamentals.md) |
| Execution environment | ☐ | ☐ | ☐ | [Harness Fundamentals](01-harness-fundamentals.md) |
| Permissions | ☐ | ☐ | ☐ | [Harness Fundamentals](01-harness-fundamentals.md) |
| Termination conditions | ☐ | ☐ | ☐ | [Harness Fundamentals](01-harness-fundamentals.md) |

## 2. Context Management

| Topic | L1 | L2 | L3 | Where |
|---|:-:|:-:|:-:|---|
| Context compaction ⭐ | ☐ | ☐ | ☐ | [Context Management](02-context-management.md) |
| Summarization | ☐ | ☐ | ☐ | [Context Management](02-context-management.md) |
| Selective retrieval | ☐ | ☐ | ☐ | [Context Management](02-context-management.md) |
| File discovery | ☐ | ☐ | ☐ | [Context Management](02-context-management.md) |
| Context budgeting ⭐ | ☐ | ☐ | ☐ | [Context Management](02-context-management.md) |
| Persistent instructions | ☐ | ☐ | ☐ | [Context Management](02-context-management.md) |

## 3. Skills

| Topic | L1 | L2 | L3 | Where |
|---|:-:|:-:|:-:|---|
| Reusable skills ⭐ | ☐ | ☐ | ☐ | [Skills](03-skills.md) |
| Skill discovery & loading | ☐ | ☐ | ☐ | [Skills](03-skills.md) |
| Instructions vs tools vs skills ⭐ | ☐ | ☐ | ☐ | [Skills](03-skills.md) |
| Progressive disclosure ⭐ | ☐ | ☐ | ☐ | [Skills](03-skills.md) |
| Skill composition | ☐ | ☐ | ☐ | [Skills](03-skills.md) |
| Domain-specific skills | ☐ | ☐ | ☐ | [Skills](03-skills.md) |

## 4. Coding-Agent Harnesses

| Topic | L1 | L2 | L3 | Where |
|---|:-:|:-:|:-:|---|
| Filesystem tools ⭐ | ☐ | ☐ | ☐ | [Coding-Agent Harnesses](04-coding-agent-harnesses.md) |
| Shell execution | ☐ | ☐ | ☐ | [Coding-Agent Harnesses](04-coding-agent-harnesses.md) |
| Search | ☐ | ☐ | ☐ | [Coding-Agent Harnesses](04-coding-agent-harnesses.md) |
| Patch / edit tools ⭐ | ☐ | ☐ | ☐ | [Coding-Agent Harnesses](04-coding-agent-harnesses.md) |
| Tests & git | ☐ | ☐ | ☐ | [Coding-Agent Harnesses](04-coding-agent-harnesses.md) |
| Browser | ☐ | ☐ | ☐ | [Coding-Agent Harnesses](04-coding-agent-harnesses.md) |
| Subagents | ☐ | ☐ | ☐ | [Coding-Agent Harnesses](04-coding-agent-harnesses.md) |
| Sandboxing ⭐ | ☐ | ☐ | ☐ | [Coding-Agent Harnesses](04-coding-agent-harnesses.md) |

## 5. Execution

| Topic | L1 | L2 | L3 | Where |
|---|:-:|:-:|:-:|---|
| Planning & task decomposition ⭐ | ☐ | ☐ | ☐ | [Execution](05-execution.md) |
| Parallelization & subagents | ☐ | ☐ | ☐ | [Execution](05-execution.md) |
| Checkpoints | ☐ | ☐ | ☐ | [Execution](05-execution.md) |
| Long-running tasks & resumability ⭐ | ☐ | ☐ | ☐ | [Execution](05-execution.md) |
| Artifacts | ☐ | ☐ | ☐ | [Execution](05-execution.md) |

## 6. Reliability

| Topic | L1 | L2 | L3 | Where |
|---|:-:|:-:|:-:|---|
| Verification loops ⭐ | ☐ | ☐ | ☐ | [Reliability](06-reliability.md) |
| Test-driven agents | ☐ | ☐ | ☐ | [Reliability](06-reliability.md) |
| Self-correction ⭐ | ☐ | ☐ | ☐ | [Reliability](06-reliability.md) |
| Permission boundaries | ☐ | ☐ | ☐ | [Reliability](06-reliability.md) |
| Loop detection | ☐ | ☐ | ☐ | [Reliability](06-reliability.md) |
| Token budgets | ☐ | ☐ | ☐ | [Reliability](06-reliability.md) |
| Error recovery ⭐ | ☐ | ☐ | ☐ | [Reliability](06-reliability.md) |

## 7. Study Real Systems

| System | L1 | L2 | L3 | Where |
|---|:-:|:-:|:-:|---|
| Claude Code ⭐ | ☐ | ☐ | ☐ | [Real Systems](07-real-systems.md) |
| Codex | ☐ | ☐ | ☐ | [Real Systems](07-real-systems.md) |
| OpenAI Agents SDK concepts | ☐ | ☐ | ☐ | [Real Systems](07-real-systems.md) |
| LangGraph ⭐ | ☐ | ☐ | ☐ | [LangGraph Deep Dive](../04-agents-and-orchestration/05-langgraph-deep-dive.md) |
| CrewAI / AutoGen concepts | ☐ | ☐ | ☐ | [CrewAI & AutoGen](../04-agents-and-orchestration/07-crewai-and-autogen.md) |

---

**Done with this chapter?** Every ⭐ topic at L3 → you can reason about *why* a coding agent behaves the way it does and design one. Next: [Evaluation Checklist](../05-evaluation/00-checklist.md).
