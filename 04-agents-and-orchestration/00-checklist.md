# ✅ Agents, Memory & MCP — Interview Checklist

> **TL;DR**: A **syllabus, not notes**. Study each item in the linked lesson, then tick the depth you've actually reached. For the high-leverage (⭐) topics, aim for the **third** column — *Design/Debug*.

**How to use this page**

| Level | Box | You can honestly say... |
|---|---|---|
| **L1 — Learned** | `☐` | "I've read it and I understand the idea." |
| **L2 — Explain** | `☐` | "I can explain it out loud, unprompted, in an interview." |
| **L3 — Design/Debug** | `☐` | "I can make the tradeoff calls and debug it in a live system." |

Tick a box by replacing `☐` with `✅`. ⭐ = high interview leverage — get these to **L3** first.

**Related chapters**: [Agent Fundamentals](01-agent-fundamentals.md) · [MCP Protocol](03-mcp-protocol.md) · [Memory & State](10-memory-and-state.md) · [Agent Harnesses](../08-agent-harnesses-and-skills/00-checklist.md)

---

## 1. Agent Fundamentals

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| The agent loop (model / tools / environment) ⭐ | ☐ | ☐ | ☐ | [Agent Fundamentals](01-agent-fundamentals.md) |
| Tool calling ⭐ | ☐ | ☐ | ☐ | [Tool Use & Function Calling](02-tool-use-and-function-calling.md) |
| Planning | ☐ | ☐ | ☐ | [Agent Fundamentals](01-agent-fundamentals.md) |
| Execution & observation | ☐ | ☐ | ☐ | [Agent Fundamentals](01-agent-fundamentals.md) |
| Termination conditions | ☐ | ☐ | ☐ | [Agent Fundamentals](01-agent-fundamentals.md) |
| ReAct ⭐ | ☐ | ☐ | ☐ | [Agent Fundamentals](01-agent-fundamentals.md) |

## 2. Architecture Patterns

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Router | ☐ | ☐ | ☐ | [Agentic Patterns](11-agentic-patterns.md) |
| Orchestrator-worker ⭐ | ☐ | ☐ | ☐ | [Multi-Agent Systems](09-multi-agent-systems.md) |
| Planner-executor | ☐ | ☐ | ☐ | [Agentic Patterns](11-agentic-patterns.md) |
| Evaluator-optimizer | ☐ | ☐ | ☐ | [Agentic Patterns](11-agentic-patterns.md) |
| Supervisor / hierarchical agents ⭐ | ☐ | ☐ | ☐ | [Multi-Agent Systems](09-multi-agent-systems.md) |
| Parallel vs sequential agents | ☐ | ☐ | ☐ | [Multi-Agent Systems](09-multi-agent-systems.md) |
| Event-driven agents | ☐ | ☐ | ☐ | [Agentic Patterns](11-agentic-patterns.md) |

## 3. Tooling

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Function calling ⭐ | ☐ | ☐ | ☐ | [Tool Use & Function Calling](02-tool-use-and-function-calling.md) |
| Schema design ⭐ | ☐ | ☐ | ☐ | [Tool Use & Function Calling](02-tool-use-and-function-calling.md) |
| Tool selection | ☐ | ☐ | ☐ | [Tool Use & Function Calling](02-tool-use-and-function-calling.md) |
| Tool errors & retries | ☐ | ☐ | ☐ | [Tool Use & Function Calling](02-tool-use-and-function-calling.md) |
| Idempotency ⭐ | ☐ | ☐ | ☐ | [Tool Use & Function Calling](02-tool-use-and-function-calling.md) |
| Permissions & sandboxing | ☐ | ☐ | ☐ | [Browser & Computer Use](12-browser-and-computer-use.md) |

## 4. Memory

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Working memory ⭐ | ☐ | ☐ | ☐ | [Memory & State](10-memory-and-state.md) |
| Short-term vs long-term | ☐ | ☐ | ☐ | [Memory & State](10-memory-and-state.md) |
| Episodic / semantic / procedural | ☐ | ☐ | ☐ | [Memory & State](10-memory-and-state.md) |
| Conversation memory & summaries | ☐ | ☐ | ☐ | [Memory & State](10-memory-and-state.md) |
| Vector memory ⭐ | ☐ | ☐ | ☐ | [Memory & State](10-memory-and-state.md) |
| Entity memory | ☐ | ☐ | ☐ | [Memory & State](10-memory-and-state.md) |
| Memory retrieval | ☐ | ☐ | ☐ | [Memory & State](10-memory-and-state.md) |
| Forgetting / TTL | ☐ | ☐ | ☐ | [Memory & State](10-memory-and-state.md) |
| Memory evaluation | ☐ | ☐ | ☐ | [Memory & State](10-memory-and-state.md) |

## 5. MCP (Model Context Protocol)

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Host / client / server ⭐ | ☐ | ☐ | ☐ | [MCP Protocol](03-mcp-protocol.md) |
| Tools / resources / prompts ⭐ | ☐ | ☐ | ☐ | [MCP Protocol](03-mcp-protocol.md) |
| Transports | ☐ | ☐ | ☐ | [MCP Protocol](03-mcp-protocol.md) |
| Capability negotiation | ☐ | ☐ | ☐ | [MCP Protocol](03-mcp-protocol.md) |
| Auth & security ⭐ | ☐ | ☐ | ☐ | [MCP Protocol](03-mcp-protocol.md) |
| MCP vs plain function calling | ☐ | ☐ | ☐ | [MCP Protocol](03-mcp-protocol.md) |
| Building an MCP server/client ⭐ | ☐ | ☐ | ☐ | [MCP Protocol](03-mcp-protocol.md) |

## 6. Advanced Orchestration

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| State machines / stateful graphs ⭐ | ☐ | ☐ | ☐ | [LangGraph Deep Dive](05-langgraph-deep-dive.md) |
| Durable execution & checkpoints ⭐ | ☐ | ☐ | ☐ | [LangGraph Deep Dive](05-langgraph-deep-dive.md) |
| Human-in-the-loop & interrupts | ☐ | ☐ | ☐ | [LangGraph Deep Dive](05-langgraph-deep-dive.md) |
| Agent communication | ☐ | ☐ | ☐ | [Multi-Agent Systems](09-multi-agent-systems.md) |
| Concurrency | ☐ | ☐ | ☐ | [Multi-Agent Systems](09-multi-agent-systems.md) |
| Failure recovery ⭐ | ☐ | ☐ | ☐ | [Agentic Patterns](11-agentic-patterns.md) |
| Deterministic vs autonomous workflows ⭐ | ☐ | ☐ | ☐ | [Agentic Patterns](11-agentic-patterns.md) |
| Framework landscape (LangChain / CrewAI / AutoGen / LlamaIndex) | ☐ | ☐ | ☐ | [LangChain Overview](04-langchain-overview.md) · [CrewAI & AutoGen](07-crewai-and-autogen.md) |

## 7. Security

| Topic | L1 | L2 | L3 | Learn it in |
|---|:-:|:-:|:-:|---|
| Prompt injection into agents ⭐ | ☐ | ☐ | ☐ | [Prompt Security](../02-prompt-engineering/05-prompt-security.md) |
| Tool poisoning | ☐ | ☐ | ☐ | [MCP Protocol](03-mcp-protocol.md) |
| Data exfiltration | ☐ | ☐ | ☐ | [Guardrails & Safety](../06-production-and-ops/02-guardrails-and-safety.md) |
| Excessive agency ⭐ | ☐ | ☐ | ☐ | [Guardrails & Safety](../06-production-and-ops/02-guardrails-and-safety.md) |
| Least privilege & approval boundaries ⭐ | ☐ | ☐ | ☐ | [Browser & Computer Use](12-browser-and-computer-use.md) |

---

**Done with this chapter?** Every ⭐ topic at L3 → you can design a multi-agent system and defend its failure handling. Next: [Agent Harnesses Checklist](../08-agent-harnesses-and-skills/00-checklist.md).
