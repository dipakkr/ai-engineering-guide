# CrewAI and AutoGen

> **TL;DR**: CrewAI and AutoGen are multi-agent frameworks that abstract away the coordination layer between agents. They're useful for exploring multi-agent patterns but not production-ready for most applications. Before reaching for either: exhaust single-agent approaches first. The added complexity of coordinating multiple agents is rarely worth it unless tasks genuinely require parallel specialized work.

**Prerequisites**: [Agent Fundamentals](01-agent-fundamentals.md), [Multi-Agent Systems](09-multi-agent-systems.md)
**Related**: [LangGraph Deep Dive](05-langgraph-deep-dive.md), [Agentic Patterns](11-agentic-patterns.md)

---

## The Honest Context

Multi-agent frameworks get a lot of hype. The demos are compelling: a "researcher" agent feeds findings to a "writer" agent which gets reviewed by an "editor" agent. It feels like an AI team.

The production reality is different. Multi-agent systems multiply failure modes. If each agent has 90% reliability, a 3-agent pipeline has 0.9^3 = 73% reliability on a perfect run. Coordination overhead adds latency and cost. Debugging inter-agent communication is significantly harder than debugging a single agent.

My recommendation: use single agents until you have evidence they can't handle the task. If you've pushed a single agent to its limits and it's genuinely failing on tasks that require parallel specialized work, then evaluate multi-agent frameworks.

---

## CrewAI

CrewAI provides a high-level abstraction for defining agents with roles, goals, and backstories, then organizing them into crews that collaborate on tasks.

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Research Analyst",
    goal="Find accurate information about {topic}",
    backstory="You are an expert researcher with a talent for finding reliable sources.",
    tools=[search_tool],
    verbose=True
)

writer = Agent(
    role="Content Writer",
    goal="Write compelling, accurate content based on research",
    backstory="You are a skilled writer who turns research into clear narratives.",
    verbose=True
)

research_task = Task(
    description="Research the current state of {topic} and identify key trends.",
    agent=researcher,
    expected_output="A structured list of findings with sources"
)

writing_task = Task(
    description="Write a 500-word article based on the research findings.",
    agent=writer,
    context=[research_task],
    expected_output="A complete article ready for publication"
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, writing_task])
result = crew.kickoff(inputs={"topic": "quantum computing"})
```

**What CrewAI gets right:**
- Task dependencies (`context=[research_task]` passes output to the next task)
- Role-based agent definitions are intuitive
- Low boilerplate for simple sequential pipelines

**What CrewAI gets wrong:**
- Limited error recovery: if `researcher` fails, the pipeline often fails silently
- Hard to add custom state or branching logic
- Debugging requires reading CrewAI's source code
- Not designed for complex conditional flows

---

## AutoGen

AutoGen (from Microsoft Research) takes a different approach: agents that communicate with each other through a conversation protocol. Instead of a predefined pipeline, agents discuss and delegate.

```python
from autogen import AssistantAgent, UserProxyAgent

config = {"config_list": [{"model": "gpt-4o", "api_key": "sk-..."}]}

assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful AI assistant. Solve tasks step by step.",
    llm_config=config
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # fully automated
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "/tmp", "use_docker": False}
)

user_proxy.initiate_chat(
    assistant,
    message="Write and run a Python script that calculates the first 10 Fibonacci numbers"
)
```

AutoGen's `UserProxyAgent` can execute code automatically. This is powerful for coding tasks and dangerous for everything else.

**What AutoGen gets right:**
- Conversational agent-to-agent interaction feels natural for some tasks
- Code execution loop (write → run → debug → run again) works well
- Good research paper support (GroupChat, nested chat patterns)

**What AutoGen gets wrong:**
- Code auto-execution is a serious security concern in production
- Conversations can drift or spiral unpredictably
- Hard to add deterministic business logic
- Documentation quality lags behind LangChain/LangGraph

---

## Framework Comparison

| Aspect | CrewAI | AutoGen | LangGraph |
|---|---|---|---|
| Abstraction level | Very high (roles, crews) | High (conversational) | Low (explicit graph) |
| Control flow | Sequential/hierarchical | Conversational | Explicit graph edges |
| Debugging | Hard (abstracted) | Hard (conversation traces) | Easier (node-by-node) |
| Production readiness | Medium | Medium | High |
| State management | Basic | Basic | First-class |
| Human-in-loop | Limited | `human_input_mode` | `interrupt_before` |
| Community | Growing | Active | Active |
| Best for | Simple sequential pipelines | Code execution, research | Complex stateful agents |
| When to avoid | Complex branching, production | Untrusted environments | Simple single-agent tasks |

---

## When Multi-Agent Actually Makes Sense

The pattern that justifies multi-agent complexity:

**Genuinely parallel, specialized work.** A market research task where you simultaneously research competitors, analyze customer data, and review regulatory landscape. Three agents working in parallel, each with domain-specific tools and context, combine results faster than one sequential agent.

**Separation of concerns with different tool access.** Agent A can read production databases (read-only access). Agent B generates draft content in a sandbox. Agent C reviews and approves before writing to production. The separation is enforced by which tools each agent has, not just prompt instructions.

**Debate and review patterns.** One agent generates a response, a second agent critiques it, the first agent revises. This is more reliable than asking one agent to generate and self-critique.

**Long-running research tasks.** Where different sub-tasks can proceed asynchronously over hours.

The key question: can a single agent with good tool design accomplish this? Usually the answer is yes for tasks that seem to require multiple agents.

---

## Gotchas

**Reliability compounds down.** Every additional agent in a sequential pipeline reduces overall reliability. Test your overall task completion rate, not individual agent reliability.

**Context sharing is the hard problem.** How does Agent B know what Agent A did? CrewAI passes task output as context. AutoGen uses the conversation history. Both approaches have failure modes when the relevant context is long or structured.

**Cost scales with agents.** Three agents each making 5 LLM calls = 15 LLM calls. At $0.01 per call, that's $0.15 per task, not $0.05. Model costs and latency scale linearly with agent count.

**"Emergent behavior" is usually "emergent failure."** Multi-agent systems can produce unexpected outputs from agent interactions. In demos this looks like creativity. In production it looks like unreliable behavior.

---

> **Key Takeaways:**
> 1. Multi-agent frameworks look impressive in demos. In production, they multiply failure modes and debugging complexity.
> 2. Exhaust single-agent approaches before multi-agent. The threshold for "we need multiple agents" should be high.
> 3. If you do need multi-agent, LangGraph gives you more control than CrewAI or AutoGen for production use cases.
>
> *"Every agent you add is another LLM that can hallucinate, call the wrong tool, or loop indefinitely. Add them only when you have to."*

---

## Interview Questions

**Q: Your team wants to build a multi-agent system to automate a content creation pipeline: research, writing, and editing. How would you evaluate whether to use CrewAI, AutoGen, or LangGraph?**

I'd start by asking whether multi-agent is necessary at all. A single agent with research tools (web search, knowledge base), a generation step, and a self-review step might handle the task with less complexity. I'd prototype the single-agent version first.

If multi-agent genuinely adds value (say, we want the research and initial drafting to happen in parallel), I'd lean toward LangGraph. CrewAI is faster to prototype but has limited error recovery and state management. For a production content pipeline where we need to handle failures (research returns nothing useful, draft quality is low), LangGraph's explicit graph makes it easier to add conditional branching and retry logic.

AutoGen I'd use only if the pipeline involved code execution or an interactive back-and-forth between agents that resembles a conversation. For a linear research-write-edit pipeline, the conversational model is overkill.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| What is CrewAI? | A multi-agent framework with role-based agents organized into crews with sequential or hierarchical task execution |
| What is AutoGen? | A Microsoft multi-agent framework based on conversational agent interactions; supports code auto-execution |
| Why is reliability a concern with multi-agent systems? | Each agent's error probability compounds: 3 agents at 90% reliability = 73% pipeline reliability |
| What is the main security concern with AutoGen's code execution? | `UserProxyAgent` can execute arbitrary code generated by the LLM; requires sandboxing |
| Which framework gives the most control for complex production agents? | LangGraph: explicit graph structure, first-class state management, `interrupt_before` for human-in-loop |
| When is multi-agent genuinely justified? | Parallel specialized work that can't be serialized, or enforced separation of tool access between agents |
