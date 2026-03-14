# LangGraph Deep Dive

> **TL;DR**: LangGraph is the right tool for stateful, multi-step agents that need persistence, branching, and human-in-the-loop. It models your agent as a directed graph where nodes are functions and edges are transitions. The graph structure makes complex agent flows debuggable and maintainable. It's more code than a simple loop, but that code is structured and testable.

**Prerequisites**: [Agent Fundamentals](01-agent-fundamentals.md), [Tool Use and Function Calling](02-tool-use-and-function-calling.md)
**Related**: [LangChain Overview](04-langchain-overview.md), [Agentic Patterns](11-agentic-patterns.md), [Memory and State](10-memory-and-state.md)

---

## Why LangGraph Exists

A basic agent loop is just a `while` loop. So why use a graph framework?

When your agent needs to: (1) branch based on the result of a step, (2) loop back to a previous step under certain conditions, (3) pause and wait for human input, (4) persist state across sessions, or (5) run parallel branches and merge results, a `while` loop becomes a tangle of conditionals. You lose the ability to understand the flow at a glance, test individual paths, or add observability.

LangGraph makes the control flow explicit. Each node is a function. Each edge is a transition (conditional or unconditional). The graph is the documentation of your agent's behavior.

The mental model: LangGraph is to agents what a state machine is to workflows. The added power over a raw state machine is that the transitions can call LLMs and tools.

---

## Core Concepts

### State

State is a typed dictionary that flows through every node. Every node reads from state and writes to state. Between nodes, state is persisted (optionally to a database for cross-session continuity).

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # append-only: new messages added to list
    tool_calls: list
    final_answer: str | None
    iteration_count: int
```

The `Annotated[list, operator.add]` pattern is LangGraph's way of saying "when multiple nodes update this field, append their values rather than overwriting." This is critical for the message history.

### Nodes

Nodes are Python functions that take state and return a partial state update:

```python
def call_model(state: AgentState) -> dict:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        tools=TOOLS,
        messages=state["messages"]
    )
    return {
        "messages": [{"role": "assistant", "content": response.content}],
        "iteration_count": state["iteration_count"] + 1
    }

def execute_tools(state: AgentState) -> dict:
    tool_results = []
    for block in state["messages"][-1]["content"]:
        if hasattr(block, "type") and block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
    return {"messages": [{"role": "user", "content": tool_results}]}
```

### Edges

Edges define transitions. Conditional edges use a router function:

```python
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # Check if the last assistant message contains tool calls
    if isinstance(last_message.get("content"), list):
        has_tool_calls = any(
            getattr(b, "type", None) == "tool_use"
            for b in last_message["content"]
        )
        if has_tool_calls and state["iteration_count"] < 10:
            return "execute_tools"
    return END
```

---

## A Complete ReAct Agent in LangGraph

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def build_agent():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("call_model", call_model)
    graph.add_node("execute_tools", execute_tools)

    # Set entry point
    graph.set_entry_point("call_model")

    # Add edges
    graph.add_conditional_edges(
        "call_model",
        should_continue,
        {"execute_tools": "execute_tools", END: END}
    )
    graph.add_edge("execute_tools", "call_model")

    # Compile with memory for persistence
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

agent = build_agent()

# Run with a thread_id for session persistence
config = {"configurable": {"thread_id": "user-123-session-456"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Research the latest Claude models"}],
     "iteration_count": 0, "tool_calls": [], "final_answer": None},
    config=config
)
```

The `thread_id` is the key to persistence. Resume a conversation later with the same `thread_id` and LangGraph loads the checkpoint from where it left off.

---

## The Agent Flow as a Graph

```mermaid
graph TD
    A([START]) --> B[call_model\nLLM decides next action]
    B --> C{should_continue?}
    C --> |has tool calls| D[execute_tools\nRun tool functions]
    C --> |no tool calls or max iterations| E([END])
    D --> B

    style A fill:#90EE90
    style E fill:#FFB6C1
```

This is the simplest useful graph. Real production agents have more nodes:

```mermaid
graph TD
    A([START]) --> B[classify_query\nRoute to appropriate subgraph]
    B --> C{Query type?}
    C --> |retrieval| D[retrieve_docs]
    C --> |calculation| E[use_calculator]
    C --> |needs clarification| F[ask_clarification\nHuman-in-loop pause]
    D --> G[call_model]
    E --> G
    F --> G
    G --> H{should_continue?}
    H --> |tool calls| I[execute_tools]
    H --> |done| J[format_response]
    I --> G
    J --> K([END])
```

The graph makes this multi-path logic readable. Implementing it in a raw `while` loop would require nested conditionals that are hard to follow.

---

## Human-in-the-Loop

This is LangGraph's killer feature. Pause execution at any point and wait for human input:

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class ApprovalState(TypedDict):
    messages: Annotated[list, operator.add]
    pending_action: dict | None
    approved: bool | None

def propose_action(state: ApprovalState) -> dict:
    # Agent decides what it wants to do
    action = {"type": "delete_file", "path": "/data/old_reports/"}
    return {"pending_action": action}

def check_approval(state: ApprovalState) -> str:
    if state.get("approved") is True:
        return "execute"
    elif state.get("approved") is False:
        return "cancel"
    return "wait_for_human"  # this node is an interrupt point

def execute_action(state: ApprovalState) -> dict:
    action = state["pending_action"]
    # Execute the approved action
    return {"messages": [{"role": "assistant", "content": f"Executed: {action}"}]}

graph = StateGraph(ApprovalState)
graph.add_node("propose", propose_action)
graph.add_node("execute", execute_action)
graph.set_entry_point("propose")
graph.add_conditional_edges("propose", check_approval, {
    "execute": "execute",
    "cancel": END,
    "wait_for_human": END  # graph pauses here, resumes when state is updated
})

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer, interrupt_before=["execute"])
```

When `interrupt_before=["execute"]` is set, the graph pauses before executing the `execute` node and waits. A human reviews the pending action, updates `approved` in the state, and resumes the graph. This is the pattern for any irreversible action.

---

## Persistence: Cross-Session State

LangGraph's checkpointer saves state at every node transition. This enables:

1. **Resume interrupted sessions:** If a long-running agent times out, resume from the last checkpoint
2. **Cross-session memory:** A user's conversation context persists across sessions with the same `thread_id`
3. **Debugging:** Inspect the exact state at any point in a past execution

```python
# Production checkpointer: PostgreSQL
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://user:pass@host/db") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)

# Retrieve past state for debugging
config = {"configurable": {"thread_id": "user-123"}}
state_history = list(app.get_state_history(config))
# state_history[0] is most recent, state_history[-1] is initial
```

Available checkpointers as of early 2025:
- `MemorySaver`: In-memory, development only
- `PostgresSaver`: Production recommended
- `SqliteSaver`: Single-node production or local dev
- `RedisSaver`: High-throughput, short-lived sessions

---

## Parallel Branches

LangGraph supports running multiple nodes concurrently:

```python
from langgraph.graph import StateGraph

class ResearchState(TypedDict):
    query: str
    web_results: list
    db_results: list
    synthesis: str

def search_web(state: ResearchState) -> dict:
    return {"web_results": do_web_search(state["query"])}

def search_database(state: ResearchState) -> dict:
    return {"db_results": do_db_search(state["query"])}

def synthesize(state: ResearchState) -> dict:
    combined = state["web_results"] + state["db_results"]
    return {"synthesis": llm_synthesize(state["query"], combined)}

graph = StateGraph(ResearchState)
graph.add_node("search_web", search_web)
graph.add_node("search_database", search_database)
graph.add_node("synthesize", synthesize)

graph.set_entry_point("search_web")
# Both search nodes run in parallel from START
graph.add_edge("__start__", "search_web")
graph.add_edge("__start__", "search_database")
# synthesize waits for both to complete
graph.add_edge("search_web", "synthesize")
graph.add_edge("search_database", "synthesize")
graph.add_edge("synthesize", END)
```

LangGraph automatically waits for all incoming edges before executing a node. This is the fan-out, fan-in pattern.

---

## Subgraphs

For complex agents, organize logic into subgraphs:

```python
# Define a sub-graph for retrieval
retrieval_graph = StateGraph(RetrievalState)
retrieval_graph.add_node("embed_query", embed_query)
retrieval_graph.add_node("vector_search", vector_search)
retrieval_graph.add_node("rerank", rerank)
# ... etc
compiled_retrieval = retrieval_graph.compile()

# Use it as a node in the main graph
main_graph = StateGraph(MainState)
main_graph.add_node("retrieve", compiled_retrieval)
main_graph.add_node("generate", generate_answer)
```

Subgraphs are reusable, testable units. Test the retrieval subgraph in isolation before integrating it into the main agent.

---

## Concrete Numbers

As of early 2025:

| Metric | Value |
|---|---|
| LangGraph overhead per node | 1-5ms (excluding LLM/tool time) |
| Checkpoint write (MemorySaver) | <1ms |
| Checkpoint write (PostgresSaver) | 5-20ms |
| Checkpoint read (resume) | 10-50ms |
| Max practical graph nodes | 20-50 (beyond that, use subgraphs) |
| LangGraph package size | ~50MB with dependencies |

The framework overhead is negligible. The bottleneck is always LLM calls and tool execution.

---

## LangGraph vs Rolling Your Own

| Aspect | Custom Loop | LangGraph |
|---|---|---|
| Setup time | Minutes | 30-60 minutes |
| Persistence | Write your own | Built-in checkpointers |
| Human-in-loop | Complex custom logic | `interrupt_before` one-liner |
| Parallel branches | Threading/asyncio boilerplate | Declarative edges |
| Debugging | Print statements | State inspection API |
| Visualization | None | Built-in graph visualization |
| Testability | Hard to test partial flows | Test individual nodes |
| When to use | Simple agents, prototypes | Production agents with complex flows |

The crossover point: if your agent needs persistence, human-in-loop, or more than one conditional branch, LangGraph saves you significant debugging time.

---

## Gotchas and Real-World Lessons

**State schema drift breaks checkpoints.** If you change your `AgentState` TypedDict after deploying, old checkpoints from the database may fail to deserialize. Version your state schemas and write migration logic before changing them in production.

**Infinite loops are easy to create.** A conditional edge that always routes back to `call_model` without a max iteration check will run until you hit a timeout or context limit. Always add an iteration counter and a maximum.

**Memory leaks in the message list.** With `Annotated[list, operator.add]`, every tool result gets appended to messages permanently. For long sessions, this fills the context window and eventually causes LLM errors. Implement a message summarization node that fires when message length exceeds a threshold.

**Parallel nodes can cause state race conditions.** If two parallel nodes update the same state key without an `operator.add` reducer, one will overwrite the other. Use reducers for any state that multiple nodes update.

**Checkpointing every node is expensive at scale.** Writing a checkpoint after every node transition means a lot of database writes for complex graphs. Consider checkpointing only at high-value points (after expensive tool calls, after human approvals) rather than everywhere.

---

> **Key Takeaways:**
> 1. LangGraph models agent control flow as a typed state graph. The structure makes complex flows debuggable, testable, and maintainable.
> 2. Persistence (via checkpointers) and human-in-the-loop (via `interrupt_before`) are the features that justify using LangGraph over a raw loop.
> 3. Use subgraphs to organize complex agents into testable units. Test each subgraph independently before integrating.
>
> *"The graph IS the documentation. If you can't draw your agent's flow as a LangGraph diagram, you don't understand it well enough to ship it."*

---

## Interview Questions

**Q: Design a multi-step document processing agent using LangGraph. The agent should extract data, validate it, and conditionally trigger a human review.**

I'd model this as a graph with four nodes: `extract`, `validate`, `human_review` (conditional), and `store`.

The `extract` node takes a document from state, calls an LLM with a structured extraction tool to pull out key fields (name, date, amount, category), and writes the extracted data to state. The `validate` node runs deterministic checks: are required fields present, are dates in the right format, is the amount within expected range. It writes a `validation_result` with a list of issues.

The conditional edge after `validate` routes to `human_review` if any critical validation issues were found (wrong amount, missing required field), or goes straight to `store` if validation passed. The `human_review` node uses `interrupt_before` to pause execution. A human sees the document and the validation issues, corrects the extracted data, approves or rejects, and the graph resumes.

The `store` node writes the validated data to the database and marks the document as processed.

The state schema would include: `document_content`, `extracted_fields`, `validation_result`, `human_corrections` (optional), `approved` (bool), `stored_record_id`.

I'd use PostgresSaver for checkpointing because documents might be large and sessions could span hours (if human review is manual). The `thread_id` maps to the document ID so you can always look up the processing history for any document.

*Follow-up: "How would you handle a batch of 1000 documents?"*

Spawn one LangGraph thread per document (one `thread_id` per document), run them concurrently using `asyncio.gather`. The human review queue becomes a dashboard that shows all documents in `human_review` state, ordered by urgency. A reviewer picks one, updates the state externally, and the graph resumes. This architecture separates the automated and human steps cleanly and scales naturally.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| What is LangGraph's core abstraction? | A directed state graph where nodes are functions and edges are transitions |
| What is a checkpointer? | A storage backend that saves agent state at each node transition, enabling persistence and resumability |
| How does human-in-the-loop work in LangGraph? | Use `interrupt_before=["node_name"]` to pause the graph before a node; resume after updating state |
| What does `Annotated[list, operator.add]` mean in a state TypedDict? | Multiple nodes updating this field will append their values rather than overwrite |
| What is the `thread_id` used for? | Uniquely identifies a conversation or session; LangGraph loads the matching checkpoint on each call |
| When should you use subgraphs? | When a portion of the graph is complex enough to test independently, or reused in multiple graphs |
| What is the main risk of the `Annotated[list, operator.add]` message reducer? | The list grows indefinitely; implement summarization before it overflows the context window |
