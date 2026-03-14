# Tool Use and Function Calling

> **TL;DR**: Function calling lets LLMs invoke structured functions instead of returning free text. Every major provider (Anthropic, OpenAI, Google) supports it but with different schemas. The patterns are the same: define tools, parse the model's choice, execute, return results. Parallel tool calls are underused and can 3-5x agent throughput. Error handling is where most implementations break.

**Prerequisites**: [Agent Fundamentals](01-agent-fundamentals.md)
**Related**: [MCP Protocol](03-mcp-protocol.md), [LangGraph Deep Dive](05-langgraph-deep-dive.md), [Agentic Patterns](11-agentic-patterns.md)

---

## The Core Idea

Without function calling, an LLM returns text. You have to parse that text to extract structured data or decide what to do next. This is fragile: the model might format things differently across calls, or include explanation text that breaks your parser.

Function calling gives the model a way to say "I want to call function X with arguments Y" in a structured, parseable format. You define the available functions (name, description, parameter schema). The model chooses which function to call and with what arguments. You execute it, return the result, the model continues.

The key insight: the model doesn't execute the function. It just produces a structured request. Your code executes it. This separation is what makes function calling safe and auditable.

---

## Provider Comparison

All major providers support function calling but use different field names. If you're building multi-provider systems, abstract over these differences.

```python
# Anthropic (Claude)
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City and country"}
        },
        "required": ["location"]
    }
}]

# OpenAI (GPT-4)
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and country"}
            },
            "required": ["location"]
        }
    }
}]

# Google (Gemini)
tools = [{"function_declarations": [{
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    }
}]}]
```

| Aspect | Anthropic | OpenAI | Google |
|---|---|---|---|
| Schema key | `input_schema` | `function.parameters` | `parameters` |
| Tool wrapper | top-level object | `type: "function"` wrapper | `function_declarations` list |
| Parallel calls | Yes (multiple tool_use blocks) | Yes (multiple tool_calls) | Yes |
| Forced tool | `tool_choice: {"type": "tool", "name": "X"}` | `tool_choice: {"type": "function", "function": {"name": "X"}}` | `tool_config` |
| Stop reason | `stop_reason: "tool_use"` | `finish_reason: "tool_calls"` | `finish_reason: "STOP"` with function call |

---

## A Complete Tool Use Loop (Anthropic)

```python
import anthropic
import json

client = anthropic.Anthropic()

def get_weather(location: str) -> str:
    # Real implementation would call a weather API
    return f"Weather in {location}: 22°C, partly cloudy"

def search_web(query: str) -> str:
    return f"Top results for '{query}': [result 1, result 2, result 3]"

TOOLS = [
    {"name": "get_weather", "description": "Get weather for a location",
     "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}},
    {"name": "search_web", "description": "Search the web for current information",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]

TOOL_FNS = {"get_weather": get_weather, "search_web": search_web}

def run_with_tools(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-opus-4-6", max_tokens=1024, tools=TOOLS, messages=messages
        )

        if response.stop_reason == "end_turn":
            return next(b.text for b in response.content if hasattr(b, "text"))

        # Collect all tool calls (could be parallel)
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                fn = TOOL_FNS.get(block.name)
                result = fn(**block.input) if fn else f"Unknown tool: {block.name}"
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
```

This handles the full loop: single calls, parallel calls, and termination. The `while True` loop with `end_turn` check is the standard pattern.

---

## Parallel Tool Calls

This is the most underused feature. When an agent needs multiple independent pieces of information, a capable model will call all the tools simultaneously in one response rather than sequentially.

```python
# One request, model returns TWO tool calls simultaneously
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    tools=TOOLS,
    messages=[{"role": "user", "content": "What's the weather in Paris and Tokyo?"}]
)

# response.content has two tool_use blocks:
# block[0]: get_weather(location="Paris, France")
# block[1]: get_weather(location="Tokyo, Japan")
```

For a naive sequential implementation, this would be 2 tool calls + 2 LLM calls. With parallel tool calls it's 1 LLM call + 2 (parallelizable) tool calls + 1 final LLM call. At 500ms per LLM call and 200ms per tool call, sequential = 1400ms, parallel = 700ms.

For agents with 5+ independent lookups per step, parallel tool calls are a significant throughput win.

---

## Forcing Tool Use

Sometimes you want to guarantee the model uses a specific tool:

```python
# Anthropic: force a specific tool
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    tools=TOOLS,
    tool_choice={"type": "tool", "name": "get_weather"},
    messages=[{"role": "user", "content": "Paris"}]
)

# Anthropic: force any tool (disable free-text response)
tool_choice={"type": "any"}

# OpenAI equivalent
tool_choice={"type": "function", "function": {"name": "get_weather"}}
```

Use cases: structured extraction (always call `extract_data`), classification (always call `classify`), guaranteed JSON output without a JSON mode.

---

## Error Handling

Most tutorials skip this. Real agents break here.

```python
def safe_tool_call(name: str, arguments: dict, tool_fns: dict) -> str:
    """Execute a tool call with comprehensive error handling."""
    if name not in tool_fns:
        return f"Error: Tool '{name}' not found. Available: {list(tool_fns.keys())}"

    fn = tool_fns[name]
    try:
        result = fn(**arguments)
        if result is None:
            return "Tool returned no result"
        # Truncate very long results to avoid context overflow
        result_str = str(result)
        if len(result_str) > 5000:
            return result_str[:5000] + f"\n[Result truncated, {len(result_str)} chars total]"
        return result_str
    except TypeError as e:
        return f"Error: Invalid arguments for {name}: {e}"
    except Exception as e:
        return f"Error executing {name}: {type(e).__name__}: {e}"
```

Return error descriptions in natural language. The model can then reason about the error and decide to retry with different arguments, call a different tool, or explain the failure to the user.

---

## Tool Schema Design Principles

| Principle | Bad | Good |
|---|---|---|
| Description specificity | "Get data" | "Get user account details including name, email, plan tier, and creation date. Returns 'not found' for invalid IDs." |
| Parameter names | `q`, `p`, `n` | `query`, `page`, `num_results` |
| Optional vs required | Make everything required | Mark optional params; provide defaults |
| Return contract | Undescribed | "Returns JSON with fields: {name, email, status}" |
| Error communication | Raise exception | Return descriptive error string |
| Scope | "do_everything" mega-tool | One clear responsibility per tool |

The model reads your descriptions and uses them to decide when and how to call each tool. Vague descriptions cause wrong tool selection. Missing parameter descriptions cause wrong argument values.

---

## Structured Extraction Pattern

Function calling is the best way to extract structured data reliably:

```python
from pydantic import BaseModel

class ContactInfo(BaseModel):
    name: str
    email: str | None
    phone: str | None
    company: str | None

extract_tool = {
    "name": "extract_contact",
    "description": "Extract contact information from text",
    "input_schema": ContactInfo.model_json_schema()
}

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=256,
    tools=[extract_tool],
    tool_choice={"type": "tool", "name": "extract_contact"},
    messages=[{"role": "user", "content": "Reach out to John Smith at john@acme.com, (415) 555-0123, Acme Corp"}]
)

tool_call = next(b for b in response.content if b.type == "tool_use")
contact = ContactInfo(**tool_call.input)
```

Using Pydantic to generate the JSON schema ensures the tool schema is always in sync with your data model.

---

## Concrete Numbers

As of early 2025:

| Metric | Value |
|---|---|
| Overhead of one tool call round-trip | +1 LLM call (~500ms-2s) |
| Parallel tool call latency savings | 40-70% vs sequential for 3+ tools |
| Max tools per call (Anthropic) | No hard limit, ~20 practical |
| Input tokens for tool definitions | ~50-200 tokens per tool |
| Cost of tool definitions at scale | $0.001-0.005 per query for typical tool sets |

Tool definitions add to every input. A 10-tool agent with 150 tokens per tool definition adds 1500 tokens to every request. At $3/1M tokens, that's $0.0045 per query just for tool definitions. Not a lot individually, but it adds up at scale.

---

## Gotchas and Real-World Lessons

**Tool descriptions are prompt engineering.** The model picks tools based on descriptions. "Search for information" is worse than "Search the internal knowledge base for company policies, procedures, and documentation. Use this before web search." Be specific about when to use each tool.

**Hallucinated tool arguments are silent failures.** The model might call `get_user(user_id="john_doe")` when IDs are integers. Your tool returns an error. If you return that error as a string and the model sees it, it might try again correctly. If you raise an exception and crash, you lose the context. Always return errors as strings.

**The conversation grows with every tool call.** Each tool result gets appended to the message history. A 10-step agent with 1K-token tool results per step adds 10K tokens of history. After 50 steps, you've consumed 50K tokens of context. Summarize or prune history for long-running agents.

**Parallel tool calls aren't guaranteed.** Smaller models and less capable model versions may call tools sequentially even when parallel calls are possible. If your agent's performance depends on parallel calls, test with the specific model version you're deploying.

**JSON schema validation is your first defense.** Add server-side validation of tool arguments before executing. If the model hallucinates an extra field or wrong type, reject it with a descriptive error: "Invalid argument: user_id must be integer, got string 'abc'". The model usually corrects on retry.

---

> **Key Takeaways:**
> 1. Function calling is the same concept across all providers; only the schema syntax differs. Parallel tool calls are a major throughput win.
> 2. Tool descriptions are prompt engineering. Specificity in descriptions reduces wrong tool selection and bad argument values.
> 3. Return errors as strings, never crash. The model can reason about errors and retry; a crash loses everything.
>
> *"A tool call is only as good as its description. If the model calls the wrong tool, read the description, not the model."*

---

## Interview Questions

**Q: Design the tool-use architecture for an AI assistant that needs to query both a SQL database and a vector database. How do you handle errors and prevent the model from running dangerous queries?**

I'd define two tools: `query_sql` for structured data lookups and `search_knowledge_base` for semantic retrieval. The key constraint is that `query_sql` must be read-only. I'd enforce this in two ways: the database user the tool connects with has SELECT-only permissions (defense in depth), and the tool code checks that every SQL string starts with SELECT and rejects anything else with a clear error message. The double enforcement is important because prompt injection through user input could try to manipulate the model into running DELETE or DROP statements.

For error handling, both tools return strings in all cases. A database connection failure returns "Database unavailable, try again later." A malformed SQL query returns the PostgreSQL error message in plain text. The model can then decide whether to try a different query, use the vector search instead, or tell the user it can't retrieve the data.

For the vector database tool, I'd add a `max_results` parameter capped at 10 to prevent the model from retrieving enormous result sets that overflow the context window. The tool description specifies what the vector database contains so the model knows when to use it vs the SQL database.

*Follow-up: "How would you handle a situation where the model calls a tool with invalid arguments?"*

The tool validates arguments before executing and returns a descriptive error string: "Invalid argument: user_id must be a positive integer, received 'user-123'". This goes back to the model as a tool result. Most capable models will read the error, correct the argument, and retry. I'd also add logging on all invalid argument attempts, because patterns in errors often reveal prompt injection attempts or edge cases in user queries that the tool descriptions didn't anticipate.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| What does `stop_reason: "tool_use"` mean in Anthropic's API? | The model wants to call one or more tools; check content blocks for tool_use blocks |
| How do parallel tool calls reduce latency? | Model returns multiple tool calls in one response; execute them concurrently, then one more LLM call |
| How do you force the model to use a specific tool? | `tool_choice: {"type": "tool", "name": "tool_name"}` in Anthropic |
| What is the best way to do structured extraction with function calling? | Define a tool whose schema matches your target structure, force tool use, parse the input as your model |
| Why should tools return error strings rather than raise exceptions? | The model can reason about string errors and retry; exceptions crash the agent loop and lose context |
| How do tool definitions affect cost? | Each tool adds ~50-200 tokens to every input; 10 tools adds ~1500 tokens per request |
