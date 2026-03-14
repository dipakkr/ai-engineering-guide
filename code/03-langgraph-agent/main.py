"""
LangGraph ReAct Agent
=====================
A ReAct (Reasoning + Acting) agent with:
- Web search tool (simulated)
- Calculator tool
- Persistent conversation state
- Loop with tool execution

Run: python main.py
"""

import os
import json
import math
from typing import Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from typing_extensions import TypedDict

load_dotenv()

# ─── Tools ────────────────────────────────────────────────────────────────────

@tool
def search(query: str) -> str:
    """Search the web for information about a topic."""
    # Simulated search results — replace with real search API
    knowledge = {
        "eiffel tower height": "The Eiffel Tower is 330 meters (1,083 feet) tall.",
        "python creator": "Python was created by Guido van Rossum, first released in 1991.",
        "anthropic": "Anthropic is an AI safety company founded in 2021, creator of Claude.",
        "population france": "France has a population of approximately 68 million people.",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if any(word in query_lower for word in key.split()):
            return value
    return f"No specific results found for '{query}'. Try a more specific query."


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Examples: '2 + 2', 'sqrt(16)', '10 * 5 / 2'."""
    # Safe evaluation with limited scope
    safe_globals = {
        "__builtins__": {},
        "sqrt": math.sqrt,
        "pow": math.pow,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, safe_globals)
        return f"{expression} = {result}"
    except Exception as ex:
        return f"Error evaluating '{expression}': {ex}"


TOOLS = [search, calculate]

# ─── Agent State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ─── Agent Graph ──────────────────────────────────────────────────────────────

llm = ChatAnthropic(model="claude-haiku-4-5-20251001").bind_tools(TOOLS)
tools_by_name = {t.name: t for t in TOOLS}


def call_model(state: AgentState) -> AgentState:
    """LLM decides what to do next."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def execute_tools(state: AgentState) -> AgentState:
    """Execute any tool calls the LLM made."""
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_fn = tools_by_name.get(tool_call["name"])
        if tool_fn:
            result = tool_fn.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    """Route: call tools if LLM made tool calls, else end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", execute_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")  # After tools, always go back to agent

app = graph.compile()


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_agent(question: str) -> str:
    """Run the agent on a question and return the final answer."""
    state = app.invoke({"messages": [HumanMessage(content=question)]})
    return state["messages"][-1].content


def main():
    questions = [
        "How tall is the Eiffel Tower in feet? Also, what is 330 * 3.28084?",
        "Who created Python and in what year? How many years ago was that from 2024?",
        "What is the square root of 144 plus the square root of 225?",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        answer = run_agent(question)
        print(f"A: {answer}")


if __name__ == "__main__":
    main()
