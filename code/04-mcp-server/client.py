"""
MCP Client
===========
Connects to the notes-tasks MCP server and lets Claude use its tools.

Start the server first: python server.py
Then run: python client.py
"""

import os
import asyncio
import sys
from dotenv import load_dotenv
import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant with access to a notes and tasks management system.
Use the available tools to help users manage their notes and tasks.
When users ask to create, list, or modify notes or tasks, use the appropriate tools."""


async def run_with_mcp(user_message: str):
    """Connect to MCP server, get tools, run Claude with those tools."""

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Get available tools from the MCP server
            tools_response = await session.list_tools()
            tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools_response.tools
            ]

            client = anthropic.Anthropic()
            messages = [{"role": "user", "content": user_message}]

            # Agentic loop
            while True:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=messages,
                )

                if response.stop_reason == "end_turn":
                    # Extract text response
                    for block in response.content:
                        if hasattr(block, "text"):
                            print(f"Assistant: {block.text}")
                    break

                if response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})

                    # Execute each tool call via MCP
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            print(f"  [Tool: {block.name}({block.input})]")
                            result = await session.call_tool(block.name, block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result.content[0].text if result.content else "Done",
                            })

                    messages.append({"role": "user", "content": tool_results})


async def main():
    tasks = [
        "Create a note titled 'Meeting Notes' with content 'Discussed Q1 roadmap and budget.'",
        "Create tasks: Buy groceries, Schedule dentist appointment, Review PR #42",
        "List all my notes and pending tasks",
    ]

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"User: {task}")
        await run_with_mcp(task)


if __name__ == "__main__":
    asyncio.run(main())
