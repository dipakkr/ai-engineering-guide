"""
MCP Server — Notes and Tasks
=============================
A simple MCP server with tools for managing notes and tasks.
Demonstrates the Model Context Protocol server pattern.

Run the server: python server.py
Then use the client: python client.py
"""

import json
import asyncio
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ─── In-memory storage ────────────────────────────────────────────────────────

notes: dict[str, dict] = {}
tasks: dict[str, dict] = {}
next_id = {"notes": 1, "tasks": 1}

# ─── MCP Server Setup ─────────────────────────────────────────────────────────

server = Server("notes-tasks-server")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="create_note",
            description="Create a new note with a title and content",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Note title"},
                    "content": {"type": "string", "description": "Note content"},
                },
                "required": ["title", "content"]
            }
        ),
        types.Tool(
            name="list_notes",
            description="List all notes with their titles and IDs",
            inputSchema={"type": "object", "properties": {}}
        ),
        types.Tool(
            name="get_note",
            description="Get the full content of a note by ID",
            inputSchema={
                "type": "object",
                "properties": {"note_id": {"type": "string"}},
                "required": ["note_id"]
            }
        ),
        types.Tool(
            name="create_task",
            description="Create a task with a description and optional due date",
            inputSchema={
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "due_date": {"type": "string", "description": "ISO date string, e.g. 2025-12-31"},
                },
                "required": ["description"]
            }
        ),
        types.Tool(
            name="complete_task",
            description="Mark a task as completed",
            inputSchema={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"]
            }
        ),
        types.Tool(
            name="list_tasks",
            description="List all tasks, optionally filtered by status",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["all", "pending", "completed"],
                        "default": "all"
                    }
                }
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "create_note":
        note_id = str(next_id["notes"])
        next_id["notes"] += 1
        notes[note_id] = {
            "id": note_id,
            "title": arguments["title"],
            "content": arguments["content"],
            "created_at": datetime.now().isoformat(),
        }
        return [types.TextContent(type="text", text=f"Created note {note_id}: {arguments['title']}")]

    elif name == "list_notes":
        if not notes:
            return [types.TextContent(type="text", text="No notes found.")]
        note_list = "\n".join(f"- [{n['id']}] {n['title']}" for n in notes.values())
        return [types.TextContent(type="text", text=f"Notes:\n{note_list}")]

    elif name == "get_note":
        note = notes.get(arguments["note_id"])
        if not note:
            return [types.TextContent(type="text", text=f"Note {arguments['note_id']} not found.")]
        return [types.TextContent(
            type="text",
            text=f"# {note['title']}\n\n{note['content']}\n\nCreated: {note['created_at']}"
        )]

    elif name == "create_task":
        task_id = str(next_id["tasks"])
        next_id["tasks"] += 1
        tasks[task_id] = {
            "id": task_id,
            "description": arguments["description"],
            "due_date": arguments.get("due_date"),
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        }
        return [types.TextContent(type="text", text=f"Created task {task_id}: {arguments['description']}")]

    elif name == "complete_task":
        task = tasks.get(arguments["task_id"])
        if not task:
            return [types.TextContent(type="text", text=f"Task {arguments['task_id']} not found.")]
        task["status"] = "completed"
        task["completed_at"] = datetime.now().isoformat()
        return [types.TextContent(type="text", text=f"Completed task {arguments['task_id']}: {task['description']}")]

    elif name == "list_tasks":
        status_filter = arguments.get("status", "all")
        filtered = [t for t in tasks.values()
                    if status_filter == "all" or t["status"] == status_filter]
        if not filtered:
            return [types.TextContent(type="text", text=f"No {status_filter} tasks found.")]
        task_list = "\n".join(
            f"- [{t['id']}] {'[x]' if t['status'] == 'completed' else '[ ]'} {t['description']}"
            for t in filtered
        )
        return [types.TextContent(type="text", text=f"Tasks ({status_filter}):\n{task_list}")]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
