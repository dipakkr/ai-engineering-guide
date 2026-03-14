# Code Examples

Seven self-contained Python projects demonstrating the patterns covered in this guide.

## Prerequisites

```bash
pip install anthropic python-dotenv
```

Each project has its own requirements.txt with additional dependencies.

## Projects

| Project | What It Demonstrates | Key Dependencies |
|---|---|---|
| [01-basic-rag](./01-basic-rag/) | Full RAG pipeline from scratch | anthropic, sentence-transformers, numpy |
| [02-advanced-rag](./02-advanced-rag/) | HyDE, reranking, parent-child chunking | anthropic, sentence-transformers, qdrant-client |
| [03-langgraph-agent](./03-langgraph-agent/) | ReAct agent with LangGraph | anthropic, langgraph, langchain-anthropic |
| [04-mcp-server](./04-mcp-server/) | Custom MCP server | anthropic, mcp |
| [05-eval-pipeline](./05-eval-pipeline/) | LLM-as-judge evaluation pipeline | anthropic, pandas |
| [06-semantic-cache](./06-semantic-cache/) | Semantic caching with embeddings | anthropic, sentence-transformers, redis |
| [07-structured-output](./07-structured-output/) | Pydantic structured extraction | anthropic, instructor, pydantic |

## Setup

1. Copy `.env.example` to `.env` in any project directory
2. Add your `ANTHROPIC_API_KEY`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the example: `python main.py`

## Design Philosophy

- Each project is standalone — no shared code between projects
- Under 200 lines per main file so the full logic fits in one read
- Real runnable code, not pseudocode
- Uses the Anthropic Python SDK as the primary interface
