# LangChain Overview

> **TL;DR**: LangChain is great for prototyping AI applications quickly. It has abstractions for most things you'd want to do: chains, retrievers, document loaders, prompt templates. The problem is production: the abstractions leak, debug messages are hard to trace, and simple things require understanding the abstraction layer. Use LangChain to explore, consider replacing critical paths with direct API calls before launch.

**Prerequisites**: [RAG Fundamentals](../03-retrieval-and-rag/01-rag-fundamentals.md), [Agent Fundamentals](01-agent-fundamentals.md)
**Related**: [LangGraph Deep Dive](05-langgraph-deep-dive.md), [LlamaIndex and Haystack](08-llamaindex-haystack.md)

---

## The Honest Assessment

LangChain accelerated the entire AI application ecosystem. When it launched in 2022, there was no standard way to wire together LLMs with retrieval, memory, and tools. LangChain provided that scaffolding and became the de facto starting point for AI applications.

The problems emerged as applications scaled. The abstractions that make prototyping fast make production debugging hard. A `RetrievalQA` chain that "just works" in a notebook becomes a black box when you need to trace why it's returning wrong answers. The abstraction layers between your code and the actual API calls obscure what's happening.

My position: LangChain is a productivity tool for exploration. Treat it like scaffolding, not foundation. Use it to figure out what you want to build, then decide how much of the LangChain code to keep.

---

## What LangChain Does Well

**Document loaders:** LangChain has pre-built loaders for PDFs, HTML, Confluence, Notion, Google Drive, S3, and dozens more. This saves real time:

```python
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and chunk a PDF in 5 lines
loader = PyPDFLoader("document.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)
```

Writing this from scratch requires handling PyPDF edge cases, page breaks, metadata extraction. LangChain handles it.

**Rapid RAG prototyping:**

```python
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
qa = RetrievalQA.from_chain_type(
    llm=ChatAnthropic(model="claude-opus-4-6"),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)
answer = qa.invoke({"query": "What is the refund policy?"})
```

Six lines from chunks to working RAG. For a demo or prototype, this is genuinely valuable.

**Prompt templates with variable injection:**

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} at {company}. Answer questions about {domain}."),
    ("human", "{question}")
])
chain = prompt | ChatAnthropic(model="claude-opus-4-6")
chain.invoke({"role": "support agent", "company": "Acme", "domain": "billing", "question": "How do I cancel?"})
```

The pipe operator (`|`) is LangChain's LCEL (LangChain Expression Language), a composable way to chain components. It reads cleanly for simple chains.

---

## What LangChain Does Poorly

**Debugging.** When a `RetrievalQA` chain gives a wrong answer, you need to know: what was retrieved, what prompt was sent, what the raw LLM response was. LangChain's abstractions hide all of this unless you add callbacks. Compare:

```python
# LangChain: need callbacks to see what's happening
from langchain.callbacks import StdOutCallbackHandler
qa = RetrievalQA.from_chain_type(..., callbacks=[StdOutCallbackHandler()])

# Direct: you see everything
query_embedding = embed_model.encode([query])
results = vectorstore.similarity_search_by_vector(query_embedding, k=3)
context = "\n".join([r.page_content for r in results])
response = client.messages.create(model="claude-opus-4-6", messages=[...])
```

The direct version is 4 lines more but you have complete visibility.

**Version instability.** LangChain has a history of breaking changes between minor versions. If you import from `langchain` instead of `langchain_community` or `langchain_core`, you'll get deprecation warnings. The codebase was restructured significantly in 0.1.x and again in 0.2.x. Production applications that aren't pinned to specific versions can break on `pip install --upgrade`.

**Complex agent behavior.** The original LangChain `AgentExecutor` has unpredictable behavior with edge cases: tool errors, max iterations, and output formatting. This is why LangGraph was created. If you're building a serious agent, use LangGraph directly.

---

## The LCEL Pattern

LangChain Expression Language is the modern way to compose LangChain components. It's actually quite clean for simple use cases:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Compose a RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the return policy?")
```

LCEL chains support streaming, batch, and async out of the box. For simple linear chains, LCEL is genuinely elegant.

---

## When to Use LangChain vs When Not To

| Situation | Use LangChain? | Why |
|---|---|---|
| Prototyping a RAG system | Yes | Fast document loaders, quick vectorstore integration |
| Building a production Q&A system | Partially | Use document loaders, replace `RetrievalQA` with direct calls |
| Complex stateful agent | No | Use LangGraph directly |
| Multi-step pipeline with many providers | Maybe | LCEL is clean, but direct is more debuggable |
| You need to trace every LLM call | No | Use LangSmith (which doesn't require LangChain) or direct calls |
| Simple LLM calls in production | No | Direct API is 10x simpler |
| Integrating 3rd-party data sources | Yes | The document loader ecosystem is genuinely valuable |

**The rule I follow:** Use LangChain's utilities (document loaders, text splitters, embeddings wrappers), but avoid LangChain's high-level abstractions (chains, agents) in production code where debuggability matters.

---

## LangSmith: The Actually Useful Part

LangSmith is LangChain's observability platform. It's worth evaluating independently of LangChain itself:

- Traces every LLM call with inputs, outputs, latency, and cost
- Dataset management for eval
- Human feedback collection
- Playground for prompt iteration

It works with direct Anthropic/OpenAI calls via the `LANGCHAIN_TRACING_V2=true` environment variable. You don't need to use LangChain to use LangSmith.

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
# Now all LLM calls are traced, regardless of which SDK you use
```

---

## Gotchas

**The abstraction cost compounds.** `RetrievalQA` → `BaseRetriever` → `VectorStoreRetriever` → `Chroma` → `embedding_function`. When something breaks, you're debugging 4 layers of abstraction.

**`pip install langchain` installs the world.** LangChain has hundreds of optional dependencies. `langchain-community` installs integrations you don't need. Use `langchain-core` for just the base abstractions and add specific integration packages.

**Don't confuse LangChain (framework) with LangGraph (stateful graphs) with LangSmith (observability).** They're related but separate products. You can use any combination.

---

> **Key Takeaways:**
> 1. LangChain excels at prototyping and has a valuable ecosystem of document loaders and integrations. Its high-level chains are hard to debug in production.
> 2. For production, use LangChain's utilities but replace its high-level chains with direct API calls or LangGraph for agents.
> 3. LangSmith (observability) is worth evaluating independently of whether you use LangChain.
>
> *"LangChain is scaffolding. Great for building up, should be reviewed before the building opens to the public."*

---

## Interview Questions

**Q: When would you use LangChain vs building directly on the provider SDK?**

I'd use LangChain when I'm exploring a new use case and want to move fast. The document loaders and text splitters save real time. If I need to load PDFs from S3 or Confluence pages, LangChain has that built. Writing it from scratch would take a day.

For production code, I evaluate each component separately. Document loaders and text splitters are mature and reliable in LangChain. I'd keep those. The high-level chains (`RetrievalQA`, `AgentExecutor`) I'd replace with either direct API calls or LangGraph, because when they fail in production I need to know exactly what happened at each step.

The crossover point for me is: if the code needs to be debuggable and maintainable by someone who hasn't read LangChain's source code, use direct API calls. If it's a prototype or internal tool, LangChain's abstractions are fine.

---

**Quick-fire Questions**

| Question | Answer |
|---|---|
| What is LCEL? | LangChain Expression Language: composable pipeline syntax using the `|` operator |
| What is LangSmith? | Observability platform for LLM applications; works independently of LangChain |
| What replaced LangChain's AgentExecutor? | LangGraph: more structured and debuggable for complex agents |
| What is the main criticism of LangChain in production? | Abstractions make debugging hard; difficult to trace what's happening inside chains |
| What part of LangChain is most worth keeping in production? | Document loaders and text splitters; the integration ecosystem saves real development time |
