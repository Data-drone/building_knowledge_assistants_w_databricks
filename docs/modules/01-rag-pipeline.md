# Module 1: RAG Pipeline

<span class="badge-duration">40 minutes</span>

Build your first agent on Databricks and add document-based Q&A with Vector Search.

## What You'll Build

An agent that answers policy questions from your documents using the RAG (Retrieval-Augmented Generation) pattern.

## Notebooks

| Notebook | Topics | Duration |
|---|---|---|
| [`00_your_first_agent_on_databricks.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/01_rag_pipeline/00_your_first_agent_on_databricks.py) | First agent loop with Databricks-hosted LLM | 15 min |
| [`01_building_a_doc_store_on_vector_search.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/01_rag_pipeline/01_building_a_doc_store_on_vector_search.py) | Document chunking, Delta Sync Vector Search, RAG | 25 min |

## Prerequisites

- Completed [Module 0](00-foundations.md)
- LLM endpoint available (`databricks-claude-sonnet-4-6` or similar)
- Unity Catalog volume with sample documents

## Key Concepts

### Your First Agent

The first notebook demonstrates the core agent pattern:

1. Connect to a Databricks-hosted LLM via `databricks_langchain`
2. Build a conversation loop with LangGraph's `StateGraph`
3. Observe calls with MLflow auto-logging

This is the simplest possible agent — an LLM in a loop. It establishes the foundation that every later module builds on.

### Vector Search + RAG

The second notebook adds document retrieval:

1. **Chunk** documents from the Unity Catalog volume
2. **Create a Delta Sync index** that automatically syncs embeddings
3. **Wrap** Vector Search as a LangChain tool using `VectorSearchRetrieverTool`
4. **Combine** retrieval and generation in one agent

The agent decides when to search documents and weaves retrieved context into its answers.

### Delta Sync Indexes

Delta Sync indexes automatically recompute embeddings when the source Delta table changes. You configure the embedding endpoint and source column once — Databricks handles the rest.

## What You'll Understand

- How agents work: LLM + loop + tools
- How to use Databricks LLM endpoints with LangGraph
- How Vector Search enables RAG
- How `VectorSearchRetrieverTool` wraps search as a tool

## What's Next?

Continue to [Module 2: Memory](02-memory.md) to add conversation context across turns and sessions.
