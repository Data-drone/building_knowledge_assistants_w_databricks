# RAG Pipeline - Quick Start

Build your first AI agents on Databricks in under 40 minutes!

## What You'll Learn

This module teaches you to build AI agents using Databricks primitives:
1. **Notebook 00** (15 min): Your first agent - LLM + conversation loop
2. **Notebook 01** (20-25 min): Add Vector Search tool - RAG agent

## Prerequisites

- Access to Databricks workspace
- LLM endpoint available (`databricks-claude-sonnet-4-5` or similar)
- For Notebook 01: Unity Catalog volume with sample documents

## Learning Path

### Start Here: 00_your_first_agent_on_databricks.py

**What you'll build**: A simple chatbot using LangGraph

**Key concepts**:
- Using `databricks_langchain` to call LLMs
- Building conversation loops with LangGraph
- MLflow auto-logging for observability
- Understanding what makes something an "agent"

**Time**: 15 minutes

---

### Next: 01_building_a_doc_store_on_vector_search.py

**What you'll build**: RAG agent that answers from your documents

**Key concepts**:
- Document chunking for Vector Search
- Creating Delta Sync indexes
- Wrapping Vector Search as a LangChain tool
- RAG pattern (Retrieval + Generation)

**Time**: 20-25 minutes

---

## What's Next?

After completing these notebooks, you'll understand:
- ✅ How agents work (LLM + loop + tools)
- ✅ How to use Databricks LLM endpoints
- ✅ How to build with LangGraph
- ✅ How Vector Search enables RAG

### Continue Learning

**For deeper understanding**:
- Chunking strategies → See other tutorial sections
- Vector Search optimization → See other tutorial sections
- Advanced agent patterns → See other tutorial sections

**For production features** (covered in later modules):
- **Module 02**: Conversation memory with Lakebase
- **Module 03**: Evaluation and metrics
- **Module 04**: Multi-tool agents (Genie, UC Functions)
- **Module 05**: Deployment to Model Serving

---

## Quick Start

1. Open `00_your_first_agent_on_databricks.py`
2. Run all cells
3. Chat with your first agent!
4. Continue to `01_building_a_doc_store_on_vector_search.py`
5. Add Vector Search and build a RAG agent

---

## Technical Stack

- **LLM**: Databricks LLM endpoints (Claude Sonnet 4.5)
- **Orchestration**: LangGraph
- **Vector Search**: Databricks Vector Search with Delta Sync
- **Observability**: MLflow auto-logging
- **Storage**: Unity Catalog (Delta tables, Volumes)

---

## Need Help?

- Check MLflow traces for debugging
- Review error messages (version pins prevent most issues)
- Ensure endpoints are ONLINE before querying

---

**Total Time**: 35-40 minutes from zero to RAG agent 🚀
