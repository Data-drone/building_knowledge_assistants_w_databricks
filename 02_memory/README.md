# Agent Memory - Adding Conversation Context

Give your RAG agents memory in under 45 minutes!

## What You'll Learn

This module teaches you to add memory to AI agents using Databricks primitives:
1. **Notebook 01** (20 min): Short-term memory with CheckpointSaver - Multi-turn conversations
2. **Notebook 02** (30 min): Long-term memory with DatabricksStore - Cross-session personalization

## Prerequisites

### Required
- **Vector Search index** available (from Module 01 or workspace setup)
  - Expected index: `{CATALOG}.{SCHEMA}.policy_index`
  - If you haven't set this up, see [Module 01](../01_rag_pipeline/README.md) or workspace setup guide
- **Lakebase instance** created (usually done in workspace setup)
- LLM endpoint available (`databricks-claude-sonnet-4-6` or similar)

### Recommended
- Completed [Module 01 (RAG Pipeline)](../01_rag_pipeline/README.md) to understand the RAG agent architecture
  - Module 02 is self-contained, but Module 01 provides valuable context

## Learning Path

### Start Here: 01_short_term_memory.py

**What you'll build**: RAG agent with conversation memory

**Key concepts**:
- Building RAG agents with Vector Search tools
- Adding CheckpointSaver for conversation continuity
- Thread-based conversation management
- Multi-turn conversations with context
- "Before/after" demonstration of memory value

**Time**: 20 minutes

---

### Next: 02_long_term_memory.py

**What you'll build**: RAG agent with LLM-managed cross-session memory

**Key concepts**:
- DatabricksStore for persistent facts (manual put/get → understand the primitives)
- Namespace organization for memory types
- Limitations of manual memory and why the LLM-managed pattern is better
- Memory tools: `get_user_memory` (semantic search), `save_user_memory`, `delete_user_memory`
- Store and user_id passed via `RunnableConfig` (production pattern)
- Combining short-term + long-term memory with the agent deciding what to remember

**Time**: 30 minutes

---

## What's Next?

After completing these notebooks, you'll understand:
- ✅ How to add memory to RAG agents
- ✅ Short-term vs long-term memory patterns
- ✅ Thread-based conversation management
- ✅ Cross-session personalization
- ✅ LLM-managed memory tools (the production pattern)
- ✅ Semantic search over memories vs exact key lookups
- ✅ Memory deletion for user control and GDPR compliance

### Continue Learning

**For deeper understanding**:
- Memory management strategies → See notebook sections
- Privacy and retention policies → See notebook sections
- Production deployment patterns → See notebook sections

**For production features** (covered in later modules):
- **Module 03**: Evaluation and observability
- **Module 04**: Extending with data tools (SQL, Genie, custom tools)
- **Module 05**: Deployment to Databricks Apps

---

## Quick Start

1. Verify Vector Search index exists: `{CATALOG}.{SCHEMA}.policy_index`
   - If not, complete [Module 01](../01_rag_pipeline/README.md) or see workspace setup guide
2. Open `01_short_term_memory.py`
3. Add CheckpointSaver to enable multi-turn conversations
4. Test the "before/after" examples
5. Continue to `02_long_term_memory.py`
6. Add DatabricksStore for cross-session personalization
7. See agents remember facts across sessions!

---

## Technical Stack

- **LLM**: Databricks LLM endpoints (Claude Sonnet 4.6)
- **Orchestration**: LangGraph
- **Vector Search**: Databricks Vector Search (for RAG)
- **Short-term memory**: CheckpointSaver (Lakebase PostgreSQL)
- **Long-term memory**: DatabricksStore (Lakebase PostgreSQL)
- **Storage**: Unity Catalog (Delta tables, Lakebase)

---

## Memory Architecture

### Short-Term Memory (CheckpointSaver)
- **Scope**: Single conversation thread
- **Content**: Full message history, tool outputs
- **Lifetime**: Duration of conversation (minutes to hours)
- **Use case**: Multi-turn conversations within one session

### Long-Term Memory (DatabricksStore)
- **Scope**: Across threads, sessions, users
- **Content**: Facts, preferences, user profiles
- **Lifetime**: Indefinite (days to years)
- **Use case**: Personalization, learning from past interactions

---

## Need Help?

- Check MLflow traces for debugging
- Review error messages (version pins prevent most issues)
- Ensure Vector Search index and Lakebase are properly configured
- See Module 01 for Vector Search setup details

---

**Total Time**: 50 minutes from RAG agent to fully memory-enabled agent 🚀
