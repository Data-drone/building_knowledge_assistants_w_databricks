# Agent Memory - Adding Conversation Context

Give your RAG agents memory in under 45 minutes!

## What You'll Learn

This module teaches you to add memory to AI agents using Databricks primitives:
1. **Notebook 01** (20 min): Short-term memory with CheckpointSaver - Multi-turn conversations
2. **Notebook 02** (25 min): Long-term memory with DatabricksStore - Cross-session personalization

These notebooks align with Databricks' latest stateful agent guidance:
[AI agent memory](https://docs.databricks.com/aws/en/generative-ai/agent-framework/stateful-agents)

## Prerequisites

### Required
- **Vector Search index** available (from Module 01 or workspace setup)
  - Expected index: `{CATALOG}.{SCHEMA}.policy_index`
  - If you haven't set this up, see [Module 01](../01_rag_pipeline/README.md) or workspace setup guide
- **Lakebase target** created (usually done in workspace setup)
  - Provisioned: `LAKEBASE_INSTANCE_NAME`
  - Autoscaling: `LAKEBASE_AUTOSCALING_PROJECT` + `LAKEBASE_AUTOSCALING_BRANCH`
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

**Production note**: Use a stable `thread_id` per conversation when you deploy
the agent on Databricks Apps. Reusing the same `thread_id` resumes the session;
changing it starts a fresh conversation.
The short-term examples in this repo now support either a provisioned Lakebase
instance or an autoscaling Lakebase project/branch.

**Time**: 20 minutes

---

### Next: 02_long_term_memory.py

**What you'll build**: RAG agent with cross-session personalization

**Key concepts**:
- DatabricksStore for persistent facts
- Namespace organization for memory types
- Combining short-term + long-term memory
- Personalization across sessions
- Memory management and privacy
- Stable `user_id` patterns for cross-session recall

**Time**: 25 minutes

---

## What's Next?

After completing these notebooks, you'll understand:
- ✅ How to add memory to RAG agents
- ✅ Short-term vs long-term memory patterns
- ✅ Thread-based conversation management
- ✅ Cross-session personalization

### Continue Learning

**For deeper understanding**:
- Memory management strategies → See notebook sections
- Privacy and retention policies → See notebook sections
- Production deployment patterns → See notebook sections

**For production features** (covered in later modules):
- **Module 03**: Evaluation and observability
- **Module 04**: Multi-tool agents (Genie, UC Functions)
- **Module 05**: Deployment to Databricks Apps with explicit `thread_id` handling

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
- **Config**: either `LAKEBASE_INSTANCE_NAME` or `LAKEBASE_AUTOSCALING_PROJECT` + `LAKEBASE_AUTOSCALING_BRANCH`

### Long-Term Memory (DatabricksStore)
- **Scope**: Across threads, sessions, users
- **Content**: Facts, preferences, user profiles
- **Lifetime**: Indefinite (days to years)
- **Use case**: Personalization, learning from past interactions
- **Config**: same Lakebase target pattern as short-term memory, plus embedding endpoint settings

---

## Need Help?

- Check MLflow traces for debugging
- Review error messages (version pins prevent most issues)
- Ensure Vector Search index and Lakebase are properly configured
- On Databricks Apps, ensure the app role has `CREATE` on schema `public` for checkpoint tables
- For long-term memory, also grant access to `public.store`, `public.store_vectors`, `public.store_migrations`, and `public.vector_migrations`
- See Module 01 for Vector Search setup details

---

**Total Time**: 45 minutes from RAG agent to fully memory-enabled agent 🚀
