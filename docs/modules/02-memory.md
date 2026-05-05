# Module 2: Memory

<span class="badge-duration">50 minutes</span>

Add conversation memory so your agent remembers what was said earlier in a conversation and what it learned about users across sessions.

## What You'll Build

A RAG agent with two types of memory:

- **Short-term memory** — `CheckpointSaver` keeps conversation history within a thread
- **Long-term memory** — `DatabricksStore` persists user facts and preferences across sessions

## Notebooks

| Notebook | Topics | Duration |
|---|---|---|
| [`01_short_term_memory.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/02_memory/01_short_term_memory.py) | CheckpointSaver, thread memory, multi-turn conversations | 20 min |
| [`02_long_term_memory.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/02_memory/02_long_term_memory.py) | DatabricksStore, user preferences, LLM-managed memory tools | 30 min |

## Prerequisites

- Vector Search index from [Module 1](01-rag-pipeline.md) (`{CATALOG}.{SCHEMA}.policy_index`)
- Lakebase instance created (from [Module 0](00-foundations.md))
- LLM endpoint available

## Key Concepts

### Short-Term Memory (CheckpointSaver)

| Property | Value |
|---|---|
| **Scope** | Single conversation thread |
| **Content** | Full message history and tool outputs |
| **Lifetime** | Duration of conversation (minutes to hours) |
| **Key** | `thread_id` |

The `CheckpointSaver` stores the full LangGraph state after each step. When a new message arrives with the same `thread_id`, the agent resumes with full context.

The notebook demonstrates a "before/after" comparison — without memory, the agent forgets what you just asked. With `CheckpointSaver`, follow-up questions work naturally.

### Long-Term Memory (DatabricksStore)

| Property | Value |
|---|---|
| **Scope** | Across threads, sessions, and time |
| **Content** | Facts, preferences, user profiles |
| **Lifetime** | Indefinite (days to years) |
| **Key** | `user_id` namespace |

The second notebook teaches long-term memory in two stages:

1. **Manual memory** (put/get) — understand the primitives
2. **LLM-managed memory tools** — the production pattern where the agent decides what to remember

### LLM-Managed Memory Tools

The production pattern uses three tools that the agent calls autonomously:

- `get_user_memory` — semantic search over stored memories
- `save_user_memory` — persist a new fact about the user
- `delete_user_memory` — remove a memory (GDPR compliance)

The store and `user_id` are passed through `RunnableConfig`, so the agent's memory is scoped per-user without hardcoding.

### Lakebase Targets

Short-term and long-term memory both connect to Lakebase. The notebooks support two connection patterns:

=== "Autoscaling (recommended)"

    ```python
    LAKEBASE_AUTOSCALING_PROJECT = "knowledge-assistant-state"
    LAKEBASE_AUTOSCALING_BRANCH = "development"
    ```

=== "Provisioned instance"

    ```python
    LAKEBASE_INSTANCE_NAME = "my-lakebase-instance"
    ```

## What You'll Understand

- Short-term vs long-term memory patterns
- Thread-based conversation management
- LLM-managed memory tools (the production pattern)
- Semantic search over memories vs exact key lookups
- Memory deletion for user control and GDPR compliance

## What's Next?

Continue to [Module 3: Evaluation](03-evaluation.md) to add observability and quality checks to your agent.
