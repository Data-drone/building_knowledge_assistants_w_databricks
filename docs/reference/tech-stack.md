# Tech Stack

## Component Overview

| Component | Technology | Purpose |
|---|---|---|
| **Agent Framework** | MLflow ResponsesAgent + LangGraph | Agent orchestration and execution |
| **LLM** | Mosaic AI Gateway (Claude Sonnet 4.6) | Reasoning and generation |
| **Vector Search** | Databricks Vector Search (Delta Sync) | Document similarity search |
| **Structured Data** | Genie | Natural language to SQL |
| **Memory** | Lakebase (PostgreSQL) | Conversation state (checkpointer + store) |
| **Observability** | MLflow 3.0+ | Tracing, evaluation, judges |
| **Governance** | Unity Catalog | Permissions, audit logs |
| **Deployment** | Databricks Apps | Production app endpoint |

## Agent Framework

### ResponsesAgent

MLflow's recommended agent base class:

- OpenAI Responses API compatibility
- Multi-agent orchestration support
- Advanced streaming with events
- Thread-based conversation management

### LangGraph

Used for the core agent loop:

- `StateGraph` with `MessagesState`
- `ToolNode` for tool execution
- Conditional edges for routing
- `CheckpointSaver` integration for memory

## Model Context Protocol (MCP)

Databricks' standard for exposing services as agent tools:

| MCP Server | Service | Governance |
|---|---|---|
| Vector Search MCP | Governed similarity search | UC permissions on index |
| Genie MCP | Natural language to SQL | UC permissions on tables |
| UC Functions MCP | Execute registered functions | UC permissions on function |

MCP ensures Unity Catalog permissions flow through automatically to all tool calls.

## MLflow 3.0 Evaluation

| Feature | Description |
|---|---|
| **Scorers** | Built-in: `Correctness`, `Safety`, `Guidelines`, `ToolCallCorrectness` |
| **`make_judge()`** | Custom LLM judges with categorical/boolean/numeric feedback |
| **`@scorer`** | Code-based scorers for deterministic checks |
| **Agent-as-a-Judge** | Analyze traces with `{{ trace }}` variable |
| **Registered scorers** | Continuous production monitoring with sampling |

## Key Python Packages

| Package | Version Range | Purpose |
|---|---|---|
| `mlflow[databricks]` | `>=3.10,<3.11` | Tracing, evaluation, serving |
| `langgraph` | `>=0.3` | Agent orchestration |
| `langchain-core` | `>=0.3` | LLM abstractions |
| `databricks-langchain` | `>=0.5` | Databricks integrations |
| `databricks-agents` | `>=0.16` | Agent framework |
