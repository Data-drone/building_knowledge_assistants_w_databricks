# Configuration

All shared configuration lives in `config.py` at the repo root. Update these values to match your Databricks workspace.

## Core Settings

```python
# Unity Catalog namespace
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"

# Workspace
HOST = "https://your-workspace.cloud.databricks.com"
REGION = "us-west-2"  # Your cloud region

# Model endpoints
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
```

## Lakebase Settings

```python
# Project and branches
LAKEBASE_PROJECT = "knowledge-assistant-state"
LAKEBASE_BRANCH = "development"
LAKEBASE_PRODUCTION_BRANCH = "production"
```

Notebooks support two connection patterns:

| Pattern | Variables | When to Use |
|---|---|---|
| Autoscaling | `LAKEBASE_AUTOSCALING_PROJECT` + `LAKEBASE_AUTOSCALING_BRANCH` | Production (recommended) |
| Provisioned | `LAKEBASE_INSTANCE_NAME` | Fixed-size instances |

## Genie Settings

```python
# Set after creating the Genie space in Module 04
GENIE_SPACE_ID = ""
```

## Apps Deployment

```python
APP_NAME = "knowledge-assistant-agent-app"
APP_EXPERIMENT = "/Shared/knowledge_assistant_agent_app"
```

## Helper Functions

`config.py` provides utility functions used across notebooks:

| Function | Purpose |
|---|---|
| `get_workspace_client()` | Authenticated `WorkspaceClient` |
| `get_lakebase_connection_string()` | PostgreSQL connection string for Lakebase |
| `sanitize_namespace_id()` | Make user IDs safe for `DatabricksStore` namespaces |
| `get_mcp_endpoint_url()` | Build MCP endpoint URLs for Vector Search, Genie, UC Functions |
| `get_app_url()` | Get the deployed app's base URL |

## Per-Notebook Config

Later modules carry small local config blocks for teaching clarity. These override or extend `config.py` where needed, so you don't have to edit the shared config for every notebook.
