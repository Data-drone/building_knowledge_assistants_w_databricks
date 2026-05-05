# Module 0: Foundations

<span class="badge-duration">15 minutes</span>

Set up your Databricks workspace with the shared assets used by the rest of the bootcamp.

## What You'll Build

This module creates the foundational infrastructure:

- **Unity Catalog** catalog and schema
- **Sample data** — policy documents, employee tables, leave balances
- **Governed UC function** — `get_employee` for later MCP demos
- **Lakebase project** with development and production branches

## Notebook

| Notebook | Topics |
|---|---|
| [`00_setup.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/00_foundations/00_setup.py) | Unity Catalog, Lakebase, sample data, governed UC function |

## Prerequisites

- Databricks Runtime MLR 17.3 LTS or higher
- Unity Catalog enabled
- Permissions to create catalogs and schemas
- Lakebase access
- Foundation Model endpoints available

## Key Concepts

### Unity Catalog

All bootcamp assets live under a single catalog and schema (`agent_bootcamp.knowledge_assistant` by default). This gives you:

- Centralized governance and permissions
- Volume storage for source documents
- Delta tables for structured data
- UC functions for custom tools

### Lakebase

Lakebase is Databricks' managed PostgreSQL service with:

- **Scale-to-zero** — autoscales after 15 min idle
- **Branching** — copy-on-write branches (like Git for databases)
- **CheckpointSaver** — conversation memory
- **DatabricksStore** — long-term facts and preferences

The setup notebook creates a project with `development` and `production` branches.

## What's Next?

After completing this notebook, continue to [Module 1: RAG Pipeline](01-rag-pipeline.md) to build your first agent.
