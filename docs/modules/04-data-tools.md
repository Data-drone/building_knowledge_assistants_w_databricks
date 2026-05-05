# Module 4: Data Tools

<span class="badge-duration">75 minutes</span>

Extend your agent with structured data queries (Genie), SQL access, and custom MCP tools.

## What You'll Build

A multi-tool agent that intelligently routes between documents (Vector Search), data queries (Genie), and custom tools — all governed through Unity Catalog.

## Notebooks

| Notebook | Topics | Duration |
|---|---|---|
| [`00_setup.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/04_mcp_tool_integration/00_setup.py) | Genie space creation, sample data | 10 min |
| [`01_genie_integration.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/04_mcp_tool_integration/01_genie_integration.py) | Genie spaces, natural language SQL, multi-tool routing | 35 min |
| [`02_custom_tools.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/04_mcp_tool_integration/02_custom_tools.py) | Custom tools, MCP servers, advanced patterns | 30 min |

## Prerequisites

- Vector Search index from [Module 1](01-rag-pipeline.md)
- Lakebase instance from [Module 0](00-foundations.md)
- Employee data tables from Module 0 setup
- LLM endpoint available
- Recommended: completed [Module 2](02-memory.md)

## Key Concepts

### When to Use Which Tool

| Data Type | Tool | How It Works |
|---|---|---|
| Unstructured docs | Vector Search MCP | Similarity search over embeddings |
| Structured data (user writes SQL) | SQL MCP | Execute SQL directly |
| Structured data (natural language) | Genie MCP | Genie converts question to SQL |
| Custom logic | UC Functions MCP | Execute governed Python functions |

### Genie Spaces

Genie converts natural language questions to SQL queries against your tables. You create a Genie Space pointing at specific tables, and the agent queries it through the MCP server.

The notebook progression:

1. Start with the RAG agent — discover it can't answer data questions
2. Add SQL MCP tool — works but requires writing SQL
3. Add Genie MCP tool — natural language to SQL
4. Multi-tool routing — the agent decides which tool to use

### Model Context Protocol (MCP)

Databricks exposes services as MCP servers. The agent connects to these servers and the tools appear automatically:

```python
# Vector Search MCP
f"{HOST}/api/2.0/mcp/vector-search/{CATALOG}/{SCHEMA}"

# Genie MCP
f"{HOST}/api/2.0/mcp/genie/{GENIE_SPACE_ID}"

# UC Functions MCP
f"{HOST}/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}"
```

Unity Catalog permissions flow through automatically — no separate auth configuration needed.

### Custom MCP Servers

The third notebook covers building your own MCP tools:

- When to build custom tools vs use built-in MCP
- Creating tools with UC functions
- MCP server concepts and local testing
- Deploying custom MCP servers as Databricks Apps

## What You'll Understand

- How to add multiple tools to an agent
- Genie for natural language data queries
- Building and deploying custom MCP servers
- Multi-tool routing — letting the agent choose
- Unity Catalog governance for all tools

## What's Next?

Continue to [Module 5: Deployment](05-deployment.md) to deploy your agent to production as a Databricks App.
