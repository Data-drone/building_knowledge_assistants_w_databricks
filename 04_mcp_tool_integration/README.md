# MCP Tool Integration - Multi-Tool Agents

Add data query capabilities and custom tools to your agent in under 75 minutes!

## What You'll Learn

This module teaches you to build multi-tool agents using Databricks MCP:
1. **Notebook 00** (10 min): Setup - Create Genie space and configure structured data
2. **Notebook 01** (35 min): Genie Integration - Natural language SQL, multi-tool routing
3. **Notebook 02** (30 min): Custom Tools - Build and deploy your own MCP servers

## Prerequisites

### Required
- **Vector Search index** available (from Module 01)
  - Expected index: `{CATALOG}.{SCHEMA}.policy_index`
- **Lakebase instance** created (from Module 00 setup)
- **Employee data tables** created (from Module 00 setup): `employee_data`, `leave_balances`
- LLM endpoint available (`databricks-claude-sonnet-4-6` or similar)

### Recommended
- Completed [Module 02 (Memory)](../02_memory/README.md) — this module builds on the memory-enabled RAG agent

## Learning Path

### Start Here: 00_setup.py

**What you'll build**: Genie space and prerequisites for structured data queries

**Key concepts**:
- When to use Vector Search vs SQL vs Genie
- Setting up Genie spaces for natural language data access
- Preparing structured data for agent queries

**Time**: 10 minutes

---

### Next: 01_genie_integration.py

**What you'll build**: Multi-tool agent with Vector Search + Genie

**Key concepts**:
- Starting with the RAG agent and discovering structured data limitations
- Adding SQL MCP tool (requires writing SQL)
- Adding Genie MCP tool (natural language to SQL)
- Multi-tool routing: agent decides which tool to use
- Combining documents and data queries in one agent

**Time**: 35 minutes

---

### Next: 02_custom_tools.py

**What you'll build**: Custom MCP tools deployed to production

**Key concepts**:
- When to build custom tools vs use built-in MCP
- Simple custom tool creation
- MCP server concepts and local testing
- Deploying custom MCP servers as Databricks Apps

**Time**: 30 minutes (Quick Start path) or 50 minutes (Deep Dive)

---

## What's Next?

After completing these notebooks, you'll understand:
- ✅ How to add multiple tools to agents
- ✅ Genie for natural language data queries
- ✅ Building and deploying custom MCP servers
- ✅ Multi-tool routing strategies
- ✅ Unity Catalog governance for tools

### Continue Learning

**For production deployment** (covered in the next module):
- **Module 05**: Deployment to Databricks Apps with monitoring

---

## Quick Start

1. Open `00_setup.py` and create the Genie space
2. Continue to `01_genie_integration.py`
3. Build a multi-tool agent with Vector Search + Genie
4. Continue to `02_custom_tools.py`
5. Create and deploy custom MCP tools

---

## Technical Stack

- **LLM**: Databricks LLM endpoints (Claude Sonnet 4.6)
- **Orchestration**: LangGraph
- **Vector Search**: Databricks Vector Search (for RAG)
- **Structured Data**: Genie MCP (natural language to SQL)
- **Custom Tools**: UC Functions MCP, custom MCP servers
- **Memory**: CheckpointSaver + DatabricksStore (from Module 02)
- **Governance**: Unity Catalog permissions

---

## Need Help?

- Check MLflow traces for debugging tool routing
- Verify Genie space is created and accessible
- Ensure MCP URL format is correct for your workspace
- Check Unity Catalog permissions on underlying tables

---

**Total Time**: 75 minutes from RAG agent to fully multi-tool agent 🚀
