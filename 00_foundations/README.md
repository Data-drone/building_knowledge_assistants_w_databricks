# Foundations - Workspace Setup

Set up your Databricks workspace in under 15 minutes.

## What You'll Learn

This module sets up the bootcamp infrastructure used by the rest of the repo:
1. **Notebook 00** (15 min): Setup - Create Unity Catalog assets, sample data, governed UC function, and Lakebase project

## Prerequisites

- Databricks Runtime MLR 17.3 LTS or higher
- Unity Catalog enabled
- Permissions to create catalogs/schemas
- Lakebase access
- Foundation Model endpoints available

## Learning Path

### Start Here: 00_setup.py

**What you'll build**: Foundational infrastructure for the entire bootcamp

**Key concepts**:
- Creating Unity Catalog catalog and schema
- Uploading sample policy documents to a Volume
- Creating employee and leave balance tables
- Creating a governed UC function for later MCP demos
- Creating a Lakebase project and branches

**Time**: 15 minutes

---

## What's Next?

After completing this notebook, you'll have:
- ✅ How the Databricks workspace is configured for agents
- ✅ Core assets for the RAG, memory, and MCP modules
- ✅ A governed UC function ready for later tool demos

### Continue Learning

**For building agents** (covered in later modules):
- **Module 01**: RAG pipeline with Vector Search
- **Module 02**: Conversation memory with Lakebase
- **Module 03**: Evaluation and observability
- **Module 04**: Extending with data tools (SQL, Genie, custom tools)
- **Module 05**: Deployment to Databricks Apps

---

## Quick Start

1. Open `00_setup.py`
2. Run all cells — this creates catalog, schema, tables, documents, and Lakebase project
3. Wait for setup to complete before continuing
4. Continue to `../01_rag_pipeline/00_your_first_agent_on_databricks.py`

---

## Technical Stack

- **LLM**: Databricks LLM endpoints (Claude Sonnet 4.6)
- **Storage**: Unity Catalog (Delta tables, Volumes)
- **State**: Lakebase (PostgreSQL)
- **Governance**: Unity Catalog permissions

---

## Need Help?

- Check that Unity Catalog is enabled for your workspace
- Verify you have permissions to create catalogs and schemas
- Ensure Lakebase is available in your region

---

**Total Time**: 15 minutes from zero to platform-ready 🚀
