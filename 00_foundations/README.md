# Foundations - Platform Setup & Orientation

Set up your Databricks workspace and understand the platform in under 45 minutes!

## What You'll Learn

This module sets up the bootcamp infrastructure and introduces the Databricks platform:
1. **Notebook 00** (15 min): Setup - Create Unity Catalog assets, sample data, and Lakebase project
2. **Notebook 01** (10 min): Mosaic AI Gateway - LLM endpoints and pay-per-token vs provisioned
3. **Notebook 02** (15 min): Platform Orientation - MCP, Unity Catalog, ResponsesAgent vs ChatAgent

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
- Creating a Lakebase project and branches

**Time**: 15 minutes

---

### Next: 01_mosaic_gateway.py

**What you'll build**: Understanding of Databricks LLM endpoints

**Key concepts**:
- Mosaic AI Gateway architecture
- Listing available Foundation Model endpoints
- Testing chat completions with different models
- Pay-per-token vs provisioned throughput

**Time**: 10 minutes

---

### Next: 02_platform_orientation.py

**What you'll build**: Mental model for Databricks agent development

**Key concepts**:
- Comparing Databricks to AWS, Azure, Google Cloud
- Unity Catalog governance model
- Why MCP (Model Context Protocol) matters
- ResponsesAgent vs ChatAgent

**Time**: 15 minutes

---

## What's Next?

After completing these notebooks, you'll understand:
- ✅ How the Databricks workspace is configured for agents
- ✅ How to access LLMs through Mosaic AI Gateway
- ✅ Platform concepts: Unity Catalog, MCP, governance
- ✅ ResponsesAgent vs ChatAgent trade-offs

### Continue Learning

**For building agents** (covered in later modules):
- **Module 01**: RAG pipeline with Vector Search
- **Module 02**: Conversation memory with Lakebase
- **Module 03**: Evaluation and observability
- **Module 04**: Multi-tool agents (Genie, UC Functions)
- **Module 05**: Deployment to Databricks Apps

---

## Quick Start

1. Open `00_setup.py`
2. Run all cells — this creates catalog, schema, tables, documents, and Lakebase project
3. Wait for setup to complete before continuing
4. Continue to `01_mosaic_gateway.py` and `02_platform_orientation.py`

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

**Total Time**: 40 minutes from zero to platform-ready 🚀
