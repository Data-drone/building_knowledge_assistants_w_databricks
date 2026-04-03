# Databricks notebook source
# MAGIC %md
# MAGIC # Module 04: Extending Your Agent with Data Tools - Setup & Prerequisites
# MAGIC
# MAGIC ## Module Overview
# MAGIC
# MAGIC In this module, you'll learn how to give your agent access to **multiple data sources**:
# MAGIC
# MAGIC **What You'll Build:**
# MAGIC 1. **RAG Agent** - Answers questions from policy documents (Vector Search)
# MAGIC 2. **+ SQL Tool** - Query structured data tables directly
# MAGIC 3. **+ Genie MCP** - Natural language to SQL (no SQL writing needed!)
# MAGIC 4. **+ Custom Tools** - Deploy your own MCP servers for external APIs
# MAGIC
# MAGIC **Why MCP (Model Context Protocol)?**
# MAGIC - Standardized way to expose tools to agents
# MAGIC - Unity Catalog permissions flow through automatically
# MAGIC - Governance and audit trails
# MAGIC
# MAGIC In this repo, MCP becomes important once the agent needs access to governed data tools.
# MAGIC The setup notebook already created the employee tables and a `get_employee` UC function;
# MAGIC this module shows how to expose those kinds of assets to the agent in a controlled way.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - ✅ Understand when to use Vector Search vs SQL vs Genie
# MAGIC - ✅ Build multi-tool agents that route intelligently
# MAGIC - ✅ Create and deploy custom MCP servers
# MAGIC - ✅ Handle complex queries requiring multiple tools
# MAGIC
# MAGIC ## Estimated Time
# MAGIC - **Notebook 01 (Genie Integration)**: 45 minutes
# MAGIC - **Notebook 02 (Custom Tools)**: 30 minutes
# MAGIC - **Total**: 75 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies
# MAGIC
# MAGIC Same core stack from earlier modules plus the Databricks SDK for Genie space creation.

# COMMAND ----------

%pip install -q --upgrade \
  "databricks-sdk>=0.101" \
  "mlflow[databricks]>=3.10" \
  "databricks-langchain[memory]>=0.17" \
  "databricks-vectorsearch>=0.66" \
  "langgraph>=1.1"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Verify Core Setup
# MAGIC
# MAGIC Check that foundational assets from Module 00 exist.
# MAGIC These assets replace the need for a separate platform-orientation notebook because the
# MAGIC governance story is introduced here, where the tools are actually used.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# ── Configuration ── edit these to match your workspace ──
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"

DOCS_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/source_docs"
EMPLOYEE_TABLE = f"{CATALOG}.{SCHEMA}.employee_data"
LEAVE_BALANCES_TABLE = f"{CATALOG}.{SCHEMA}.leave_balances"

VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.policy_index"
ENDPOINT_NAME = "one-env-shared-endpoint-15"  # Your Vector Search endpoint

LAKEBASE_PROJECT = "knowledge-assistant-state"
SQL_WAREHOUSE_ID = ""  # Your SQL warehouse ID (find in SQL Warehouses UI)

w = WorkspaceClient()

print("=" * 80)
print("PREREQUISITES CHECK")
print("=" * 80)

# Check catalog and schema
try:
    spark.sql(f"USE CATALOG {CATALOG}")
    spark.sql(f"USE SCHEMA {SCHEMA}")
    print(f"✅ Catalog and schema ready: {CATALOG}.{SCHEMA}")
except Exception as e:
    print(f"❌ Catalog/Schema missing: {e}")
    print("\n⚠️  You need to run: 00_foundations/00_setup.py first!")
    dbutils.notebook.exit("Prerequisites not met")

# Check employee tables
try:
    emp_count = spark.table(EMPLOYEE_TABLE).count()
    leave_count = spark.table(LEAVE_BALANCES_TABLE).count()
    print(f"✅ Employee tables ready:")
    print(f"   - {EMPLOYEE_TABLE}: {emp_count} rows")
    print(f"   - {LEAVE_BALANCES_TABLE}: {leave_count} rows")
except Exception as e:
    print(f"❌ Employee tables missing: {e}")
    print("\n⚠️  You need to run: 00_foundations/00_setup.py first!")
    dbutils.notebook.exit("Prerequisites not met")

# Check source documents
try:
    files = dbutils.fs.ls(DOCS_VOLUME)
    print(f"✅ Source documents ready: {len(files)} files in {DOCS_VOLUME}")
except Exception as e:
    print(f"❌ Source documents missing: {e}")
    print("\n⚠️  You need to run: 00_foundations/00_setup.py first!")
    dbutils.notebook.exit("Prerequisites not met")

# Check Lakebase project
try:
    project = w.lakebase_projects.get(name=LAKEBASE_PROJECT)
    print(f"✅ Lakebase project ready: {LAKEBASE_PROJECT}")
except Exception as e:
    print(f"⚠️  Lakebase project: {e}")
    print("   (This is OK - Lakebase is optional for Module 04)")

print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Verify Vector Search Assets
# MAGIC
# MAGIC Check that Vector Search endpoint and index from Module 01 exist.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

# Check Vector Search endpoint
try:
    endpoint = vsc.get_endpoint(ENDPOINT_NAME)
    print(f"✅ Vector Search endpoint ready: {ENDPOINT_NAME}")
    print(f"   Status: {endpoint.get('endpoint_status', {}).get('state', 'Unknown')}")
except Exception as e:
    print(f"❌ Vector Search endpoint missing: {e}")
    print("\n⚠️  You need to run: 01_rag_pipeline/01_building_a_doc_store_on_vector_search.py first!")
    print("\nModule 04 requires Vector Search to be set up.")
    dbutils.notebook.exit("Vector Search endpoint not found")

# Check vector index
try:
    index = vsc.get_index(ENDPOINT_NAME, VECTOR_INDEX)
    print(f"✅ Vector index ready: {VECTOR_INDEX}")

    # Test a quick search
    results = index.similarity_search(
        query_text="vacation policy",
        columns=["chunk_text"],
        num_results=1
    )
    if results.get("result", {}).get("data_array"):
        print(f"   ✓ Index is searchable")
    else:
        print(f"   ⚠️  Index exists but may be empty")

except Exception as e:
    print(f"❌ Vector index missing: {e}")
    print("\n⚠️  You need to run: 01_rag_pipeline/01_building_a_doc_store_on_vector_search.py first!")
    dbutils.notebook.exit("Vector index not found")

print("\n✅ Vector Search assets verified!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create Genie Space
# MAGIC
# MAGIC Create a Genie space that will enable natural language queries over the employee tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ### What is Genie?
# MAGIC
# MAGIC **Genie** is Databricks' natural language to SQL service:
# MAGIC - Users ask questions in plain English
# MAGIC - Genie generates and executes SQL
# MAGIC - Returns results in natural language
# MAGIC
# MAGIC **Why use Genie?**
# MAGIC - No SQL knowledge required
# MAGIC - Automatic table schema understanding
# MAGIC - Built-in data governance (Unity Catalog permissions)
# MAGIC - Query optimization
# MAGIC
# MAGIC **Genie Space:**
# MAGIC - A "space" contains:
# MAGIC   - One or more tables
# MAGIC   - Instructions for the AI
# MAGIC   - Query history and refinements

# COMMAND ----------

import json, uuid

if not SQL_WAREHOUSE_ID:
    raise ValueError(
        "SQL_WAREHOUSE_ID is required to create a Genie space. "
        "Find it in the Databricks UI under SQL Warehouses and paste it into the "
        "SQL_WAREHOUSE_ID variable at the top of this notebook."
    )

# Build the serialized_space payload — tables, instructions, and sample questions
# in a single JSON structure (version 2 format).
serialized_space = json.dumps({
    "version": 2,
    "data_sources": {
        "tables": [
            {"identifier": EMPLOYEE_TABLE, "description": ["Employee information: names, departments, roles, hire dates"]},
            {"identifier": LEAVE_BALANCES_TABLE, "description": ["Leave balances: vacation and sick days (total vs used)"]},
        ]
    },
    "instructions": {
        "text_instructions": [
            {
                "id": uuid.uuid4().hex,
                "content": [
                    "This space contains employee HR data. "
                    "Answer questions about employee counts by department, "
                    "leave balances (vacation and sick days), and employee information. "
                    "Be concise and specific, include relevant employee names, "
                    "and calculate remaining leave days as (total - used)."
                ],
            }
        ]
    },
    "config": {
        "sample_questions": [
            {"id": uuid.uuid4().hex, "question": ["How many employees are in each department?"]},
            {"id": uuid.uuid4().hex, "question": ["Who has the most vacation days remaining?"]},
        ]
    },
})

try:
    space = w.genie.create_space(
        warehouse_id=SQL_WAREHOUSE_ID,
        serialized_space=serialized_space,
        title=f"{SCHEMA}_employee_data",
        description="Employee data for the Agent Bootcamp — HR questions via natural language.",
    )

    space_id = space.space_id
    print(f"✅ Created Genie space: {space_id}")
    print(f"   Tables: {EMPLOYEE_TABLE}, {LEAVE_BALANCES_TABLE}")
    print(f"\n📝 Paste this Genie space ID into 01_genie_integration.py:")
    print(f'   GENIE_SPACE_ID = "{space_id}"')

except Exception as e:
    print(f"❌ Failed to create Genie space: {e}")
    print("\nNote: You may need Genie access. Check with your workspace admin.")
    print("You can still continue with Module 04 - we'll work around this.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Module 04 Learning Path
# MAGIC
# MAGIC Now that prerequisites are verified, here's what you'll learn:
# MAGIC
# MAGIC ### Notebook 01: Genie Integration (45 min)
# MAGIC
# MAGIC **Learning Flow:**
# MAGIC 1. **Start**: RAG agent (Vector Search only)
# MAGIC 2. **Problem**: Can't answer structured data questions
# MAGIC 3. **Solution 1**: Add SQL MCP tool (manual SQL)
# MAGIC 4. **Solution 2**: Add Genie MCP tool (natural language)
# MAGIC 5. **Result**: Multi-tool agent that routes intelligently
# MAGIC
# MAGIC **Key Concepts:**
# MAGIC - When to use Vector Search vs SQL vs Genie
# MAGIC - Tool routing and selection
# MAGIC - Combining multiple tools in one agent
# MAGIC - Handling complex queries
# MAGIC
# MAGIC ### Notebook 02: Custom Tools & Deployment (30 min)
# MAGIC
# MAGIC **What You'll Build:**
# MAGIC - Simple custom tools wrapping external APIs
# MAGIC - FastMCP server deployment
# MAGIC - Production MCP server as Databricks App
# MAGIC - Integration with agents
# MAGIC
# MAGIC **Key Concepts:**
# MAGIC - When to build custom tools
# MAGIC - MCP server architecture
# MAGIC - Databricks Apps deployment
# MAGIC - OAuth authentication for MCP servers

# COMMAND ----------

# MAGIC %md
# MAGIC ## Complete Architecture Preview
# MAGIC
# MAGIC After Module 04, your agent will have access to:
# MAGIC
# MAGIC ```
# MAGIC User Query: "Who in Engineering has the most vacation days left?"
# MAGIC      ↓
# MAGIC   Agent (LangGraph + Claude Sonnet 4.6)
# MAGIC      ↓
# MAGIC ┌──────────────┬─────────────┬─────────────────┬──────────────┐
# MAGIC │              │             │                 │              │
# MAGIC │ Vector       │ SQL MCP     │ Genie MCP      │ Custom MCP   │
# MAGIC │ Search       │             │                 │              │
# MAGIC │              │             │                 │              │
# MAGIC │ "vacation    │ "SELECT     │ "Who in        │ "get_weather │
# MAGIC │  policy"     │ COUNT(*)    │  Engineering   │  for Tokyo"  │
# MAGIC │              │ FROM..."    │  has most..."  │              │
# MAGIC │              │             │                 │              │
# MAGIC └──────────────┴─────────────┴─────────────────┴──────────────┘
# MAGIC      ↓               ↓               ↓               ↓
# MAGIC  Documents       Tables          Tables        External APIs
# MAGIC  (policies)      (direct)        (NL→SQL)      (weather, etc)
# MAGIC ```
# MAGIC
# MAGIC **Tool Selection Logic:**
# MAGIC - **Vector Search**: Unstructured text, policies, documentation
# MAGIC - **SQL MCP**: Precise queries, you know the SQL
# MAGIC - **Genie MCP**: Complex queries, natural language, no SQL needed
# MAGIC - **Custom MCP**: External data, third-party APIs, custom logic

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Verification

# COMMAND ----------

print("=" * 80)
print("MODULE 04 SETUP COMPLETE!")
print("=" * 80)
print("\n✅ Prerequisites Verified:")
print(f"   - Unity Catalog: {CATALOG}.{SCHEMA}")
print(f"   - Employee tables: {EMPLOYEE_TABLE}, {LEAVE_BALANCES_TABLE}")
print(f"   - Source documents: {len(dbutils.fs.ls(DOCS_VOLUME))} files")
print(f"   - Vector Search endpoint: {ENDPOINT_NAME}")
print(f"   - Vector index: {VECTOR_INDEX}")
print(f"   - Lakebase project: {LAKEBASE_PROJECT}")

print("\n📚 Ready to Start:")
print("   1. Continue to: 01_genie_integration.py")
print("   2. Then: 02_custom_tools.py")

print("\n💡 Remember:")
print("   - Paste your Genie space ID into 01_genie_integration.py")
print("   - Each notebook builds on the previous one")
print("   - Test as you go!")

print("=" * 80)
