# Databricks notebook source
# MAGIC %md
# MAGIC # Module 04: MCP Tool Integration - Setup & Prerequisites
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

# COMMAND ----------

%pip install -q --upgrade databricks-sdk mlflow[databricks]>=3.1.0 databricks-langchain[memory]>=0.8.0 databricks-vectorsearch>=0.30 langgraph>=0.2.50

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Verify Core Setup
# MAGIC
# MAGIC Check that foundational assets from Module 00 exist.

# COMMAND ----------

import sys
sys.path.append("/Workspace" + "/".join(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[:-2]))

from config import *
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

print("=" * 80)
print("PREREQUISITES CHECK")
print("=" * 80)

# Check catalog and schema
try:
    spark.sql(f"USE CATALOG {CATALOG}")
    spark.sql(f"USE SCHEMA {SCHEMA}")
    print(f"✅ Catalog and schema ready: {UC_NAMESPACE}")
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

# Create Genie space for employee data
try:
    # Create space
    space = w.genie.create_message_query_result(
        space_name=f"{SCHEMA}_employee_data",
        description="Employee data including employee information and leave balances. Use this to answer HR-related questions.",
    )

    space_id = space.space_id
    print(f"✅ Created Genie space: {space_id}")

    # Add tables to the space
    w.genie.add_tables(
        space_id=space_id,
        table_full_names=[EMPLOYEE_TABLE, LEAVE_BALANCES_TABLE]
    )
    print(f"✅ Added tables to Genie space:")
    print(f"   - {EMPLOYEE_TABLE}")
    print(f"   - {LEAVE_BALANCES_TABLE}")

    # Add instructions
    instructions = """
    This space contains employee HR data. Use it to answer questions about:
    - Employee counts by department
    - Leave balances (vacation and sick days)
    - Employee information (names, roles, hire dates)

    When answering:
    - Be concise and specific
    - Include relevant employee names
    - Calculate remaining leave days (total - used)
    """

    w.genie.set_space_instructions(
        space_id=space_id,
        instructions=instructions
    )
    print(f"✅ Added instructions to Genie space")

    # Update config.py with space ID
    print(f"\n📝 Update config.py with this Genie space ID:")
    print(f"   GENIE_SPACE_ID = \"{space_id}\"")

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
# MAGIC   Agent (LangGraph + Claude Sonnet 4.5)
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
print(f"   - Unity Catalog: {UC_NAMESPACE}")
print(f"   - Employee tables: {EMPLOYEE_TABLE}, {LEAVE_BALANCES_TABLE}")
print(f"   - Source documents: {len(dbutils.fs.ls(DOCS_VOLUME))} files")
print(f"   - Vector Search endpoint: {ENDPOINT_NAME}")
print(f"   - Vector index: {VECTOR_INDEX}")
print(f"   - Lakebase project: {LAKEBASE_PROJECT}")

print("\n📚 Ready to Start:")
print("   1. Continue to: 01_genie_integration.py")
print("   2. Then: 02_custom_tools.py")

print("\n💡 Remember:")
print("   - Update config.py with your Genie space ID")
print("   - Each notebook builds on the previous one")
print("   - Test as you go!")

print("=" * 80)
