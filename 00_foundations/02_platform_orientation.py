# Databricks notebook source
# MAGIC %md
# MAGIC # Platform Orientation: Databricks for Agent Developers
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Compare Databricks to other agent development platforms
# MAGIC - Understand Unity Catalog's governance model
# MAGIC - Learn why MCP (Model Context Protocol) matters
# MAGIC - Distinguish between ResponsesAgent and ChatAgent
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed [01_mosaic_gateway.py](01_mosaic_gateway.py)
# MAGIC - Familiarity with agents (LangChain, LangGraph, or similar)
# MAGIC
# MAGIC ## Estimated Time
# MAGIC 15 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Databricks vs Other Agent Platforms
# MAGIC
# MAGIC If you're coming from AWS, Azure, or open-source agent frameworks, here's how Databricks compares:
# MAGIC
# MAGIC | Component | AWS | Azure | Google Cloud | Databricks |
# MAGIC |-----------|-----|-------|--------------|------------|
# MAGIC | **LLM Access** | Bedrock | Azure OpenAI | Vertex AI | Mosaic AI Gateway |
# MAGIC | **Vector Store** | OpenSearch | AI Search | Vertex Matching | Vector Search |
# MAGIC | **Functions/Tools** | Lambda | Functions | Cloud Functions | MCP + UC Functions |
# MAGIC | **State Storage** | DynamoDB | CosmosDB | Firestore | Lakebase (PostgreSQL) |
# MAGIC | **Governance** | IAM | RBAC | IAM | Unity Catalog |
# MAGIC | **Observability** | CloudWatch | App Insights | Cloud Trace | MLflow |
# MAGIC | **Deployment** | ECS/Lambda | Container Apps | Cloud Run | Databricks Apps |
# MAGIC
# MAGIC ### Key Differentiators
# MAGIC
# MAGIC 1. **Unified Governance**: Unity Catalog governs ALL assets (data, models, functions)
# MAGIC 2. **Built-in Lineage**: Automatic tracking from data → features → models → agents
# MAGIC 3. **Delta Lake Foundation**: Time travel, ACID transactions, versioning for free
# MAGIC 4. **Native Evaluation**: MLflow judges and scorers integrated into platform

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Unity Catalog: Governance for Agents
# MAGIC
# MAGIC Unity Catalog provides fine-grained access control for all agent components.
# MAGIC
# MAGIC ### Three-Level Namespace
# MAGIC
# MAGIC ```
# MAGIC catalog.schema.asset
# MAGIC   │      │      │
# MAGIC   │      │      └─ Tables, Functions, Models, Volumes
# MAGIC   │      └──────── Logical grouping (e.g., "hr_assistant")
# MAGIC   └─────────────── Organizational boundary (e.g., "production")
# MAGIC ```
# MAGIC
# MAGIC ### Permissions Model
# MAGIC
# MAGIC Unlike IAM or RBAC, Unity Catalog permissions are:
# MAGIC - **Hierarchical**: GRANT on catalog flows down to schemas/tables
# MAGIC - **Asset-specific**: Different permissions for SELECT, EXECUTE, MODIFY
# MAGIC - **Audited**: All access logged automatically
# MAGIC
# MAGIC ### Example: Agent Access Control
# MAGIC
# MAGIC ```sql
# MAGIC -- Grant agent service principal read access to docs
# MAGIC GRANT SELECT ON TABLE production.knowledge_base.documents
# MAGIC   TO `agent-service-principal`;
# MAGIC
# MAGIC -- Grant execute permission on UC function
# MAGIC GRANT EXECUTE ON FUNCTION production.tools.get_employee_info
# MAGIC   TO `agent-service-principal`;
# MAGIC
# MAGIC -- Grant write access to conversation history
# MAGIC GRANT MODIFY ON TABLE production.agent_state.conversations
# MAGIC   TO `agent-service-principal`;
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Why MCP (Model Context Protocol)?
# MAGIC
# MAGIC MCP is Databricks' standard for exposing capabilities as tools for agents.
# MAGIC
# MAGIC ### Traditional Tool Definition (LangChain)
# MAGIC
# MAGIC ```python
# MAGIC # Manual tool definition
# MAGIC from langchain.tools import tool
# MAGIC
# MAGIC @tool
# MAGIC def search_documents(query: str) -> str:
# MAGIC     """Search internal documentation."""
# MAGIC     # You write all the code:
# MAGIC     # - Authentication
# MAGIC     # - Permission checks
# MAGIC     # - Error handling
# MAGIC     # - Logging
# MAGIC     results = vector_index.search(query)  # What if user lacks SELECT permission?
# MAGIC     return results
# MAGIC ```
# MAGIC
# MAGIC **Problems:**
# MAGIC - Permissions NOT enforced at tool boundary
# MAGIC - No audit trail of tool calls
# MAGIC - Agent can bypass governance
# MAGIC
# MAGIC ### MCP-Based Tools (Databricks)
# MAGIC
# MAGIC ```python
# MAGIC from databricks_mcp import DatabricksMCPClient
# MAGIC
# MAGIC # Connect to Vector Search MCP
# MAGIC client = DatabricksMCPClient(
# MAGIC     "https://workspace.databricks.com/api/2.0/mcp/vector-search/catalog/schema",
# MAGIC     auth_provider  # Uses user's credentials!
# MAGIC )
# MAGIC
# MAGIC tools = client.get_tools()  # Permissions flow through automatically
# MAGIC ```
# MAGIC
# MAGIC **Benefits:**
# MAGIC - ✅ Unity Catalog permissions enforced on every call
# MAGIC - ✅ Automatic audit logs
# MAGIC - ✅ Consistent error handling
# MAGIC - ✅ Built-in rate limiting
# MAGIC - ✅ No custom permission logic needed
# MAGIC
# MAGIC ### Available MCP Endpoints
# MAGIC
# MAGIC | MCP Type | URL Pattern | What It Exposes |
# MAGIC |----------|-------------|-----------------|
# MAGIC | Vector Search | `/api/2.0/mcp/vector-search/{catalog}/{schema}` | Similarity search on indexes |
# MAGIC | Genie | `/api/2.0/mcp/genie/{space_id}` | Natural language → SQL queries |
# MAGIC | UC Functions | `/api/2.0/mcp/functions/{catalog}/{schema}` | Execute registered SQL/Python UDFs |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. ResponsesAgent vs ChatAgent
# MAGIC
# MAGIC MLflow 3.0 introduced `ResponsesAgent` as the recommended base class.
# MAGIC
# MAGIC ### ChatAgent (Legacy, Still Supported)
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC
# MAGIC class MyAgent(ChatAgent):
# MAGIC     def predict(self, messages: List[ChatMessage]) -> ChatMessage:
# MAGIC         # Returns single ChatMessage
# MAGIC         return ChatMessage(role="assistant", content="Hello")
# MAGIC ```
# MAGIC
# MAGIC **Limitations:**
# MAGIC - Simple request/response only
# MAGIC - No multi-agent support
# MAGIC - Limited tool call tracking
# MAGIC - Basic streaming
# MAGIC
# MAGIC ### ResponsesAgent (Recommended)
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.agent import ResponsesAgentRequest, ResponsesAgentResponse
# MAGIC
# MAGIC class MyAgent(ResponsesAgent):
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         # Richer request/response types
# MAGIC         return ResponsesAgentResponse(
# MAGIC             content="Hello",
# MAGIC             metadata={"model": "claude-4", "tools_used": ["search"]}
# MAGIC         )
# MAGIC ```
# MAGIC
# MAGIC **Advantages:**
# MAGIC - ✅ OpenAI Responses API compatible
# MAGIC - ✅ Multi-agent orchestration support
# MAGIC - ✅ Better tool call history tracking
# MAGIC - ✅ Streaming with structured events
# MAGIC - ✅ Thread-based conversations (`thread_id`)
# MAGIC - ✅ Future-proof for new features
# MAGIC
# MAGIC ### When to Use Each
# MAGIC
# MAGIC | Use ResponsesAgent | Use ChatAgent |
# MAGIC |-------------------|---------------|
# MAGIC | New agents | Legacy migrations |
# MAGIC | Production deployments | Simple demos |
# MAGIC | Multi-agent systems | Single-turn QA |
# MAGIC | Complex tool calling | Basic chatbots |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Agent Architecture Patterns on Databricks
# MAGIC
# MAGIC ### Pattern 1: RAG Agent (Document Search)
# MAGIC
# MAGIC ```
# MAGIC User Query → ResponsesAgent → Vector Search MCP → Delta Table
# MAGIC                ↓
# MAGIC              Mosaic AI Gateway (LLM)
# MAGIC                ↓
# MAGIC              Response
# MAGIC ```
# MAGIC
# MAGIC **When to use:** FAQ bots, documentation assistants, policy Q&A
# MAGIC
# MAGIC ### Pattern 2: Data Agent (SQL Generation)
# MAGIC
# MAGIC ```
# MAGIC User Query → ResponsesAgent → Genie MCP → Delta Tables
# MAGIC                ↓
# MAGIC              Mosaic AI Gateway
# MAGIC                ↓
# MAGIC              Structured Answer
# MAGIC ```
# MAGIC
# MAGIC **When to use:** Business intelligence, analytics, reporting
# MAGIC
# MAGIC ### Pattern 3: Multi-Tool Agent (Hybrid)
# MAGIC
# MAGIC ```
# MAGIC User Query → ResponsesAgent → [Tool Router] → Vector Search MCP
# MAGIC                ↓                  ↓           → Genie MCP
# MAGIC              LLM                  ↓           → UC Functions
# MAGIC                ↓                  ↓
# MAGIC              Response ← [Tool Executor]
# MAGIC ```
# MAGIC
# MAGIC **When to use:** Complex assistants requiring multiple data sources
# MAGIC
# MAGIC ### Pattern 4: Stateful Agent (with Memory)
# MAGIC
# MAGIC ```
# MAGIC User Query → ResponsesAgent → Lakebase (PostgresSaver)
# MAGIC                ↓                      ↓
# MAGIC              LLM + Tools        Conversation History
# MAGIC                ↓                      ↓
# MAGIC              Response          Updated State
# MAGIC ```
# MAGIC
# MAGIC **When to use:** Multi-turn conversations, personalized assistants

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Install Required Packages

# COMMAND ----------

# MAGIC %pip install --upgrade langchain databricks-mcp mcp

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Demo: Compare Tool Definition Approaches

# COMMAND ----------

import sys
sys.path.append("..")

from config import get_workspace_client, CATALOG, SCHEMA

# COMMAND ----------

# Approach 1: Manual tool (NO governance)
from langchain.tools import tool

@tool
def manual_employee_lookup(employee_id: int) -> str:
    """Look up employee info by ID."""
    # Problem: This bypasses Unity Catalog permissions!
    # If user doesn't have SELECT on employee_data, this still works
    df = spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.employee_data WHERE employee_id = {employee_id}")
    return df.collect()[0].asDict()

# Test manual tool
print("Manual Tool (NO governance):")
result = manual_employee_lookup.invoke({"employee_id": 1001})
print(f"  Result: {result}")
print("  ⚠️  No permission check! Tool bypasses Unity Catalog.\n")

# COMMAND ----------

# Approach 2: UC Function (WITH governance)
# First, create a UC function

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.get_employee(emp_id INT)
RETURNS STRING
LANGUAGE SQL
COMMENT 'Get employee info by ID (governed)'
RETURN (
  SELECT CONCAT(name, ' - ', title, ' (', department, ')')
  FROM {CATALOG}.{SCHEMA}.employee_data
  WHERE employee_id = emp_id
  LIMIT 1
)
""")

print(f"✓ Created UC Function: {CATALOG}.{SCHEMA}.get_employee")

# COMMAND ----------

# Now expose via MCP
from databricks_mcp import DatabricksMCPClient, DatabricksOAuthClientProvider
from mcp.client.streamable_http import streamablehttp_client
from config import get_mcp_endpoint_url

# Connect to UC Functions MCP
uc_mcp_url = get_mcp_endpoint_url("uc_functions", catalog=CATALOG, schema=SCHEMA)

print(f"Connecting to UC Functions MCP: {uc_mcp_url}")

client = DatabricksMCPClient(
    uc_mcp_url,
    DatabricksOAuthClientProvider(),
    streamablehttp_client()
)

# Get governed tools
uc_tools = client.get_tools()
print(f"\n✓ Retrieved {len(uc_tools)} governed tools from MCP")

for tool in uc_tools:
    print(f"  - {tool.name}: {tool.description}")

print("\n✅ These tools enforce Unity Catalog permissions on every call!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Key Concepts Summary
# MAGIC
# MAGIC ### Unity Catalog
# MAGIC - Governs data, functions, models, and MCP endpoints
# MAGIC - Hierarchical permissions with automatic audit logs
# MAGIC - Fine-grained access control (SELECT, EXECUTE, MODIFY)
# MAGIC
# MAGIC ### MCP (Model Context Protocol)
# MAGIC - Standard interface for exposing Databricks services to agents
# MAGIC - Enforces Unity Catalog permissions automatically
# MAGIC - Provides consistent error handling and logging
# MAGIC
# MAGIC ### ResponsesAgent
# MAGIC - Recommended base class for agents (vs ChatAgent)
# MAGIC - Supports complex workflows and multi-agent systems
# MAGIC - Compatible with OpenAI Responses API
# MAGIC
# MAGIC ### Lakebase
# MAGIC - Managed PostgreSQL for agent state
# MAGIC - Autoscaling with scale-to-zero
# MAGIC - Branch-based development (like Git for databases)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC You've learned:
# MAGIC
# MAGIC - ✅ How Databricks compares to AWS/Azure/GCP for agents
# MAGIC - ✅ Unity Catalog's governance model for agent components
# MAGIC - ✅ Why MCP enforces permissions better than manual tools
# MAGIC - ✅ When to use ResponsesAgent vs ChatAgent
# MAGIC - ✅ Common agent architecture patterns
# MAGIC
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC 1. **Always use MCP tools** - they enforce Unity Catalog permissions
# MAGIC 2. **Choose ResponsesAgent** for new agents (more features, future-proof)
# MAGIC 3. **Leverage Unity Catalog** - governance is built-in, not bolt-on
# MAGIC 4. **Lakebase for state** - better than managing PostgreSQL yourself
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC Continue to [01_rag_pipeline/01_data_prep.py](../01_rag_pipeline/01_data_prep.py) to build your first RAG agent.
