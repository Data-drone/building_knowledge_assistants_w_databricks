# Databricks notebook source
# MAGIC %md
# MAGIC # Module 04: Adding Data Query Capabilities with MCP
# MAGIC
# MAGIC ## What You'll Build
# MAGIC Start with your RAG agent from Module 02, discover its limitations with structured data, then progressively add SQL and Genie MCP tools to handle both documents AND data queries.
# MAGIC
# MAGIC ## Learning Flow
# MAGIC 1. **Start**: RAG agent from Module 02 (Vector Search + Memory)
# MAGIC 2. **Problem**: Agent can't answer structured data questions
# MAGIC 3. **Solution 1**: Add SQL MCP tool (requires writing SQL)
# MAGIC 4. **Solution 2**: Add Genie MCP tool (natural language → SQL)
# MAGIC 5. **Result**: Multi-tool agent that routes between documents and data
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed Module 01 (Vector Search) and Module 02 (Memory)
# MAGIC - Employee tables created in Module 00: `employee_data`, `leave_balances`
# MAGIC - Lakebase instance created
# MAGIC
# MAGIC ## Estimated Time
# MAGIC 45 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies

# COMMAND ----------

%pip install -q --upgrade mlflow[databricks]>=3.1.0 databricks-sdk databricks-langchain[memory]>=0.8.0 databricks-vectorsearch>=0.30 langgraph>=0.2.50 langchain-core>=0.3.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Rebuild the RAG Agent from Module 02
# MAGIC
# MAGIC Let's start with the working agent you built earlier: Vector Search for documents + Memory for conversation context.

# COMMAND ----------

import mlflow
from databricks_langchain import ChatDatabricks, CheckpointSaver
from databricks.vector_search.client import VectorSearchClient
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

# Enable MLflow tracing
mlflow.langchain.autolog()
mlflow.tracing.enable()

# Configuration - Use the sample data created in Module 00
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
LLM_ENDPOINT = "databricks-claude-sonnet-4-5"
LAKEBASE_PROJECT = "knowledge-assistant-state"
VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.policy_index"
ENDPOINT_NAME = "agent_bootcamp_endpoint"

print("✅ Configuration loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the Vector Search Tool
# MAGIC
# MAGIC This tool searches policy documents (vacation policy, remote work policy, etc.).

# COMMAND ----------

@tool
def search_policy_documents(query: str) -> str:
    """
    Search company policy documents for information about vacation, leave,
    remote work, professional development, and benefits.

    Use this when the user asks about:
    - Company policies
    - HR procedures
    - Benefits and perks
    - Work arrangements

    Args:
        query: Natural language search query

    Returns:
        Relevant document excerpts
    """
    vsc = VectorSearchClient(disable_notice=True)
    index = vsc.get_index(ENDPOINT_NAME, VECTOR_INDEX)

    results = index.similarity_search(
        query_text=query,
        columns=["source_file", "chunk_text"],
        num_results=3
    )

    formatted = []
    for row in results.get("result", {}).get("data_array", []):
        formatted.append(f"[Source: {row[0]}]\n{row[1]}\n")

    return "\n".join(formatted) if formatted else "No relevant documents found."

print("✅ Vector Search tool created")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build the Agent with Memory
# MAGIC
# MAGIC Same pattern as Module 02: LLM + tools + CheckpointSaver.

# COMMAND ----------

# Initialize LLM
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1)

# Bind tools to LLM
tools = [search_policy_documents]
llm_with_tools = llm.bind_tools(tools)

# Agent node
def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Routing logic
def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# Build graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

# Compile WITHOUT memory for now (we'll add it later for the final agent)
# This keeps the demonstration simple and avoids state management issues
rag_agent = workflow.compile()

print("✅ RAG Agent built successfully (no memory yet - that comes later)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Test the Agent - Show the Problem
# MAGIC
# MAGIC ### ✅ The agent works great for policy questions

# COMMAND ----------

# Test 1: Policy question (should work)
response = rag_agent.invoke(
    {"messages": [HumanMessage(content="What is the vacation policy?")]}
)

print("🔍 Question: What is the vacation policy?")
print("✅ Agent Response:")
print(response["messages"][-1].content)
print("\n" + "="*80 + "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### ❌ But the agent FAILS on structured data questions
# MAGIC
# MAGIC The agent has Vector Search for documents, but no way to query the `employee_data` or `leave_balances` tables.

# COMMAND ----------

# Test 2: Structured data question (will fail)
response = rag_agent.invoke(
    {"messages": [HumanMessage(content="How many employees are in the Engineering department?")]}
)

print("🔍 Question: How many employees are in the Engineering department?")
print("❌ Agent Response (Notice the problem!):")
print(response["messages"][-1].content)
print("\n💡 The agent can't count actual data - it only has access to policy documents!")

# COMMAND ----------

# Test 3: Another data question (will also fail)
response = rag_agent.invoke(
    {"messages": [HumanMessage(content="Who has the most vacation days available?")]}
)

print("🔍 Question: Who has the most vacation days available?")
print("❌ Agent Response:")
print(response["messages"][-1].content)
print("\n💡 Again, the agent can't query the leave_balances table!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Add SQL MCP Tool
# MAGIC
# MAGIC **The Problem**: Agent needs to query structured data tables.
# MAGIC
# MAGIC **Solution 1**: Give the agent a SQL execution tool.
# MAGIC
# MAGIC ### Tool Categories
# MAGIC
# MAGIC | Tool | Purpose | Use Case |
# MAGIC |------|---------|----------|
# MAGIC | **Vector Search** | Find documents/knowledge | "What is the vacation policy?" |
# MAGIC | **SQL Execution** | Query structured data (if you write SQL) | COUNT, JOIN, aggregate data |
# MAGIC | **Genie** | Query structured data (natural language) | Same as SQL but easier! |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create SQL Execution Tool

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# SQL Warehouse configuration
SQL_WAREHOUSE_ID = "your-warehouse-id"  # Update with your warehouse ID

@tool
def execute_sql_query(query: str) -> str:
    """
    Execute a SELECT query on Databricks employee tables.

    Use this when the user asks about:
    - Employee counts, lists, or details
    - Leave balances (vacation, sick days)
    - Department information
    - Organizational structure

    Available tables:
    - agent_bootcamp.knowledge_assistant.employee_data
      Columns: employee_id, name, department, title, manager, hire_date

    - agent_bootcamp.knowledge_assistant.leave_balances
      Columns: employee_id, vacation_available, sick_available, vacation_used_ytd, sick_used_ytd

    Args:
        query: SQL SELECT query to execute

    Returns:
        Query results formatted as a table
    """
    try:
        # Safety check
        if not query.strip().upper().startswith('SELECT'):
            return "Error: Only SELECT queries allowed for safety"

        # Execute via Databricks SDK
        response = w.statement_execution.execute_statement(
            warehouse_id=SQL_WAREHOUSE_ID,
            statement=query,
            wait_timeout="30s"
        )

        # Format results
        if response.result and response.result.data_array:
            columns = [col.name for col in response.manifest.schema.columns]
            rows = response.result.data_array[:10]  # Limit to 10 rows

            result = " | ".join(columns) + "\n"
            result += "-" * 80 + "\n"
            for row in rows:
                result += " | ".join(str(val) if val is not None else "NULL" for val in row) + "\n"

            row_count = len(response.result.data_array)
            if row_count > 10:
                result += f"\n(Showing 10 of {row_count} rows)"

            return result
        else:
            return "Query returned no results"

    except Exception as e:
        return f"Error executing query: {str(e)}"

print("✅ SQL execution tool created")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rebuild Agent with SQL Tool

# COMMAND ----------

# Now with BOTH tools
tools_with_sql = [search_policy_documents, execute_sql_query]
llm_with_sql = llm.bind_tools(tools_with_sql)

# Agent node (updated)
def agent_with_sql(state: MessagesState):
    response = llm_with_sql.invoke(state["messages"])
    return {"messages": [response]}

# Rebuild graph
workflow_sql = StateGraph(MessagesState)
workflow_sql.add_node("agent", agent_with_sql)
workflow_sql.add_node("tools", ToolNode(tools_with_sql))
workflow_sql.set_entry_point("agent")
workflow_sql.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow_sql.add_edge("tools", "agent")

rag_sql_agent = workflow_sql.compile()

print("✅ Agent rebuilt with SQL tool")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Test with SQL Tool - Show It Working
# MAGIC
# MAGIC Now the agent can query structured data! 🎉

# COMMAND ----------

# Test with SQL tool
response = rag_sql_agent.invoke(
    {"messages": [HumanMessage(content="How many employees are in the Engineering department?")]}
)

print("🔍 Question: How many employees are in the Engineering department?")
print("✅ Agent Response (Now with SQL!):")
print(response["messages"][-1].content)
print("\n💡 Check MLflow traces to see the SQL query generated!")

# COMMAND ----------

# Test another data question
response = rag_sql_agent.invoke(
    {"messages": [HumanMessage(content="Who has the most vacation days available?")]}
)

print("🔍 Question: Who has the most vacation days available?")
print("✅ Agent Response:")
print(response["messages"][-1].content)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🎯 Check MLflow Traces
# MAGIC
# MAGIC **Key observations in traces:**
# MAGIC - Agent decides to call `execute_sql_query` tool
# MAGIC - LLM generates the SQL query (e.g., `SELECT name, vacation_available FROM ...`)
# MAGIC - Tool executes and returns results
# MAGIC - Agent formats results into natural language response
# MAGIC
# MAGIC **The limitation**: The agent (LLM) must write correct SQL. Complex queries are error-prone!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Why Genie is Better than SQL
# MAGIC
# MAGIC ### The SQL Problem
# MAGIC
# MAGIC With the SQL tool, the **LLM must generate correct SQL**:
# MAGIC - Table names must be exact
# MAGIC - Column names must be exact
# MAGIC - JOIN syntax must be correct
# MAGIC - SQL dialect differences matter
# MAGIC
# MAGIC **Example failure case:**
# MAGIC ```
# MAGIC User: "Show me employees hired in the last 6 months"
# MAGIC LLM generates: SELECT * FROM employees WHERE hire_date > NOW() - INTERVAL 6 MONTHS
# MAGIC Error: Table 'employees' not found (it's actually 'employee_data')
# MAGIC ```
# MAGIC
# MAGIC ### The Genie Solution
# MAGIC
# MAGIC **Genie** is a specialized AI service that:
# MAGIC 1. Understands your table schemas automatically
# MAGIC 2. Learns from your domain instructions
# MAGIC 3. Generates correct SQL in Databricks SQL dialect
# MAGIC 4. Handles complex queries better than general LLMs
# MAGIC 5. Provides explanations alongside results
# MAGIC
# MAGIC **Same question with Genie:**
# MAGIC ```
# MAGIC User: "Show me employees hired in the last 6 months"
# MAGIC Genie: Automatically generates correct query for your tables ✅
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Create Genie Space
# MAGIC
# MAGIC A **Genie space** is a configured environment that knows about your tables and domain.
# MAGIC
# MAGIC **Note**: The Genie SDK API is evolving. If programmatic creation fails, use the UI method below.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Tables for Genie to analyze (created in 00_foundations/00_setup.py)
TABLES = [
    f"{CATALOG}.{SCHEMA}.employee_data",
    f"{CATALOG}.{SCHEMA}.leave_balances"
]

print("="*80)
print("GENIE SPACE SETUP")
print("="*80)
print("\n📋 Manual Setup Steps:")
print("1. Go to Databricks UI → Genie (in left sidebar)")
print("2. Click 'Create Space'")
print("3. Configure:")
print("   - Name: 'HR Knowledge Assistant'")
print("   - Description: 'Employee and leave data for HR queries'")
print("4. Add tables:")
for table in TABLES:
    print(f"   - {table}")
print("5. Save and copy the space_id from the URL")
print("   (Format: https://.../genie/spaces/01234567-89ab-cdef...)")
print("6. Update GENIE_SPACE_ID variable in next cell")
print("="*80)

# COMMAND ----------

# TODO: After creating Genie space in UI, paste the space_id here
GENIE_SPACE_ID = "your-genie-space-id-here"  # Replace with actual space_id

if GENIE_SPACE_ID == "your-genie-space-id-here":
    print("⚠️  WARNING: Please update GENIE_SPACE_ID with your actual space_id!")
    print("💡 Steps 8-10 will be skipped until you do.")
else:
    print(f"✅ Using Genie space: {GENIE_SPACE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Create Genie Tool

# COMMAND ----------

@tool
def query_employee_data(question: str) -> str:
    """
    Query employee and leave data using natural language.

    Use this when the user asks about:
    - Employee counts, lists, or details
    - Leave balances (vacation, sick days)
    - Department information
    - Organizational structure
    - Any data analysis or aggregations

    This is better than execute_sql_query because you don't need to write SQL!

    Args:
        question: Natural language question about employee data

    Returns:
        Analysis results with explanation
    """
    try:
        # Send message to Genie
        response = w.genie.create_message(
            space_id=GENIE_SPACE_ID,
            content=question
        )

        # Get the result
        message = w.genie.get_message(
            space_id=GENIE_SPACE_ID,
            conversation_id=response.conversation_id,
            message_id=response.id
        )

        result = f"🔍 Analysis for: {question}\n\n"

        # Include generated SQL for transparency
        if message.attachments and message.attachments[0].query:
            sql_query = message.attachments[0].query.query
            result += f"📝 SQL Query:\n{sql_query}\n\n"

        # Include results
        if message.attachments and message.attachments[0].text:
            result += f"📊 Results:\n{message.attachments[0].text.content}\n\n"

        # Include explanation
        result += f"💡 Explanation:\n{message.content}"

        return result

    except Exception as e:
        return f"Error querying Genie: {str(e)}"

print("✅ Genie tool created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Build Final Multi-Tool Agent
# MAGIC
# MAGIC Now with THREE tools:
# MAGIC 1. **Vector Search** → Policy documents
# MAGIC 2. **SQL** → Direct SQL queries (if needed)
# MAGIC 3. **Genie** → Natural language data queries (preferred for data!)

# COMMAND ----------

# Final agent with all tools
all_tools = [search_policy_documents, execute_sql_query, query_employee_data]
llm_final = llm.bind_tools(all_tools)

# Agent node (final)
def final_agent_node(state: MessagesState):
    response = llm_final.invoke(state["messages"])
    return {"messages": [response]}

# Rebuild graph
workflow_final = StateGraph(MessagesState)
workflow_final.add_node("agent", final_agent_node)
workflow_final.add_node("tools", ToolNode(all_tools))
workflow_final.set_entry_point("agent")
workflow_final.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow_final.add_edge("tools", "agent")

final_agent = workflow_final.compile()

print("✅ Final multi-tool agent built!")
print("\nAgent capabilities:")
print("  📄 Vector Search → Policy documents")
print("  💾 SQL → Direct queries (when needed)")
print("  🧞 Genie → Natural language data analysis (preferred)")
print("\n💡 Note: Add CheckpointSaver from Module 02 if you need conversation memory!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Test the Complete Agent
# MAGIC
# MAGIC Watch the agent intelligently route between tools based on the question type!

# COMMAND ----------

# Test 1: Policy question → Should use Vector Search
print("="*80)
response = final_agent.invoke(
    {"messages": [HumanMessage(content="What is the vacation accrual policy?")]}
)
print("🔍 Question: What is the vacation accrual policy?")
print("🎯 Expected tool: search_policy_documents")
print("\n✅ Response:")
print(response["messages"][-1].content)
print("\n")

# COMMAND ----------

# Test 2: Data question → Should use Genie
print("="*80)
response = final_agent.invoke(
    {"messages": [HumanMessage(content="Who are the top 3 employees with the most vacation days available?")]}
)
print("🔍 Question: Who are the top 3 employees with the most vacation days available?")
print("🎯 Expected tool: query_employee_data (Genie)")
print("\n✅ Response:")
print(response["messages"][-1].content)
print("\n")

# COMMAND ----------

# Test 3: Complex question requiring both tools
print("="*80)
response = final_agent.invoke(
    {"messages": [HumanMessage(content="How many vacation days do Engineering employees have on average, and what does the policy say about maximum carryover?")]}
)
print("🔍 Question: How many vacation days do Engineering employees have on average, and what does the policy say about maximum carryover?")
print("🎯 Expected: Multiple tool calls (Genie + Vector Search)")
print("\n✅ Response:")
print(response["messages"][-1].content)
print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Inspect MLflow Traces
# MAGIC
# MAGIC **Go to MLflow UI and examine the traces to see:**
# MAGIC
# MAGIC 1. **Tool routing logic**: How does the agent decide which tool to use?
# MAGIC 2. **Genie behavior**:
# MAGIC    - Natural language question goes in
# MAGIC    - SQL query is generated
# MAGIC    - Results come back
# MAGIC    - Agent formats into natural language
# MAGIC 3. **Multi-tool coordination**: When does the agent call multiple tools?
# MAGIC 4. **Error handling**: What happens when a tool fails?
# MAGIC
# MAGIC ### Key Trace Elements to Look For:
# MAGIC - `agent` span: LLM deciding which tool to use
# MAGIC - `tools` span: Tool execution
# MAGIC - Tool inputs/outputs
# MAGIC - Latency breakdown
# MAGIC - Error messages (if any)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What You Built
# MAGIC
# MAGIC You progressively enhanced your agent:
# MAGIC
# MAGIC ```
# MAGIC Module 01: Vector Search only
# MAGIC     ↓
# MAGIC Module 02: + Memory (CheckpointSaver)
# MAGIC     ↓
# MAGIC Module 04: + SQL tool → Can query data (but LLM must write SQL)
# MAGIC     ↓
# MAGIC Module 04: + Genie tool → Can query data with natural language! ⭐
# MAGIC ```
# MAGIC
# MAGIC ### Final Agent Architecture
# MAGIC
# MAGIC ```
# MAGIC                    ┌─────────────┐
# MAGIC                    │     LLM     │
# MAGIC                    └──────┬──────┘
# MAGIC                           │
# MAGIC              ┌────────────┼────────────┐
# MAGIC              │            │            │
# MAGIC         ┌────▼───┐   ┌───▼────┐  ┌───▼─────┐
# MAGIC         │Vector  │   │  SQL   │  │  Genie  │
# MAGIC         │Search  │   │  MCP   │  │  MCP    │
# MAGIC         └────┬───┘   └───┬────┘  └───┬─────┘
# MAGIC              │           │           │
# MAGIC         Documents    Tables      Tables
# MAGIC       (unstructured) (SQL)    (Natural Lang)
# MAGIC ```
# MAGIC
# MAGIC ### Key Learnings
# MAGIC
# MAGIC 1. **Tool Selection**: Agents can intelligently route between multiple tools
# MAGIC 2. **MCP Pattern**: Wrap external services as `@tool` decorated functions
# MAGIC 3. **SQL vs Genie**: Genie is better for data queries (no SQL writing needed)
# MAGIC 4. **Tracing**: MLflow traces show tool usage, latency, errors
# MAGIC 5. **Multi-modal**: Combine unstructured (docs) + structured (data) capabilities
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC - **Module 05**: Deploy to production with Model Serving
# MAGIC - Experiment with other MCP tools (UC Functions, Web Search)
# MAGIC - Build domain-specific tools for your use case

# COMMAND ----------
