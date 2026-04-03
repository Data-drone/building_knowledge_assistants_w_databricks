# Databricks notebook source
# MAGIC %md
# MAGIC # Module 04: Extending Your Agent with Data Tools
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
# MAGIC
# MAGIC Same stack as prior modules. We add `langchain-core` explicitly since the
# MAGIC tool decorator and message types come from there.

# COMMAND ----------

%pip install -q --upgrade \
  "databricks-sdk>=0.101" \
  "mlflow[databricks]>=3.10" \
  "databricks-langchain[memory]>=0.17" \
  "databricks-vectorsearch>=0.66" \
  "langgraph>=1.1" \
  "langchain-core>=1.2"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Rebuild the RAG Agent from Module 02
# MAGIC
# MAGIC Let's start with the working agent you built earlier: Vector Search for documents + Memory for conversation context.

# COMMAND ----------

import mlflow
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

# ── Configuration ── edit these to match your workspace ──
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"

VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.policy_index"

SQL_WAREHOUSE_ID = ""   # Your SQL warehouse ID (find in SQL Warehouses UI)
GENIE_SPACE_ID = ""     # Paste from 00_setup.py output

# Enable MLflow tracing
mlflow.langchain.autolog()
mlflow.tracing.enable()

print("✅ Configuration loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the Vector Search Tool
# MAGIC
# MAGIC `VectorSearchRetrieverTool` from `databricks-langchain` is the recommended
# MAGIC way to plug a Vector Search index into a LangChain/LangGraph agent. It
# MAGIC handles querying, result formatting, and MLflow retriever tracing for you.

# COMMAND ----------

search_policy_documents = VectorSearchRetrieverTool(
    index_name=VECTOR_INDEX,
    tool_name="search_policy_documents",
    tool_description=(
        "Search company policy documents for information about vacation and leave "
        "policies, remote work, professional development, benefits, and equipment."
    ),
    columns=["source_file", "chunk_text"],
    num_results=3,
)

print("✅ Vector Search tool created")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build a Minimal Agent Graph
# MAGIC
# MAGIC Same LangGraph pattern as Module 02 but without memory for now. We keep
# MAGIC the graph simple so the focus stays on adding new tools, not state management.

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
# MAGIC ## Step 3: Test the Agent — Show the Problem
# MAGIC
# MAGIC ### The agent handles policy questions well

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
# MAGIC ### But it fails on structured data questions
# MAGIC
# MAGIC The agent has Vector Search for documents but no way to query the
# MAGIC `employee_data` or `leave_balances` tables. Watch it try to answer
# MAGIC from policy text instead of actual data.

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
# MAGIC
# MAGIC This wraps `statement_execution` from the Databricks SDK so the agent
# MAGIC can run SELECT queries against employee tables. Only SELECTs are allowed.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

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
      Columns: employee_id, name, department, role, hire_date

    - agent_bootcamp.knowledge_assistant.leave_balances
      Columns: employee_id, vacation_days_total, vacation_days_used, sick_days_total, sick_days_used

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
# MAGIC
# MAGIC Same graph shape, just a wider tool list. The LLM decides whether a
# MAGIC question needs Vector Search or SQL based on the tool descriptions.

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
# MAGIC ## Step 5: Test with SQL Tool
# MAGIC
# MAGIC The same questions that failed before should now return real data.

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
# MAGIC ### Check MLflow Traces
# MAGIC
# MAGIC **Key observations in traces:**
# MAGIC - Agent decides to call `execute_sql_query` tool
# MAGIC - LLM generates the SQL query (e.g., `SELECT name, vacation_days_total - vacation_days_used AS vacation_days_left FROM ...`)
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
# MAGIC ## Step 7: Verify Genie Space
# MAGIC
# MAGIC You created the Genie space in `00_setup.py`. This step confirms the
# MAGIC space ID is set so the Genie tool can use it.

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
print("\nUsing GENIE_SPACE_ID set at the top of this notebook.")
print("If it is empty, run 04_mcp_tool_integration/00_setup.py and paste the generated space ID above.")
print("The Genie space should include these tables:")
for table in TABLES:
    print(f"   - {table}")
print("="*80)

# COMMAND ----------

if not GENIE_SPACE_ID:
    raise ValueError(
        "GENIE_SPACE_ID is empty. Run 04_mcp_tool_integration/00_setup.py, "
        "copy the generated space ID into the GENIE_SPACE_ID variable at the top of this notebook, "
        "and rerun."
    )

print(f"✅ Using Genie space: {GENIE_SPACE_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Create Genie Tool
# MAGIC
# MAGIC Unlike the SQL tool where the LLM writes raw SQL, this tool sends a
# MAGIC natural-language question to Genie and gets back results. Genie handles
# MAGIC schema resolution, SQL generation, and dialect differences internally.

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
        # Start a fresh Genie conversation for each question using the current SDK API.
        message = w.genie.start_conversation_and_wait(
            space_id=GENIE_SPACE_ID,
            content=question
        )

        result = f"🔍 Analysis for: {question}\n\n"
        sql_query = None
        answer_text = None
        follow_up_questions = []

        for attachment in message.attachments or []:
            if getattr(attachment, "query", None) and getattr(attachment.query, "query", None):
                sql_query = attachment.query.query
            if getattr(attachment, "text", None) and getattr(attachment.text, "content", None):
                answer_text = attachment.text.content
            if (
                getattr(attachment, "suggested_questions", None)
                and getattr(attachment.suggested_questions, "questions", None)
            ):
                follow_up_questions = attachment.suggested_questions.questions

        # Include generated SQL for transparency
        if sql_query:
            result += f"📝 SQL Query:\n{sql_query}\n\n"

        # Include results
        if answer_text:
            result += f"📊 Results:\n{answer_text}\n\n"

        if follow_up_questions:
            result += "💡 Suggested follow-up questions:\n"
            result += "\n".join(f"- {question}" for question in follow_up_questions)
        elif not answer_text:
            result += "💡 Genie completed the query, but did not return a text summary."

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
# MAGIC - **Module 05**: Deploy to production with Databricks Apps
# MAGIC - Experiment with other MCP tools (UC Functions, Web Search)
# MAGIC - Build domain-specific tools for your use case

# COMMAND ----------
