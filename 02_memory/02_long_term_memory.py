# Databricks notebook source
# MAGIC %md
# MAGIC # Long-Term Memory: Cross-Session Personalization with DatabricksStore
# MAGIC
# MAGIC ## What You'll Build
# MAGIC Add persistent memory to a RAG agent, enabling personalization across sessions and threads.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Store user facts and preferences with DatabricksStore
# MAGIC - Compare short-term vs long-term memory
# MAGIC - Organize memory with namespace patterns
# MAGIC - Combine CheckpointSaver and DatabricksStore
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed Notebook 01 (short-term memory) OR understand checkpointed RAG agents
# MAGIC - Vector Search index available
# MAGIC - Lakebase instance created
# MAGIC
# MAGIC ## Estimated Time
# MAGIC 20 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies

# COMMAND ----------

%pip install -q --upgrade mlflow[databricks]>=3.1.0 databricks-langchain[memory]>=0.8.0 databricks-vectorsearch>=0.30 langgraph>=0.2.50 langchain-core>=0.3.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configuration and Imports

# COMMAND ----------

import mlflow
from databricks_langchain import ChatDatabricks, CheckpointSaver, DatabricksStore
from databricks.vector_search.client import VectorSearchClient
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from datetime import datetime

# COMMAND ----------

# Configuration - Update these for your environment
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
LLM_ENDPOINT = "databricks-claude-sonnet-4-5"
LAKEBASE_PROJECT = "knowledge-assistant-state"

# Derived values
VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.policy_index"
ENDPOINT_NAME = "agent_bootcamp_endpoint"

# Utility function for namespace-safe user IDs
def sanitize_namespace_id(user_id: str) -> str:
    """DatabricksStore doesn't allow periods in namespace labels."""
    return user_id.replace(".", "_").replace("@", "_at_")

print(f"✓ Configuration loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Understanding Memory Types
# MAGIC
# MAGIC | Aspect | Short-Term (CheckpointSaver) | Long-Term (DatabricksStore) |
# MAGIC |--------|------------------------------|------------------------------|
# MAGIC | **Scope** | Single thread/conversation | Across all threads/sessions |
# MAGIC | **Content** | Full message history | Facts, preferences |
# MAGIC | **Lifetime** | Duration of conversation | Indefinite (days to years) |
# MAGIC | **Use case** | Context within conversation | Personalization across sessions |
# MAGIC
# MAGIC **Example:** User mentions "I work in Engineering" on Monday. On Tuesday (new session/thread):
# MAGIC - CheckpointSaver: Doesn't remember (different thread)
# MAGIC - DatabricksStore: Remembers (stored across all threads)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Build Checkpointed RAG Agent

# COMMAND ----------

# Create Vector Search tool
@tool
def search_policy_documents(query: str) -> str:
    """
    Search company policy documents for information about vacation, leave,
    remote work, professional development, and benefits.
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

# Initialize LLM and tools
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1)
tools = [search_policy_documents]
llm_with_tools = llm.bind_tools(tools)

# Agent logic
def call_agent(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: MessagesState):
    last_msg = state["messages"][-1]
    return "tools" if hasattr(last_msg, "tool_calls") and last_msg.tool_calls else END

# Build graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_agent)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

# Add CheckpointSaver
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
instances = list(w.database.list_database_instances())
target_instance = next((inst for inst in instances if inst.name == LAKEBASE_PROJECT), None)
if not target_instance:
    raise Exception(f"Lakebase instance '{LAKEBASE_PROJECT}' not found.")

checkpointer = CheckpointSaver(instance_name=LAKEBASE_PROJECT)
try:
    checkpointer.setup()
except Exception as e:
    if "already exists" not in str(e).lower():
        raise

agent_with_checkpointer = workflow.compile(checkpointer=checkpointer)

print("✓ Checkpointed RAG agent built (short-term memory only)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Test - Facts Don't Persist Across Sessions

# COMMAND ----------

# Monday - User mentions department (Thread 1)
print("MONDAY SESSION (thread: session-monday):")
print("=" * 80)
config_monday = {"configurable": {"thread_id": "session-monday"}}
result_monday = agent_with_checkpointer.invoke(
    {"messages": [HumanMessage(content="I work in the Engineering department. What professional development opportunities are available for me?")]},
    config_monday
)
print(f"User: I work in the Engineering department. What professional development opportunities are available for me?")
print(f"Agent: {result_monday['messages'][-1].content[:200]}...")

# Tuesday - NEW session, ask for recommendations (Thread 2)
print("\n\nTUESDAY SESSION (thread: session-tuesday):")
print("=" * 80)
config_tuesday = {"configurable": {"thread_id": "session-tuesday"}}
result_tuesday = agent_with_checkpointer.invoke(
    {"messages": [HumanMessage(content="What professional development should I focus on?")]},
    config_tuesday
)
print(f"User: What professional development should I focus on?")
print(f"Agent: {result_tuesday['messages'][-1].content[:200]}...")
print("\n→ Agent doesn't remember Engineering department (different thread)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Initialize DatabricksStore

# COMMAND ----------

# Initialize DatabricksStore for long-term memory
store = DatabricksStore(instance_name=LAKEBASE_PROJECT)
try:
    store.setup()
    print("✓ DatabricksStore tables initialized")
except Exception as e:
    if "already exists" in str(e).lower():
        print("✓ DatabricksStore tables already exist (reusing)")
    else:
        raise

print("✓ DatabricksStore ready")
print("\nMemory Architecture:")
print("  CheckpointSaver → Short-term: conversation history (per thread)")
print("  DatabricksStore → Long-term: user facts/preferences (across all threads)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Store User Facts and Preferences
# MAGIC
# MAGIC Organize memory with **namespaces**: `(category, identifier)`
# MAGIC - Example: `("user_facts", "alice_at_company_com")`
# MAGIC - Note: Sanitize user IDs (no periods allowed in namespace labels)

# COMMAND ----------

# Example user
user_id = "alice@company.com"
sanitized_user_id = sanitize_namespace_id(user_id)

# Create namespaces
facts_namespace = ("user_facts", sanitized_user_id)
prefs_namespace = ("user_preferences", sanitized_user_id)

print(f"Storing data for user: {user_id}")
print(f"Namespaces: {facts_namespace}, {prefs_namespace}")
print("=" * 80)

# Store user facts
facts = {
    "department": "Engineering",
    "role": "Senior Engineer",
    "skills": ["Python", "SQL", "Machine Learning"]
}

for key, value in facts.items():
    store.put(
        facts_namespace,
        key,
        {"value": value, "updated_at": datetime.now().isoformat()}
    )
    print(f"✓ Stored fact: {key} = {value}")

# Store user preferences
preferences = {
    "meeting_time": "morning",
    "notification_method": "email"
}

for key, value in preferences.items():
    store.put(
        prefs_namespace,
        key,
        {"value": value, "updated_at": datetime.now().isoformat()}
    )
    print(f"✓ Stored preference: {key} = {value}")

print(f"\n✓ Stored {len(facts)} facts and {len(preferences)} preferences")

# Verify retrieval
dept = store.get(facts_namespace, "department")
print(f"\nTest retrieval: department = {dept.value['value'] if dept else 'Not found'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Modify Agent to Use Long-Term Memory

# COMMAND ----------

# Create agent that reads from DatabricksStore
def call_agent_with_store(state: MessagesState):
    """Agent logic that reads from Store for personalization."""
    messages = state["messages"]

    # Extract user_id (in production, from authentication)
    user_id = "alice@company.com"  # Hardcoded for demo
    sanitized_user_id = sanitize_namespace_id(user_id)

    # Look up user facts and preferences
    facts_namespace = ("user_facts", sanitized_user_id)
    prefs_namespace = ("user_preferences", sanitized_user_id)

    dept = store.get(facts_namespace, "department")
    role = store.get(facts_namespace, "role")
    meeting_pref = store.get(prefs_namespace, "meeting_time")

    # Build context from long-term memory
    context_parts = []
    if dept:
        context_parts.append(f"User works in {dept.value['value']} department")
    if role:
        context_parts.append(f"User is a {role.value['value']}")
    if meeting_pref:
        context_parts.append(f"User prefers {meeting_pref.value['value']} meetings")

    # Inject context as system message
    if context_parts:
        context_message = SystemMessage(
            content=f"User context: {'; '.join(context_parts)}"
        )
        messages = [context_message] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Build new workflow with Store-enabled agent
workflow_with_store = StateGraph(MessagesState)
workflow_with_store.add_node("agent", call_agent_with_store)
workflow_with_store.add_node("tools", ToolNode(tools))
workflow_with_store.set_entry_point("agent")
workflow_with_store.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow_with_store.add_edge("tools", "agent")

# Compile with BOTH checkpointer AND store access
agent_with_both = workflow_with_store.compile(checkpointer=checkpointer)

print("✓ Agent with BOTH memory types built")
print("  Short-term: CheckpointSaver (conversation context)")
print("  Long-term: DatabricksStore (user facts/preferences)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Test - Personalization Across Sessions Works

# COMMAND ----------

# Wednesday - NEW session, ask for recommendations (Thread 3)
print("WEDNESDAY SESSION (thread: session-wednesday):")
print("=" * 80)
config_wednesday = {"configurable": {"thread_id": "session-wednesday"}}
result_wednesday = agent_with_both.invoke(
    {"messages": [HumanMessage(content="What professional development should I focus on?")]},
    config_wednesday
)
print(f"User: What professional development should I focus on?")
print(f"Agent: {result_wednesday['messages'][-1].content[:250]}...")
print("\n→ Agent personalizes response using Engineering department from DatabricksStore")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Combined Memory - CheckpointSaver + DatabricksStore

# COMMAND ----------

# Start a conversation
print("COMBINED MEMORY DEMO:")
print("=" * 80)
thread_combo = "combo-demo"
config_combo = {"configurable": {"thread_id": thread_combo}}

# Turn 1: Ask about time off
print("\nTurn 1:")
result1 = agent_with_both.invoke(
    {"messages": [HumanMessage(content="I need to take some time off next month.")]},
    config_combo
)
print(f"User: I need to take some time off next month.")
print(f"Agent: {result1['messages'][-1].content[:200]}...")

# Turn 2: Follow-up question
print("\nTurn 2:")
result2 = agent_with_both.invoke(
    {"messages": [HumanMessage(content="When would be a good time to discuss this?")]},
    config_combo
)
print(f"User: When would be a good time to discuss this?")
print(f"Agent: {result2['messages'][-1].content[:200]}...")
print("\n→ Agent uses:")
print("  • CheckpointSaver: Knows 'this' refers to time off")
print("  • DatabricksStore: Suggests morning (user preference)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Update Memory Based on Conversations

# COMMAND ----------

# Learn new fact from conversation
new_interest = "cloud architecture"

# Get existing interests
interests_item = store.get(facts_namespace, "interests")
interests = interests_item.value.get("value", []) if interests_item else []

# Add new interest
if new_interest not in interests:
    interests.append(new_interest)
    store.put(
        facts_namespace,
        "interests",
        {"value": interests, "updated_at": datetime.now().isoformat()}
    )
    print(f"✓ Learned new interest: {new_interest}")
    print(f"  All interests: {interests}")
    print("\n→ This fact persists across all future conversations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What We Built
# MAGIC - ✅ Checkpointed RAG agent (short-term memory)
# MAGIC - ✅ Demonstrated cross-session memory limitation
# MAGIC - ✅ Added DatabricksStore for persistent memory
# MAGIC - ✅ Organized memory with namespaces
# MAGIC - ✅ Modified agent to inject user context
# MAGIC - ✅ Combined both memory types
# MAGIC
# MAGIC ### Key Concepts
# MAGIC
# MAGIC **Memory Types:**
# MAGIC - `CheckpointSaver` → Short-term (conversation context within one thread)
# MAGIC - `DatabricksStore` → Long-term (user facts across all sessions)
# MAGIC
# MAGIC **Namespace Patterns:**
# MAGIC ```python
# MAGIC # User-specific memory
# MAGIC ("user_facts", sanitized_user_id)
# MAGIC ("user_preferences", sanitized_user_id)
# MAGIC
# MAGIC # Team/shared memory
# MAGIC ("team_knowledge", "engineering")
# MAGIC
# MAGIC # Time-based archiving
# MAGIC ("user_facts", sanitized_user_id, "2024-Q1")
# MAGIC ```
# MAGIC
# MAGIC **Production Considerations:**
# MAGIC - Implement data retention policies (GDPR compliance)
# MAGIC - Allow users to view/delete their data
# MAGIC - Add confidence scores to stored facts
# MAGIC - Implement access controls and audit logging
# MAGIC - Consider temporal awareness (facts change over time)
# MAGIC
# MAGIC ### Complete Memory Architecture
# MAGIC
# MAGIC Your agent now has:
# MAGIC - ✅ Document search (Vector Search)
# MAGIC - ✅ Conversation memory (CheckpointSaver)
# MAGIC - ✅ User personalization (DatabricksStore)
# MAGIC - ✅ Thread isolation (separate conversations)
# MAGIC - ✅ Cross-session memory (facts persist forever)
# MAGIC
# MAGIC ### Memory Comparison
# MAGIC
# MAGIC | Type | Example |
# MAGIC |------|---------|
# MAGIC | **Short-term** | "You mentioned 10 days earlier in this conversation" |
# MAGIC | **Long-term** | "You work in Engineering department" (remembered forever) |
# MAGIC | **Vector Search** | "Company policy allows 15 days vacation per year" |
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC Continue to **Module 03: Evaluation** to add observability, tracing, and systematic testing.

# COMMAND ----------

print("✓ Long-term memory tutorial complete!")
print("\nYour agent now has:")
print("  ✓ Short-term: Conversation context (CheckpointSaver)")
print("  ✓ Long-term: User personalization (DatabricksStore)")
print("  ✓ Knowledge: Document search (Vector Search)")
