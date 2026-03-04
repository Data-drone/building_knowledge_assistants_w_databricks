# Databricks notebook source
# MAGIC %md
# MAGIC # Short-Term Memory: Multi-Turn Conversations with CheckpointSaver
# MAGIC
# MAGIC ## What You'll Build
# MAGIC A RAG agent with conversation memory that maintains context across multiple turns.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Build RAG agents with Vector Search tools
# MAGIC - Compare agent behavior with and without memory
# MAGIC - Add CheckpointSaver for conversation continuity
# MAGIC - Manage conversations with thread IDs
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Vector Search index: `{CATALOG}.{SCHEMA}.policy_index`
# MAGIC - Lakebase instance created
# MAGIC - LLM endpoint available
# MAGIC
# MAGIC ## Estimated Time
# MAGIC 15 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies

# COMMAND ----------

%pip install -q --upgrade mlflow[databricks]>=3.1.0 databricks-langchain[memory]>=0.8.0 databricks-vectorsearch>=0.30 langgraph>=0.2.50 langchain-core>=0.3.0
%restart_python
# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configuration and Imports

# COMMAND ----------

import mlflow
from databricks_langchain import ChatDatabricks, CheckpointSaver
from databricks.vector_search.client import VectorSearchClient
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
import uuid_utils

# COMMAND ----------

# Configuration - Update these for your environment
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
LLM_ENDPOINT = "databricks-claude-sonnet-4-5"
LAKEBASE_INSTANCE_NAME = "knowledge-assistant-state"

# Derived values
VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.policy_index"
#ENDPOINT_NAME = "agent_bootcamp_endpoint"
ENDPOINT_NAME = "one-env-shared-endpoint-15"

print(f"✓ Configuration loaded")
print(f"  Vector Index: {VECTOR_INDEX}")
print(f"  Lakebase Instance: {LAKEBASE_INSTANCE_NAME}")


def resolve_thread_id(custom_thread_id: str | None = None, conversation_id: str | None = None) -> str:
    """Canonical thread-id priority: custom input -> conversation id -> generated UUID."""
    if custom_thread_id:
        return custom_thread_id
    if conversation_id:
        return conversation_id
    return str(uuid_utils.uuid7())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Build RAG Agent with Vector Search Tool

# COMMAND ----------

# Create Vector Search tool
@tool
def search_policy_documents(query: str) -> str:
    """
    Search company policy documents for information about vacation, leave,
    remote work, professional development, and benefits.

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

print("✓ RAG agent built with Vector Search tool")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Test Without Memory - Multi-Turn Conversations Fail
# MAGIC
# MAGIC First, compile the agent **without** CheckpointSaver and observe the problem.

# COMMAND ----------

agent_without_memory = workflow.compile()

# Turn 1 - Works fine
print("=" * 80)
print("WITHOUT MEMORY - Turn 1:")
result1 = agent_without_memory.invoke({
    "messages": [HumanMessage(content="How much vacation time do employees get?")]
})
print(f"User: How much vacation time do employees get?")
print(f"Agent: {result1['messages'][-1].content[:200]}...")

# Turn 2 - Fails - no context
print("\n" + "=" * 80)
print("WITHOUT MEMORY - Turn 2:")
result2 = agent_without_memory.invoke({
    "messages": [HumanMessage(content="What about sick leave?")]
})
print(f"User: What about sick leave?")
print(f"Agent: {result2['messages'][-1].content[:200]}...")
print("\n→ Agent has no context from Turn 1. Each invoke() is independent.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Add CheckpointSaver for Memory

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# Verify Lakebase instance exists
w = WorkspaceClient()
instances = list(w.database.list_database_instances())

target_instance = next((inst for inst in instances if inst.name == LAKEBASE_INSTANCE_NAME), None)
if not target_instance:
    raise Exception(f"Lakebase instance '{LAKEBASE_INSTANCE_NAME}' not found. Please create it first.")

print(f"✓ Lakebase instance '{LAKEBASE_INSTANCE_NAME}' found (State: {target_instance.state})")

# Initialize CheckpointSaver
checkpointer = CheckpointSaver(instance_name=LAKEBASE_INSTANCE_NAME)

try:
    checkpointer.setup()
    print("✓ CheckpointSaver tables initialized")
except Exception as e:
    if "already exists" in str(e).lower():
        print("✓ CheckpointSaver tables already exist (reusing)")
    else:
        raise

# Compile with memory
agent_with_memory = workflow.compile(checkpointer=checkpointer)

print("✓ Agent compiled with CheckpointSaver")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Test With Memory - Multi-Turn Conversations Work
# MAGIC
# MAGIC Now the agent remembers context within a thread.

# COMMAND ----------

# Start a conversation with thread_id
thread_id = resolve_thread_id(custom_thread_id="demo-001")
config = {"configurable": {"thread_id": thread_id}}

# Turn 1
print("=" * 80)
print("WITH MEMORY - Turn 1:")
result1 = agent_with_memory.invoke(
    {"messages": [HumanMessage(content="How much vacation time do employees get?")]},
    config=config
)
print(f"User: How much vacation time do employees get?")
print(f"Agent: {result1['messages'][-1].content[:200]}...")

# Turn 2 - Now has context
print("\n" + "=" * 80)
print("WITH MEMORY - Turn 2:")
result2 = agent_with_memory.invoke(
    {"messages": [HumanMessage(content="What about sick leave?")]},
    config=config  # Same thread_id
)
print(f"User: What about sick leave?")
print(f"Agent: {result2['messages'][-1].content[:200]}...")

# Turn 3 - Full conversation context
print("\n" + "=" * 80)
print("WITH MEMORY - Turn 3:")
result3 = agent_with_memory.invoke(
    {"messages": [HumanMessage(content="Do they carry over to next year?")]},
    config=config  # Same thread_id
)
print(f"User: Do they carry over to next year?")
print(f"Agent: {result3['messages'][-1].content[:200]}...")
print("\n→ Agent maintains full conversation context via CheckpointSaver")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Thread Isolation - Each thread_id Is a Separate Conversation

# COMMAND ----------

# Different thread = no context
print("=" * 80)
print("NEW THREAD (demo-002):")
new_config = {"configurable": {"thread_id": "demo-002"}}
result_new = agent_with_memory.invoke(
    {"messages": [HumanMessage(content="Do they roll over?")]},
    new_config  # Different thread_id
)
print(f"User: Do they roll over?")
print(f"Agent: {result_new['messages'][-1].content[:200]}...")
print("\n→ New thread has no context. Agent doesn't know what 'they' refers to.")

# Resume original thread
print("\n" + "=" * 80)
print("RESUME ORIGINAL THREAD (demo-001):")
result_resume = agent_with_memory.invoke(
    {"messages": [HumanMessage(content="What's the maximum carryover?")]},
    config  # Back to original thread_id
)
print(f"User: What's the maximum carryover?")
print(f"Agent: {result_resume['messages'][-1].content[:200]}...")
print("\n→ Original thread still has full conversation history.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What We Built
# MAGIC - ✅ RAG agent with Vector Search tool
# MAGIC - ✅ Demonstrated multi-turn conversation failure without memory
# MAGIC - ✅ Added CheckpointSaver for conversation continuity
# MAGIC - ✅ Showed thread-based conversation management
# MAGIC
# MAGIC ### Key Concepts
# MAGIC
# MAGIC **CheckpointSaver** stores conversation state in Lakebase PostgreSQL:
# MAGIC - Each `thread_id` = separate conversation
# MAGIC - State saved automatically after each turn
# MAGIC - Same `thread_id` = agent remembers everything
# MAGIC - Different `thread_id` = agent has no context
# MAGIC
# MAGIC **Production Thread ID Pattern:**
# MAGIC ```python
# MAGIC thread_id = f"user:{user_id}:session:{session_uuid}"
# MAGIC # Example: "user:alice@company.com:session:abc-123"
# MAGIC ```
# MAGIC
# MAGIC ### Current Capabilities
# MAGIC - ✅ Document search (Vector Search)
# MAGIC - ✅ Multi-turn conversations (CheckpointSaver)
# MAGIC - ✅ Thread-based session management
# MAGIC
# MAGIC ### What's Missing
# MAGIC - ❌ Cross-session memory (new threads forget user facts)
# MAGIC - ❌ Personalization (agent doesn't remember preferences)
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC Continue to [02_long_term_memory.py](02_long_term_memory.py) to add **DatabricksStore** for persistent personalization across sessions.

# COMMAND ----------

print("✓ Short-term memory tutorial complete!")
print(f"\nNext: Open 02_long_term_memory.py to add cross-session memory")
