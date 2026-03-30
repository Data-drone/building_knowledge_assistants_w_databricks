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
# MAGIC - Build LLM-managed memory tools (get, save, delete)
# MAGIC - Use semantic search over memories instead of exact key lookups
# MAGIC - Combine CheckpointSaver and DatabricksStore in a production-ready pattern
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed Notebook 01 (short-term memory) OR understand checkpointed RAG agents
# MAGIC - Vector Search index available
# MAGIC - Lakebase instance created
# MAGIC
# MAGIC ## Estimated Time
# MAGIC 30 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies

# COMMAND ----------

%pip install -q --upgrade \
  "databricks-sdk>=0.101,<0.103" \
  "mlflow[databricks]>=3.10,<3.11" \
  "databricks-langchain[memory]>=0.17,<0.18" \
  "databricks-vectorsearch>=0.66,<0.67" \
  "langgraph>=1.1,<1.2" \
  "langchain-core>=1.2,<2"
%restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configuration and Imports

# COMMAND ----------

import json
from databricks_langchain import ChatDatabricks, CheckpointSaver, DatabricksStore, VectorSearchRetrieverTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from datetime import datetime

# COMMAND ----------

# Configuration - Update these for your environment
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"
LAKEBASE_INSTANCE_NAME = "knowledge-assistant-state"

# Derived values
VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.policy_index"
#ENDPOINT_NAME = "agent_bootcamp_endpoint"
ENDPOINT_NAME = "one-env-shared-endpoint-15"

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

# Initialize LLM and tools
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1)
checkpointed_rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a beginner-friendly Databricks coach helping users answer questions "
        "about company policy documents. Use retrieved context when it is available, "
        "use the conversation history to understand follow-up questions, cite source "
        "file names when the tool provides them, and do not make up policy details "
        "that are not in the documents. Answer in 2 short bullet points and end with "
        "one practical next step."
    ),
    MessagesPlaceholder("messages"),
])
tools = [search_policy_documents]
llm_with_tools = llm.bind_tools(tools)

# Agent logic
def call_agent(state: MessagesState):
    prompt_messages = checkpointed_rag_prompt.invoke({"messages": state["messages"]}).messages
    response = llm_with_tools.invoke(prompt_messages)
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
target_instance = next((inst for inst in instances if inst.name == LAKEBASE_INSTANCE_NAME), None)
if not target_instance:
    raise Exception(f"Lakebase instance '{LAKEBASE_INSTANCE_NAME}' not found.")

checkpointer = CheckpointSaver(instance_name=LAKEBASE_INSTANCE_NAME)
try:
    checkpointer.setup()
except Exception as e:
    if "already exists" not in str(e).lower():
        raise

agent_with_checkpointer = workflow.compile(checkpointer=checkpointer)

print("✓ Checkpointed RAG agent built (short-term memory only)")
print("✓ Prompt template added")

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
store = DatabricksStore(
    instance_name=LAKEBASE_INSTANCE_NAME,
    embedding_endpoint=MEMORY_EMBEDDING_ENDPOINT,
    embedding_dims=MEMORY_EMBEDDING_DIMS,
)
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
store_backed_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a beginner-friendly Databricks coach helping users answer questions "
        "about company policy documents. Use retrieved context when it is available, "
        "use the conversation history for follow-up questions, and incorporate user "
        "context from long-term memory when it is provided. Cite source file names "
        "when the tool provides them, do not make up policy details, and answer in 2 "
        "short bullet points followed by one practical next step.\n\n"
        "Known user context:\n{memory_context}"
    ),
    MessagesPlaceholder("messages"),
])

def call_agent_with_store(state: MessagesState):
    """Agent logic that reads from Store for personalization."""
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

    memory_context = "; ".join(context_parts) if context_parts else "No saved user context."
    prompt_messages = store_backed_prompt.invoke({
        "memory_context": memory_context,
        "messages": state["messages"],
    }).messages
    response = llm_with_tools.invoke(prompt_messages)
    return {"messages": [response]}

# Build new workflow with Store-enabled agent
workflow_with_store = StateGraph(MessagesState)
workflow_with_store.add_node("agent", call_agent_with_store)
workflow_with_store.add_node("tools", ToolNode(tools))
workflow_with_store.set_entry_point("agent")
workflow_with_store.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow_with_store.add_edge("tools", "agent")

# Compile with CheckpointSaver; the manual example reads from the global store above
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
# MAGIC ## Step 10: Limitations of Manual Memory
# MAGIC
# MAGIC The agent from Step 8 works, but the manual approach has real problems:
# MAGIC
# MAGIC | Limitation | Why It Matters |
# MAGIC |-----------|----------------|
# MAGIC | **Hardcoded keys** | Developer must anticipate every fact type (`department`, `role`, ...) |
# MAGIC | **No semantic search** | `store.get(ns, "department")` only works with exact keys — can't ask "what do I know about this user?" |
# MAGIC | **No user control** | User can't ask the agent to forget something (GDPR!) |
# MAGIC | **Doesn't scale** | Every new fact type requires code changes |
# MAGIC | **Agent can't learn** | The LLM never decides *what's worth remembering* — the developer decides at build time |
# MAGIC
# MAGIC The production pattern flips this: **give the LLM tools to manage memory itself.**
# MAGIC The agent decides when to read, write, and delete memories — just like it decides when to search documents.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Build Memory Tools
# MAGIC
# MAGIC Instead of hardcoding memory reads/writes in the agent node, we define three
# MAGIC tools that the LLM can call on its own:
# MAGIC
# MAGIC | Tool | Purpose | Store Method |
# MAGIC |------|---------|-------------|
# MAGIC | `get_user_memory` | Recall facts about the user | `store.search()` (semantic) |
# MAGIC | `save_user_memory` | Remember new facts | `store.put()` |
# MAGIC | `delete_user_memory` | Forget specific facts | `store.delete()` |
# MAGIC
# MAGIC Key differences from the manual approach:
# MAGIC - **Semantic search** (`store.search`) instead of exact key lookup (`store.get`)
# MAGIC - **Store passed via config** (`RunnableConfig`) — not a global variable
# MAGIC - **User-scoped** — `user_id` flows through the graph config

# COMMAND ----------

@tool
def get_user_memory(query: str, config: RunnableConfig) -> str:
    """Search for relevant information about the user from long-term memory.
    Use this to recall facts, preferences, or details previously shared by the user."""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Memory not available - no user_id provided."

    store_ref = config.get("configurable", {}).get("store")
    if not store_ref:
        return "Memory not available - store not configured."

    namespace = ("user_memories", sanitize_namespace_id(user_id))
    results = store_ref.search(namespace, query=query, limit=5)

    if not results:
        return "No memories found for this user."

    items = [f"- [{item.key}]: {json.dumps(item.value)}" for item in results]
    return f"Found {len(results)} relevant memories:\n" + "\n".join(items)


@tool
def save_user_memory(memory_key: str, memory_data_json: str, config: RunnableConfig) -> str:
    """Save information about the user to long-term memory.
    Use this when the user shares facts, preferences, or important details worth remembering."""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Cannot save memory - no user_id provided."

    store_ref = config.get("configurable", {}).get("store")
    if not store_ref:
        return "Cannot save memory - store not configured."

    namespace = ("user_memories", sanitize_namespace_id(user_id))
    try:
        memory_data = json.loads(memory_data_json)
        if not isinstance(memory_data, dict):
            return f"Failed: memory_data must be a JSON object, not {type(memory_data).__name__}"
        store_ref.put(namespace, memory_key, memory_data)
        return f"Successfully saved memory '{memory_key}' for user."
    except json.JSONDecodeError as e:
        return f"Failed to save memory: Invalid JSON - {e}"


@tool
def delete_user_memory(memory_key: str, config: RunnableConfig) -> str:
    """Delete a specific memory from the user's long-term memory.
    Use this when the user asks to forget specific information."""
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return "Cannot delete memory - no user_id provided."

    store_ref = config.get("configurable", {}).get("store")
    if not store_ref:
        return "Cannot delete memory - store not configured."

    namespace = ("user_memories", sanitize_namespace_id(user_id))
    store_ref.delete(namespace, memory_key)
    return f"Successfully deleted memory '{memory_key}' for user."


memory_tools_list = [get_user_memory, save_user_memory, delete_user_memory]
print("✓ Memory tools defined:")
for t in memory_tools_list:
    print(f"  • {t.name}: {t.description.splitlines()[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 12: Build LLM-Managed Memory Agent
# MAGIC
# MAGIC Now we wire the memory tools into the agent alongside the RAG tool.
# MAGIC The system prompt tells the LLM *how* to use memory — but the LLM
# MAGIC decides *when* and *what* to remember.

# COMMAND ----------

MEMORY_SYSTEM_PROMPT = """You are a helpful assistant with access to company policy documents and long-term memory.

You have memory tools that let you remember information about users across conversations:
- Use get_user_memory to search for previously saved information about the user
- Use save_user_memory to remember important facts, preferences, or details the user shares
- Use delete_user_memory to forget specific information when asked

Guidelines:
1. At the start of a conversation, check for relevant memories to personalize your response
2. When the user shares personal details (department, role, preferences), save them
3. When the user asks you to forget something, delete that specific memory
4. Use search_policy_documents for company policy questions
5. Cite source file names when document search results include them
6. Do not make up policy details or memory facts that you did not retrieve
7. Answer in 2 short bullet points and end with one practical next step"""

all_tools = [search_policy_documents] + memory_tools_list
llm_with_all_tools = llm.bind_tools(all_tools)
memory_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", MEMORY_SYSTEM_PROMPT),
    MessagesPlaceholder("messages"),
])

def call_memory_agent(state: MessagesState, config: RunnableConfig):
    prompt_messages = memory_agent_prompt.invoke({"messages": state["messages"]}).messages
    response = llm_with_all_tools.invoke(prompt_messages, config)
    return {"messages": [response]}

workflow_memory = StateGraph(MessagesState)
workflow_memory.add_node("agent", call_memory_agent)
workflow_memory.add_node("tools", ToolNode(all_tools))
workflow_memory.set_entry_point("agent")
workflow_memory.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow_memory.add_edge("tools", "agent")

memory_agent = workflow_memory.compile(checkpointer=checkpointer)

print("✓ LLM-managed memory agent built")
print(f"  Tools: {[t.name for t in all_tools]}")
print("  Short-term: CheckpointSaver | Long-term: DatabricksStore via memory tools")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 13: Demo — LLM-Managed Memory in Action
# MAGIC
# MAGIC Three scenarios show how the LLM autonomously manages memory:
# MAGIC 1. **Auto-save** — User shares info, LLM decides to call `save_user_memory`
# MAGIC 2. **Auto-recall** — New session, LLM calls `get_user_memory` on its own
# MAGIC 3. **Auto-delete** — User asks to forget something, LLM calls `delete_user_memory`

# COMMAND ----------

# Scenario 1: User shares info → LLM saves it autonomously
user_id = "alice@company.com"
config_s1 = {
    "configurable": {
        "thread_id": "llm-memory-session-1",
        "store": store,
        "user_id": user_id,
    }
}

print("SESSION 1 — User shares personal info:")
print("=" * 80)
result_s1 = memory_agent.invoke(
    {"messages": [HumanMessage(content=(
        "Hi! I work in the Engineering department as a Senior ML Engineer. "
        "I prefer morning meetings and I'm interested in cloud architecture."
    ))]},
    config_s1,
)
print("User: Hi! I work in the Engineering department as a Senior ML Engineer. ...")
print(f"Agent: {result_s1['messages'][-1].content[:300]}...")

# Inspect what the LLM stored
print("\n--- Memories the agent saved ---")
ns = ("user_memories", sanitize_namespace_id(user_id))
saved_items = store.search(ns, query="user information", limit=10)
for item in saved_items:
    print(f"  [{item.key}]: {item.value}")
if not saved_items:
    print("  (none yet — the LLM may batch saves differently)")

# COMMAND ----------

# Scenario 2: New session → LLM recalls memories on its own
config_s2 = {
    "configurable": {
        "thread_id": "llm-memory-session-2",
        "store": store,
        "user_id": user_id,
    }
}

print("\nSESSION 2 — New thread, agent recalls memories:")
print("=" * 80)
result_s2 = memory_agent.invoke(
    {"messages": [HumanMessage(content="What professional development should I focus on?")]},
    config_s2,
)
print("User: What professional development should I focus on?")
print(f"Agent: {result_s2['messages'][-1].content[:300]}...")
print("\n→ Compare with Step 5: the agent now remembers Engineering + ML Engineer from a previous session!")

# COMMAND ----------

# Scenario 3: User asks to forget info → LLM deletes it
config_s3 = {
    "configurable": {
        "thread_id": "llm-memory-session-3",
        "store": store,
        "user_id": user_id,
    }
}

print("\nSESSION 3 — User asks to forget info:")
print("=" * 80)
result_s3 = memory_agent.invoke(
    {"messages": [HumanMessage(content="Please forget my meeting time preference.")]},
    config_s3,
)
print("User: Please forget my meeting time preference.")
print(f"Agent: {result_s3['messages'][-1].content[:300]}...")

# Verify deletion
print("\n--- Remaining memories ---")
remaining = store.search(ns, query="meeting preference", limit=5)
for item in remaining:
    print(f"  [{item.key}]: {item.value}")
if not remaining:
    print("  ✓ Meeting preference deleted")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What We Built — Two Approaches
# MAGIC
# MAGIC | | Manual (Steps 4–9) | LLM-Managed (Steps 11–13) |
# MAGIC |---|---|---|
# MAGIC | **Who decides what to remember** | Developer (hardcoded keys) | The LLM (tool calls) |
# MAGIC | **Memory retrieval** | Exact key: `store.get(ns, "department")` | Semantic search: `store.search(ns, query=...)` |
# MAGIC | **Memory deletion** | Not supported | `delete_user_memory` tool |
# MAGIC | **Scales to new fact types** | No — requires code changes | Yes — LLM picks keys and values |
# MAGIC | **User control** | None | User can ask to save/forget |
# MAGIC
# MAGIC The **LLM-managed pattern** is what the Databricks app templates use in production.
# MAGIC
# MAGIC ### Three Memory Layers
# MAGIC
# MAGIC | Layer | Primitive | Scope | Example |
# MAGIC |-------|-----------|-------|---------|
# MAGIC | **Vector Search** | Databricks Vector Search | All users, static knowledge | "Company policy allows 15 days vacation" |
# MAGIC | **Short-term** | `CheckpointSaver` | Single thread | "You asked about time off earlier in this chat" |
# MAGIC | **Long-term** | `DatabricksStore` via tools | Across all sessions | "You work in Engineering" (remembered forever) |
# MAGIC
# MAGIC ### Key Concepts
# MAGIC
# MAGIC **Memory Tools Pattern:**
# MAGIC ```python
# MAGIC @tool
# MAGIC def get_user_memory(query: str, config: RunnableConfig) -> str:
# MAGIC     store = config.get("configurable", {}).get("store")
# MAGIC     results = store.search(namespace, query=query, limit=5)
# MAGIC     ...
# MAGIC ```
# MAGIC
# MAGIC **Store passed via config** — not a global variable:
# MAGIC ```python
# MAGIC config = {"configurable": {"store": store, "user_id": user_id, "thread_id": "..."}}
# MAGIC agent.invoke({"messages": [...]}, config)
# MAGIC ```
# MAGIC
# MAGIC **Namespace Patterns:**
# MAGIC ```python
# MAGIC ("user_memories", sanitized_user_id)       # per-user memories
# MAGIC ("user_facts", sanitized_user_id)           # structured facts
# MAGIC ("team_knowledge", "engineering")            # shared context
# MAGIC ```
# MAGIC
# MAGIC **Production Considerations:**
# MAGIC - Memory deletion enables GDPR/privacy compliance
# MAGIC - Use `AsyncDatabricksStore` in async app contexts (FastAPI)
# MAGIC - Add confidence scores or timestamps to stored facts
# MAGIC - Implement access controls and audit logging
# MAGIC - Consider data retention policies for stale memories
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC Continue to **Module 03: Evaluation** to add observability, tracing, and systematic testing.

# COMMAND ----------

print("✓ Long-term memory tutorial complete!")
print("\nYour agent now has:")
print("  ✓ Short-term: Conversation context (CheckpointSaver)")
print("  ✓ Long-term: LLM-managed personalization (DatabricksStore + memory tools)")
print("  ✓ Knowledge: Document search (Vector Search)")
print("  ✓ User control: Save, recall, and delete memories on demand")
