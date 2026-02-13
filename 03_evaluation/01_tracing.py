# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Tracing: Observability for AI Agents
# MAGIC
# MAGIC ## What You'll Build
# MAGIC Add comprehensive observability to your memory-enabled agent with MLflow tracing.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Enable MLflow tracing for full observability
# MAGIC - Instrument code with @mlflow.trace decorator
# MAGIC - Inspect traces (spans, latency, tool calls, errors)
# MAGIC - Add custom attributes for debugging
# MAGIC - Use traces to identify performance bottlenecks
# MAGIC - Apply production tracing features (async logging, sampling)
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed Module 02 (Memory) OR understand checkpointed RAG agents
# MAGIC - Vector Search index available
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
from databricks_langchain import ChatDatabricks, CheckpointSaver
from databricks.vector_search.client import VectorSearchClient
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

# COMMAND ----------

# Configuration
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
LLM_ENDPOINT = "databricks-claude-sonnet-4-5"
LAKEBASE_PROJECT = "knowledge-assistant-state"

VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.policy_index"
ENDPOINT_NAME = "agent_bootcamp_endpoint"

print(f"✓ Configuration loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Build Agent (Quick Recap from Module 02)
# MAGIC
# MAGIC We'll rebuild the memory-enabled RAG agent from Module 02 as a starting point.

# COMMAND ----------

# Create Vector Search tool
@tool
def search_policy_documents(query: str) -> str:
    """Search company policy documents for information."""
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

agent = workflow.compile(checkpointer=checkpointer)

print("✓ Memory-enabled RAG agent built")
print("  Now let's add observability and evaluation!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Enable MLflow Tracing
# MAGIC
# MAGIC Tracing captures execution details: inputs, outputs, latency, tool calls, and errors.

# COMMAND ----------

# Set experiment
experiment_name = f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/agent_bootcamp_evaluation"
mlflow.set_experiment(experiment_name)

# Enable tracing globally
mlflow.tracing.enable()

print("✓ MLflow tracing enabled")
print(f"  Experiment: {experiment_name}")
print(f"  All LangGraph calls will be automatically traced")

# COMMAND ----------

# Test agent with auto-tracing
thread_id = "eval-test-001"
config = {"configurable": {"thread_id": thread_id}}

result = agent.invoke(
    {"messages": [HumanMessage(content="How much vacation time do employees get?")]},
    config=config
)

print("✓ Agent executed with tracing")
print(f"  Response: {result['messages'][-1].content[:150]}...")
print(f"\n→ Check MLflow UI to see the trace with spans, latency, and tool calls")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Manual Instrumentation with @mlflow.trace
# MAGIC
# MAGIC Add custom tracing to your own functions for better observability.

# COMMAND ----------

@mlflow.trace(name="format_context")
def format_context(docs: list) -> str:
    """Format retrieved documents for LLM context."""
    import time
    time.sleep(0.05)  # Simulate processing
    return "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs)])

@mlflow.trace(name="custom_rag_pipeline")
def custom_rag_pipeline(query: str):
    """Custom RAG pipeline with detailed tracing."""
    # This will appear as nested spans in the trace
    # Note: search_policy_documents is a Tool, so use .invoke()
    docs = search_policy_documents.invoke(query)
    context = format_context([docs])

    # Add custom attributes
    span = mlflow.get_current_active_span()
    if span:
        span.set_attribute("query_length", len(query))
        span.set_attribute("context_length", len(context))
        span.set_attribute("num_docs", 1)

    return f"Based on context: {context[:100]}..."

# Test custom tracing
result = custom_rag_pipeline("What is the remote work policy?")
print("✓ Custom pipeline traced with nested spans")
print(f"  Result: {result[:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Inspect Traces
# MAGIC
# MAGIC Analyze traces to understand performance and identify issues.

# COMMAND ----------

# Search for recent traces
experiment = mlflow.get_experiment_by_name(experiment_name)
traces = mlflow.search_traces(
    experiment_ids=[experiment.experiment_id],
    max_results=3,
    order_by=["timestamp_ms DESC"]
)

print(f"Found {len(traces)} recent traces")
print("\nTrace Overview:")
print("=" * 80)

for idx, row in traces.iterrows():
    print(f"\nTrace ID: {row['trace_id']}")
    print(f"  Status: {row['state']}")
    print(f"  Duration: {row['execution_duration']:.2f}ms")

    # Count spans by type
    span_types = {}
    for span in row['spans']:
        span_type = span.get('span_type') or "UNKNOWN"
        span_types[span_type] = span_types.get(span_type, 0) + 1

    print(f"  Spans: {span_types}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Using Traces for Debugging
# MAGIC
# MAGIC Use traces to identify and fix issues in your agent.

# COMMAND ----------

# Example: Create a separate retrieval function for debugging
@mlflow.trace
def debug_retrieval(query: str) -> dict:
    """Enhanced retrieval with debugging info (not using the tool)."""
    vsc = VectorSearchClient(disable_notice=True)
    index = vsc.get_index(ENDPOINT_NAME, VECTOR_INDEX)

    results = index.similarity_search(
        query_text=query,
        columns=["source_file", "chunk_text"],
        num_results=3
    )

    # Extract results
    docs = results.get("result", {}).get("data_array", [])

    # Add debugging attributes
    span = mlflow.get_current_active_span()
    if span:
        span.set_attribute("num_results", len(docs))
        if docs and len(docs) > 0:
            span.set_attribute("has_results", True)

    return {
        "documents": docs,
        "count": len(docs),
        "query": query
    }

# Test improved retrieval
debug_info = debug_retrieval("vacation policy")
print(f"✓ Retrieved {debug_info['count']} documents")
print(f"  Query: {debug_info['query']}")
print(f"\n→ Check trace to see debugging attributes")
print(f"→ This demonstrates adding custom attributes for debugging")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Production Tracing Features
# MAGIC
# MAGIC Enable production-ready tracing features for high-traffic scenarios.

# COMMAND ----------

# Enable async logging (non-blocking)
mlflow.config.enable_async_logging()
print("✓ Async logging enabled")
print("  Traces logged in background without blocking agent")

# Sampling for high-traffic scenarios
import random

def should_trace_request():
    """Sample 10% of requests in production."""
    return random.random() < 0.1

# Example usage pattern
for i in range(5):
    if should_trace_request():
        # Trace this request
        with mlflow.start_span(name=f"sampled_request_{i}"):
            # Your agent logic here
            pass
        print(f"  Request {i}: Traced")
    else:
        # Skip tracing
        print(f"  Request {i}: Skipped (sampling)")

print("\n→ Use sampling to reduce trace volume in production")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What We Built
# MAGIC - ✅ Enabled MLflow tracing for full observability
# MAGIC - ✅ Added manual instrumentation with @mlflow.trace
# MAGIC - ✅ Inspected traces (spans, latency, tool calls)
# MAGIC - ✅ Used traces to identify performance issues
# MAGIC - ✅ Added custom attributes for debugging
# MAGIC - ✅ Applied production features (async logging, sampling)
# MAGIC
# MAGIC ### Key Concepts
# MAGIC
# MAGIC **Tracing Basics:**
# MAGIC - **Auto-tracing**: Automatic for LangGraph, LangChain, OpenAI
# MAGIC - **Manual instrumentation**: Use `@mlflow.trace` decorator or `mlflow.start_span()` context manager
# MAGIC - **Captures**: Inputs, outputs, latency, tool calls, errors, custom attributes
# MAGIC
# MAGIC **What Traces Show:**
# MAGIC - Execution flow (which tools were called, in what order)
# MAGIC - Performance (latency per span, total duration)
# MAGIC - Tool behavior (inputs/outputs, errors)
# MAGIC - Custom debugging info (via span attributes)
# MAGIC
# MAGIC **Using Traces for Debugging:**
# MAGIC 1. Agent giving wrong answers → Check retrieved documents in trace
# MAGIC 2. Slow responses → Identify bottleneck spans
# MAGIC 3. Tool errors → See exact error in span events
# MAGIC 4. Unexpected behavior → Inspect full execution flow
# MAGIC
# MAGIC ### Production Tracing Checklist
# MAGIC - ✅ Enable async logging (`mlflow.config.enable_async_logging()`)
# MAGIC - ✅ Implement sampling for high traffic (10-20%)
# MAGIC - ✅ Add custom attributes for business context
# MAGIC - ✅ Monitor trace data for anomalies
# MAGIC - ✅ Set up alerts on error rates
# MAGIC
# MAGIC ### Trace Analysis Patterns
# MAGIC
# MAGIC ```python
# MAGIC # Find slow requests
# MAGIC traces = mlflow.search_traces(filter_string="duration > 5000")  # >5 seconds
# MAGIC
# MAGIC # Find errors
# MAGIC traces = mlflow.search_traces(filter_string="status = 'ERROR'")
# MAGIC
# MAGIC # Analyze by user
# MAGIC span.set_attribute("user_id", user_id)  # Tag traces with user
# MAGIC traces = mlflow.search_traces(filter_string="attributes.user_id = 'alice'")
# MAGIC ```
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC Continue to [02_evaluation.py](02_evaluation.py) to add **systematic evaluation** with custom judges for domain-specific quality assessment.
# MAGIC
# MAGIC **What's in Notebook 02:**
# MAGIC - Create evaluation datasets
# MAGIC - Build custom judges (accuracy, completeness, tone)
# MAGIC - Run comprehensive evaluation
# MAGIC - Set up quality gates
# MAGIC - Human feedback loops

# COMMAND ----------

print("✓ MLflow Tracing tutorial complete!")
print("\nYour agent now has:")
print("  ✓ Full observability with MLflow tracing")
print("  ✓ Tools to debug and optimize performance")
print("  ✓ Production-ready tracing features")
print("\nNext: Open 02_evaluation.py to add systematic evaluation")
