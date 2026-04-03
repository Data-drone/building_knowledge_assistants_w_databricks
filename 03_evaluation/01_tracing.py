# Databricks notebook source
# MAGIC %md
# MAGIC # Trace One Run Before You Score Many
# MAGIC
# MAGIC ## What You'll Build
# MAGIC Use MLflow tracing to inspect one execution of the same memory-capable HR bot
# MAGIC you carried forward from Module 02 and `00_first_eval_loop.py`.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Enable MLflow tracing for a LangGraph agent
# MAGIC - Inspect one run end-to-end before doing batch evaluation
# MAGIC - Add targeted `@mlflow.trace` spans for custom debugging
# MAGIC - Use trace search to answer practical debugging questions
# MAGIC - Apply production-ready tracing patterns such as async logging and sampling
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed [00_first_eval_loop.py](00_first_eval_loop.py)
# MAGIC - Completed Module 02 (Memory) OR understand memory-capable LangGraph agents
# MAGIC
# MAGIC ## Estimated Time
# MAGIC 15 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies
# MAGIC
# MAGIC We install the same core pieces used across the bootcamp:
# MAGIC
# MAGIC - `mlflow[databricks]` for tracing and evaluation APIs
# MAGIC - `databricks-langchain[memory]` for the checkpointed agent pattern
# MAGIC - `langgraph` for the graph itself
# MAGIC
# MAGIC **When this matters:** if traces are missing or the wrong integrations appear in
# MAGIC MLflow, version mismatches are one of the first things to check.

# COMMAND ----------

# MAGIC %pip install -q --upgrade \
# MAGIC   "databricks-sdk>=0.101,<0.103" \
# MAGIC   "mlflow[databricks]>=3.10,<3.11" \
# MAGIC   "databricks-langchain[memory]>=0.17,<0.18" \
# MAGIC   "databricks-vectorsearch>=0.66,<0.67" \
# MAGIC   "langgraph>=1.1,<1.2" \
# MAGIC   "langchain-core>=1.2,<2"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configuration and Imports
# MAGIC
# MAGIC This step keeps the notebook self-contained: the catalog, schema, endpoint, and
# MAGIC experiment path are all visible in one place.
# MAGIC
# MAGIC **When this matters:** tracing bugs are often not really tracing bugs. They are
# MAGIC often configuration bugs:
# MAGIC - traces logging to the wrong experiment
# MAGIC - the wrong vector index being queried
# MAGIC - a notebook accidentally pointing at a different environment than expected

# COMMAND ----------

import json
import random
import re
import uuid

import mlflow
from databricks_langchain import ChatDatabricks, CheckpointSaver, VectorSearchRetrieverTool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# COMMAND ----------

# Hardcoded notebook config keeps Module 03 self-contained and easy to debug.
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"
LAKEBASE_INSTANCE_NAME = None
LAKEBASE_AUTOSCALING_PROJECT = "knowledge-assistant-state"
LAKEBASE_AUTOSCALING_BRANCH = "production"
VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.policy_index"

TRACING_RUN_ID = uuid.uuid4().hex[:8]
EXPERIMENT_NAME = (
    f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/"
    "agent_bootcamp_evaluation"
)

DEMO_USER_MEMORIES: dict[str, dict[str, dict[str, str]]] = {
    "trace-demo-user": {
        "department": {"department": "Engineering"},
        "role": {"role": "manager"},
    }
}


def lakebase_target() -> str:
    if LAKEBASE_INSTANCE_NAME:
        return LAKEBASE_INSTANCE_NAME
    return f"{LAKEBASE_AUTOSCALING_PROJECT}/{LAKEBASE_AUTOSCALING_BRANCH}"


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


mlflow.langchain.autolog()

print("✓ Configuration loaded")
print(f"  Experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Rebuild the Same Bot Shape from Module 02
# MAGIC
# MAGIC We keep the same prompted LangGraph layout from the memory section:
# MAGIC document retrieval plus memory tools in one graph. The example below still
# MAGIC focuses on policy Q&A, but the graph shape now matches the story learners have
# MAGIC already seen.
# MAGIC
# MAGIC This is important for tracing because the value of a trace depends on whether it
# MAGIC reflects the *real* system you care about. If we switched to a simplified agent
# MAGIC here, the trace might be easier to read but much less useful.
# MAGIC
# MAGIC **When this matters:** use this kind of tracing setup when you want to answer
# MAGIC questions like:
# MAGIC - "Did the agent choose the right tool?"
# MAGIC - "Did memory affect the answer?"
# MAGIC - "Was the retriever called before the final response?"

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


def memory_bucket(config: RunnableConfig):
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return None, "Memory not available - no user_id provided."
    return DEMO_USER_MEMORIES.setdefault(user_id, {}), None


@tool
def get_user_memory(query: str, config: RunnableConfig) -> str:
    """Search for saved user details that may help personalize the response."""
    bucket, error = memory_bucket(config)
    if error:
        return error

    query_text = normalize_text(query)
    matches = []
    for memory_key, memory_value in bucket.items():
        searchable = normalize_text(json.dumps(memory_value))
        if query_text in searchable or any(token in searchable for token in query_text.split()):
            matches.append(f"- [{memory_key}]: {json.dumps(memory_value)}")

    if not matches:
        return "No memories found for this user."
    return "Found relevant memories:\n" + "\n".join(matches)


@tool
def save_user_memory(memory_key: str, memory_data_json: str, config: RunnableConfig) -> str:
    """Save durable user details such as role, department, or preferences."""
    bucket, error = memory_bucket(config)
    if error:
        return error

    try:
        memory_data = json.loads(memory_data_json)
    except json.JSONDecodeError as exc:
        return f"Could not save memory: invalid JSON ({exc})"

    if not isinstance(memory_data, dict):
        return "Could not save memory: memory_data_json must decode to a JSON object."

    bucket[memory_key] = memory_data
    return f"Saved memory '{memory_key}' for this user."


@tool
def delete_user_memory(memory_key: str, config: RunnableConfig) -> str:
    """Delete a saved memory when the user asks to forget something."""
    bucket, error = memory_bucket(config)
    if error:
        return error

    if memory_key not in bucket:
        return f"No memory named '{memory_key}' was found."

    del bucket[memory_key]
    return f"Deleted memory '{memory_key}'."


MEMORY_SYSTEM_PROMPT = """You are a beginner-friendly Databricks coach with access to company policy documents and long-term memory tools.

Use tools deliberately:
- Use search_policy_documents for policy questions
- Use get_user_memory to recall relevant saved user details
- Use save_user_memory when the user shares durable facts or preferences
- Use delete_user_memory when the user asks you to forget something

Answering rules:
1. Use retrieved policy context when available
2. Personalize with memory only when it is relevant
3. Cite source file names when the retriever includes them
4. Do not invent policy details or memory facts
5. Answer in 2 short bullet points and end with one practical next step"""

llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1)
all_tools = [search_policy_documents, get_user_memory, save_user_memory, delete_user_memory]
llm_with_tools = llm.bind_tools(all_tools)
memory_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", MEMORY_SYSTEM_PROMPT),
        MessagesPlaceholder("messages"),
    ]
)


def call_memory_agent(state: MessagesState, config: RunnableConfig):
    prompt_messages = memory_agent_prompt.invoke({"messages": state["messages"]}).messages
    response = llm_with_tools.invoke(prompt_messages, config)
    return {"messages": [response]}


def should_continue(state: MessagesState):
    last_msg = state["messages"][-1]
    return "tools" if getattr(last_msg, "tool_calls", None) else END


workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_memory_agent)
workflow.add_node("tools", ToolNode(all_tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

checkpointer = CheckpointSaver(
    instance_name=LAKEBASE_INSTANCE_NAME,
    project=LAKEBASE_AUTOSCALING_PROJECT,
    branch=LAKEBASE_AUTOSCALING_BRANCH,
)
try:
    checkpointer.setup()
except Exception as exc:
    if "already exists" not in str(exc).lower():
        raise Exception(
            f"CheckpointSaver setup failed for {lakebase_target()}. "
            "Verify the Lakebase target exists and your role has CREATE access on schema public."
        ) from exc

agent = workflow.compile(checkpointer=checkpointer)

print("✓ Memory-capable bot rebuilt for tracing")
print(f"  Tools: {[tool.name for tool in all_tools]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Enable MLflow Tracing
# MAGIC
# MAGIC Tracing answers a different question than the toy loop:
# MAGIC instead of asking \"did this answer pass?\", we ask
# MAGIC \"what happened inside this specific run?\"
# MAGIC
# MAGIC After tracing is enabled, MLflow starts capturing the execution tree for traced
# MAGIC frameworks like LangGraph and LangChain.
# MAGIC
# MAGIC **When this matters:** this is the first thing to turn on when:
# MAGIC - the agent gives a surprising answer
# MAGIC - the response is slower than expected
# MAGIC - you need to see whether a tool was called at all

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.tracing.enable()

print("✓ MLflow tracing enabled")
print(f"  Experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Inspect One Run End-to-End
# MAGIC
# MAGIC Start with a single question and a single run before looking at averages or
# MAGIC score tables. One trace is easier to reason about than fifty.
# MAGIC
# MAGIC Here the main thing to inspect is the LangGraph trace tree for `agent.invoke()`.
# MAGIC Look for:
# MAGIC - the root run for the invocation
# MAGIC - model/tool child spans
# MAGIC - the order in which the graph steps executed
# MAGIC
# MAGIC **When this matters:** this is the right technique when someone says:
# MAGIC - "The answer looks wrong, but I don't know why."
# MAGIC - "I expected the retriever to run, but maybe it didn't."
# MAGIC - "I want to understand one real execution before I score many."

# COMMAND ----------

trace_config = {
    "configurable": {
        "thread_id": f"trace-demo-{TRACING_RUN_ID}",
        "user_id": "trace-demo-user",
    }
}

traced_result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                content="What professional development budget should I know about for managers?"
            )
        ]
    },
    config=trace_config,
)

print("✓ Agent executed with tracing")
print(f"  Thread ID: {trace_config['configurable']['thread_id']}")
print(f"  Response: {traced_result['messages'][-1].content[:180]}...")
print("\n→ Open the MLflow trace for this agent.invoke() call to inspect the LangGraph span tree.")
print("→ You should now see the LangGraph trace as the main expandable run in MLflow.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Add Manual Instrumentation with `@mlflow.trace`
# MAGIC
# MAGIC Auto-tracing covers the graph. Manual spans help when you want to inspect your
# MAGIC own helper logic in more detail.
# MAGIC
# MAGIC Framework traces usually show the big pieces well, but they do not always reveal
# MAGIC enough about your own formatting, parsing, routing, or post-processing logic.
# MAGIC That is where `@mlflow.trace` is useful.
# MAGIC
# MAGIC **When this matters:** add manual spans for helpers when:
# MAGIC - retrieval formatting might be dropping important context
# MAGIC - a parser or adapter is transforming data incorrectly
# MAGIC - you want custom attributes like query length, context length, or branch choice

# COMMAND ----------

@mlflow.trace(name="format_retrieved_context")
def format_retrieved_context(raw_context: str) -> str:
    sections = [part.strip() for part in raw_context.split("\n\n") if part.strip()]
    return "\n".join(f"- {section[:180]}" for section in sections)


@mlflow.trace(name="inspect_policy_context")
def inspect_policy_context(query: str) -> str:
    raw_context = search_policy_documents.invoke(query)
    formatted_context = format_retrieved_context(raw_context)

    span = mlflow.get_current_active_span()
    if span:
        span.set_attribute("query_length", len(query))
        span.set_attribute("formatted_context_length", len(formatted_context))
        span.set_attribute("has_context", formatted_context != "")

    return formatted_context


formatted_context = inspect_policy_context("remote work policy for hybrid employees")
print("✓ Manual tracing added for helper functions")
print(formatted_context[:300] + "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Search Recent Traces
# MAGIC
# MAGIC Once you know what one good run looks like, `search_traces()` lets you inspect
# MAGIC patterns across recent runs without opening each trace manually.
# MAGIC
# MAGIC This is the bridge between "understand one run" and "monitor many runs." You are
# MAGIC still not grading quality yet, but you are starting to summarize operational
# MAGIC behavior across multiple executions.
# MAGIC
# MAGIC **When this matters:** use `search_traces()` when you want to answer questions like:
# MAGIC - "How many recent runs called tools?"
# MAGIC - "Are the latest runs slower than before?"
# MAGIC - "Did several runs fail in the same way?"

# COMMAND ----------

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
traces = mlflow.search_traces(
    experiment_ids=[experiment.experiment_id],
    max_results=5,
    order_by=["timestamp_ms DESC"],
)

print(f"Found {len(traces)} recent traces")
print("=" * 80)
for _, row in traces.iterrows():
    span_types = {}
    for span in row["spans"]:
        span_type = span.get("span_type") or "UNKNOWN"
        span_types[span_type] = span_types.get(span_type, 0) + 1

    print(f"Trace ID: {row['trace_id']}")
    print(f"  State: {row['state']}")
    print(f"  Duration: {row['execution_duration']:.2f} ms")
    print(f"  Span types: {span_types}")
    print("-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Production Tracing Patterns
# MAGIC
# MAGIC Once tracing works in a notebook, the production question becomes: "How do I keep
# MAGIC the signal without creating too much cost or noise?"
# MAGIC
# MAGIC Good news: MLflow trace export is already asynchronous by default — traces
# MAGIC are batched and sent in the background, so tracing adds minimal latency to
# MAGIC your agent calls.
# MAGIC
# MAGIC The main lever you control is **sampling**: trace a representative slice of
# MAGIC requests instead of every single one. The manual approach below shows the
# MAGIC idea. In production, MLflow's `ScorerSamplingConfig` (covered in Module 05)
# MAGIC handles this for you when you register scorers for continuous monitoring.
# MAGIC
# MAGIC **When this matters:** these patterns are useful when:
# MAGIC - you are moving from a demo notebook to a live app
# MAGIC - you want observability without storing every single request
# MAGIC - you need to control cost for LLM-based scorers on live traffic

# COMMAND ----------

def should_trace_request() -> bool:
    return random.random() < 0.1


for idx in range(5):
    if should_trace_request():
        with mlflow.start_span(name=f"sampled_request_{idx}"):
            pass
        print(f"Request {idx}: traced")
    else:
        print(f"Request {idx}: skipped by sampling")

print("\n→ Sampling keeps production tracing costs and volume under control.")
print("→ In Module 05, ScorerSamplingConfig handles this declaratively.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What We Built
# MAGIC - ✅ Reused the same memory-capable bot shape from Module 02
# MAGIC - ✅ Traced one end-to-end agent run
# MAGIC - ✅ Added manual helper spans with `@mlflow.trace`
# MAGIC - ✅ Searched recent traces to inspect run patterns
# MAGIC - ✅ Learned production tracing patterns: async export and sampling
# MAGIC
# MAGIC ### Why Tracing Comes Before Batch Evaluation
# MAGIC
# MAGIC Tracing gives you the *inside view* of one execution:
# MAGIC - which tools were called
# MAGIC - what context was retrieved
# MAGIC - where latency was spent
# MAGIC - which spans might explain a weak answer
# MAGIC
# MAGIC That makes the next step much more meaningful: once you understand one run,
# MAGIC you are ready to score **many** runs with built-in MLflow scorers.
# MAGIC
# MAGIC ## Next Step
# MAGIC
# MAGIC Continue to [02_evaluation.py](02_evaluation.py) to:
# MAGIC - generate predictions for a small eval set
# MAGIC - score them with built-in scorers first
# MAGIC - add custom judges only when built-ins stop being enough

# COMMAND ----------

print("✓ Tracing notebook complete!")
print("Next: open 02_evaluation.py to batch-score many runs with built-in scorers first.")
