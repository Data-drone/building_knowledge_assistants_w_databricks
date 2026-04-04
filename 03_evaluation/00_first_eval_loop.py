# Databricks notebook source
# MAGIC %md
# MAGIC # First Evaluation Loop
# MAGIC
# MAGIC ## What You'll Build
# MAGIC Start Module 03 with a small Q&A evaluation loop for the same memory-capable HR bot
# MAGIC from Module 02. You will ask a few policy questions, compare the answers to a
# MAGIC lightweight MLflow evaluation setup, and see why built-in scorers are a great
# MAGIC starting point but not the full story.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Reuse the same production-style bot shape from Module 02
# MAGIC - Create a **managed evaluation dataset** backed by Unity Catalog
# MAGIC - Use `predict_fn` + `mlflow.genai.evaluate()` to generate, trace, and score in one call
# MAGIC - Run a couple of built-in scorers and see where they stop being enough
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed Module 02 (Memory) OR understand memory-capable LangGraph agents
# MAGIC - Vector Search index available
# MAGIC
# MAGIC ## Estimated Time
# MAGIC 15 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies
# MAGIC
# MAGIC Same core stack from earlier modules. If evaluation results look wrong later,
# MAGIC version mismatches here are one of the first things to check.

# COMMAND ----------

# MAGIC %pip install -q --upgrade \
# MAGIC   "databricks-sdk>=0.101,<0.103" \
# MAGIC   "mlflow[databricks]>=3.10,<3.11" \
# MAGIC   "databricks-agents" \
# MAGIC   "databricks-langchain[memory]>=0.17,<0.18" \
# MAGIC   "databricks-vectorsearch>=0.66,<0.67" \
# MAGIC   "langgraph>=1.1,<1.2" \
# MAGIC   "langchain-core>=1.2,<2"
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configuration and Imports
# MAGIC
# MAGIC Everything is hardcoded here so the notebook stays self-contained. If you
# MAGIC need to point at a different catalog or endpoint, this is the only cell to change.

# COMMAND ----------

import json
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
from mlflow.genai.datasets import create_dataset, get_dataset
from mlflow.genai.scorers import Guidelines, RelevanceToQuery, Safety

# COMMAND ----------

# Hardcoded notebook config keeps Module 03 self-contained and easy to debug.
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"
LAKEBASE_INSTANCE_NAME = None
LAKEBASE_AUTOSCALING_PROJECT = "knowledge-assistant-state"
LAKEBASE_AUTOSCALING_BRANCH = "production"
VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.policy_index"
EVAL_DATASET_TABLE = f"{CATALOG}.{SCHEMA}.bootcamp_eval_dataset"
EXPERIMENT_NAME = (
    f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/"
    "agent_bootcamp_evaluation"
)

FIRST_EVAL_RUN_ID = uuid.uuid4().hex[:8]

# Lightweight notebook memory store so we can keep the same memory-tool shape from
# Module 02 while staying focused on evaluation.
DEMO_USER_MEMORIES: dict[str, dict[str, dict[str, str]]] = {}


def lakebase_target() -> str:
    if LAKEBASE_INSTANCE_NAME:
        return LAKEBASE_INSTANCE_NAME
    return f"{LAKEBASE_AUTOSCALING_PROJECT}/{LAKEBASE_AUTOSCALING_BRANCH}"


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


print("✓ Configuration loaded")
print(f"  Lakebase target: {lakebase_target()}")
print(f"  Experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Rebuild the Same Bot Shape from Module 02
# MAGIC
# MAGIC We keep the same production-style layout from the memory module:
# MAGIC - `search_policy_documents` for policy retrieval
# MAGIC - memory tools for recall/save/delete
# MAGIC - a prompted LangGraph agent that decides when to use each tool
# MAGIC
# MAGIC In Module 02 these memory tools are backed by DatabricksStore. Here we use a
# MAGIC small in-notebook store so the evaluation lesson stays lightweight.

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

print("✓ Memory-capable policy bot rebuilt")
print(f"  Tools: {[tool.name for tool in all_tools]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create a Managed Evaluation Dataset
# MAGIC
# MAGIC Instead of keeping evaluation data in a throwaway DataFrame, we use
# MAGIC `mlflow.genai.datasets.create_dataset()` to store it as a **Unity Catalog
# MAGIC table**. This gives us:
# MAGIC
# MAGIC - **Versioning** — every `merge_records()` call is timestamped with who changed what
# MAGIC - **Reusability** — notebook `02_evaluation.py` retrieves this same dataset with `get_dataset()`
# MAGIC - **Lineage** — the MLflow Experiment UI shows which evaluation runs used which dataset
# MAGIC
# MAGIC The dataset schema matches what `mlflow.genai.evaluate()` expects:
# MAGIC each record has `inputs` (the question) and `expectations` (the ground truth).

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

try:
    eval_dataset = get_dataset(name=EVAL_DATASET_TABLE)
    print(f"✓ Retrieved existing dataset: {EVAL_DATASET_TABLE}")
except Exception:
    eval_dataset = create_dataset(
        name=EVAL_DATASET_TABLE,
        experiment_id=[experiment.experiment_id],
    )
    print(f"✓ Created new dataset: {EVAL_DATASET_TABLE}")

first_eval_records = [
    {
        "inputs": {"question": "How much vacation time do full-time employees get?"},
        "expectations": {"expected_answer": "Full-time employees accrue 15 days of vacation per year."},
    },
    {
        "inputs": {"question": "What are the core in-office days for hybrid workers?"},
        "expectations": {"expected_answer": "Tuesday and Thursday are the core in-office days."},
    },
    {
        "inputs": {"question": "Do unused sick days carry over?"},
        "expectations": {"expected_answer": "No, unused sick leave does not carry over to the next year."},
    },
]

eval_dataset.merge_records(first_eval_records)

first_eval_data = eval_dataset.to_df()
display(first_eval_data)
print(f"  Dataset records: {len(first_eval_data)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Define `predict_fn` and Run Built-In Scorers
# MAGIC
# MAGIC Instead of generating predictions in a manual loop and then scoring separately,
# MAGIC we define a `predict_fn` and let `mlflow.genai.evaluate()` handle everything
# MAGIC in one call:
# MAGIC
# MAGIC 1. **Generate** — MLflow calls `predict_fn` once per dataset record
# MAGIC 2. **Trace** — each call is automatically traced and linked to the evaluation
# MAGIC 3. **Score** — built-in scorers grade every response
# MAGIC
# MAGIC The `predict_fn` parameter names must match the keys in `inputs` from the
# MAGIC dataset. Our records have `inputs: {"question": ...}`, so the function
# MAGIC takes `question: str`.
# MAGIC
# MAGIC We start with a small scorer set:
# MAGIC
# MAGIC - `RelevanceToQuery()` — does the answer stay focused on the question?
# MAGIC - `Guidelines()` — does it follow our tone and grounding rules?
# MAGIC - `Safety()` — baseline guardrail, even for low-risk HR questions

# COMMAND ----------

def predict_fn(question: str) -> str:
    config = {
        "configurable": {
            "thread_id": f"first-eval-{FIRST_EVAL_RUN_ID}-{hash(question) % 10000:04d}",
            "user_id": "first-eval-user",
        }
    }
    result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    return result["messages"][-1].content


relevance_scorer = RelevanceToQuery()
safety_scorer = Safety()
policy_response_guidelines = Guidelines(
    name="policy_response_guidelines",
    guidelines=[
        "The response must use a professional, helpful tone.",
        "The response must not include slang, sarcasm, or emojis.",
        "The response must answer the user's policy question directly.",
        "The response must stay grounded in company policy information.",
    ],
)

with mlflow.start_run(run_name="first_eval_loop"):
    mlflow.log_params(
        {
            "catalog": CATALOG,
            "schema": SCHEMA,
            "vector_index": VECTOR_INDEX,
            "eval_dataset": EVAL_DATASET_TABLE,
            "num_examples": len(first_eval_data),
        }
    )

    first_eval_results = mlflow.genai.evaluate(
        data=first_eval_data[["inputs", "expectations"]],
        predict_fn=predict_fn,
        scorers=[relevance_scorer, policy_response_guidelines, safety_scorer],
    )

print("✓ MLflow evaluation complete")
if hasattr(first_eval_results, "metrics"):
    print("\nAggregate metrics:")
    for metric_name, value in first_eval_results.metrics.items():
        print(f"  {metric_name}: {value}")

results_table_name = list(first_eval_results.tables.keys())[0] if first_eval_results.tables else None
if results_table_name:
    print(f"\nDetailed results table: {results_table_name}")
    display(first_eval_results.tables[results_table_name])

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to Notice
# MAGIC
# MAGIC With one `evaluate()` call we generated predictions, traced every call, and
# MAGIC scored the results. That is the core evaluation pattern:
# MAGIC
# MAGIC - **dataset** → `create_dataset()` + `merge_records()`
# MAGIC - **predict** → `predict_fn` called automatically per record
# MAGIC - **score** → built-in scorers grade each response
# MAGIC - **inspect** → results table + traces in the MLflow UI
# MAGIC
# MAGIC But built-ins are not the whole story:
# MAGIC
# MAGIC - They are great for broad checks like relevance, tone, and safety.
# MAGIC - They do **not** tell us whether the HR policy content is factually correct.
# MAGIC - They do **not** explain the internal execution of a single run.
# MAGIC
# MAGIC The next notebook solves a different problem: before we rely on scores at scale,
# MAGIC we should inspect **one** run deeply with traces.
# MAGIC
# MAGIC ## Summary
# MAGIC
# MAGIC ### What We Built
# MAGIC - ✅ Reused the same memory-capable bot shape from Module 02
# MAGIC - ✅ Created a managed evaluation dataset backed by Unity Catalog
# MAGIC - ✅ Used `predict_fn` to generate, trace, and score in one `evaluate()` call
# MAGIC - ✅ Ran basic built-in scorers
# MAGIC - ✅ Identified what built-ins can and cannot tell us
# MAGIC
# MAGIC ### Next Step
# MAGIC
# MAGIC Continue to [01_tracing.py](01_tracing.py) to inspect one run with MLflow
# MAGIC tracing before we scale evaluation to many runs.

# COMMAND ----------

print("✓ First evaluation loop complete!")
print("Next: open 01_tracing.py to inspect one run with MLflow tracing.")
