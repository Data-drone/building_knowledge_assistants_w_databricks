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
# MAGIC - Create a small evaluation dataset in MLflow's `inputs` / `outputs` / `expectations` shape
# MAGIC - Generate answers in a simple loop
# MAGIC - Run a couple of built-in scorers with `mlflow.genai.evaluate()`
# MAGIC - See why built-ins are useful, and where they stop being enough
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

# COMMAND ----------

import json
import re
import uuid

import mlflow
import pandas as pd
from databricks_langchain import ChatDatabricks, CheckpointSaver, VectorSearchRetrieverTool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
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
# MAGIC ## Step 4: Create a Small MLflow Evaluation Dataset
# MAGIC
# MAGIC We keep the first loop small, but we already use the same row shape that
# MAGIC `mlflow.genai.evaluate()` expects later:
# MAGIC
# MAGIC - `inputs`
# MAGIC - `outputs`
# MAGIC - `expectations`
# MAGIC
# MAGIC At this stage only `question` and `expected_answer` are known. We will generate
# MAGIC `outputs` in the next step.

# COMMAND ----------

first_eval_data = pd.DataFrame(
    [
        {
            "question": "How much vacation time do full-time employees get?",
            "expected_answer": "Full-time employees accrue 15 days of vacation per year.",
        },
        {
            "question": "What are the core in-office days for hybrid workers?",
            "expected_answer": "Tuesday and Thursday are the core in-office days.",
        },
        {
            "question": "Do unused sick days carry over?",
            "expected_answer": "No, unused sick leave does not carry over to the next year.",
        },
    ]
)

display(first_eval_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Generate Answers and Format MLflow Eval Rows
# MAGIC
# MAGIC We will:
# MAGIC 1. ask the bot each question
# MAGIC 2. save the generated answer
# MAGIC 3. format each example as an MLflow evaluation row
# MAGIC
# MAGIC This keeps the notebook lightweight while still introducing the structure we
# MAGIC will reuse throughout the rest of the module.

# COMMAND ----------

mlflow_eval_rows = []
first_results = []
for idx, row in first_eval_data.iterrows():
    config = {
        "configurable": {
            "thread_id": f"first-eval-{FIRST_EVAL_RUN_ID}-{idx:03d}",
            "user_id": "first-eval-user",
        }
    }
    agent_result = agent.invoke({"messages": [HumanMessage(content=row["question"])]}, config=config)
    answer = agent_result["messages"][-1].content

    mlflow_eval_rows.append(
        {
            "inputs": {"question": row["question"]},
            "outputs": answer,
            "expectations": {"expected_answer": row["expected_answer"]},
        }
    )

    first_results.append(
        {
            "question": row["question"],
            "expected_answer": row["expected_answer"],
            "agent_answer": answer,
        }
    )

first_results_df = pd.DataFrame(first_results)
display(pd.DataFrame(mlflow_eval_rows))

print("✓ Generated answers and formatted MLflow evaluation rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Run a Few Basic Built-In Scorers
# MAGIC
# MAGIC These three built-ins are a good default starter set for direct policy Q&A:
# MAGIC
# MAGIC - `RelevanceToQuery()` checks whether the answer stays focused on the user question
# MAGIC - `Guidelines()` checks a few basic response rules we care about early
# MAGIC - `Safety()` gives us a baseline guardrail, even for low-risk HR questions
# MAGIC
# MAGIC Built-in scorers are the easiest way to get signal quickly:
# MAGIC
# MAGIC - `RelevanceToQuery()` checks for question-answer fit
# MAGIC - `Guidelines()` lets us express simple natural-language requirements
# MAGIC - `Safety()` checks for unsafe output
# MAGIC
# MAGIC We will keep the scorer set intentionally small here. Later, once we inspect a
# MAGIC single run with traces, we will scale this up.

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_NAME)

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
            "num_examples": len(first_eval_data),
        }
    )
    mlflow.log_table(first_eval_data[["question", "expected_answer"]], "first_eval_dataset.json")
    mlflow.log_table(pd.DataFrame(mlflow_eval_rows), "first_eval_rows.json")

    first_eval_results = mlflow.genai.evaluate(
        data=mlflow_eval_rows,
        scorers=[relevance_scorer, policy_response_guidelines, safety_scorer],
    )

    mlflow.log_metric("num_examples", len(first_eval_data))

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
# MAGIC ## Step 7: Why This First Loop Helps, and Why It Stops Helping
# MAGIC
# MAGIC This first loop is useful because it already teaches the core evaluation pattern
# MAGIC in an MLflow-native way:
# MAGIC
# MAGIC - create a dataset
# MAGIC - generate answers
# MAGIC - score each answer
# MAGIC - inspect the result table
# MAGIC
# MAGIC But built-ins are not the whole story:
# MAGIC
# MAGIC - they are great for broad checks like relevance, tone, and safety
# MAGIC - they do not fully tell us whether the HR policy content is factually correct
# MAGIC - they do not explain the internal execution of a single run
# MAGIC
# MAGIC The next notebook solves a different problem: before we rely on scores at scale,
# MAGIC we should inspect **one** run deeply with traces.

# COMMAND ----------

print("Full first-loop results:")
display(first_results_df)

print("What to notice:")
print("  1. The eval rows already match MLflow's expected structure")
print("  2. Built-in scorers give quick signal with very little setup")
print("  3. We still do not know why the model behaved the way it did")
print("  4. We still have not checked domain-specific HR correctness")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What We Built
# MAGIC - ✅ Reused the same memory-capable bot shape from Module 02
# MAGIC - ✅ Created a small evaluation dataset
# MAGIC - ✅ Formatted rows for `mlflow.genai.evaluate()`
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
