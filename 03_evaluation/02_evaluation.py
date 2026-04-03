# Databricks notebook source
# MAGIC %md
# MAGIC # Batch Evaluation: Built-In Scorers First, Custom Judges Second
# MAGIC
# MAGIC ## What You'll Build
# MAGIC Score many runs of the same memory-capable HR bot from Module 02. You will
# MAGIC start with built-in MLflow scorers, see where they help, then add custom
# MAGIC judges only for the gaps that built-ins cannot cover well.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Reuse the same production-style agent structure from the memory section
# MAGIC - Create an evaluation dataset with expected answers
# MAGIC - Use `predict_fn` to let MLflow handle prediction, tracing, and parallelism
# MAGIC - Run `mlflow.genai.evaluate()` with built-in scorers first (including `Correctness`)
# MAGIC - Write code-based scorers with the `@scorer` decorator
# MAGIC - Add custom LLM judges only for domain-specific HR policy checks
# MAGIC - Evaluate agent behavior with `ToolCallCorrectness` and `ToolCallEfficiency`
# MAGIC - Test multi-turn memory with conversation-level scorers
# MAGIC - Apply human feedback and quality gates to the final scorer set
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed [01_tracing.py](01_tracing.py)
# MAGIC - Understanding of memory-capable agents from Module 02
# MAGIC
# MAGIC ## Estimated Time
# MAGIC 45 minutes

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
# MAGIC ## Step 2: Setup and Imports

# COMMAND ----------

import json
import re
import uuid
from typing import Literal

import mlflow
import pandas as pd
from databricks_langchain import ChatDatabricks, CheckpointSaver, VectorSearchRetrieverTool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from mlflow.entities import Feedback
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import (
    Correctness,
    Guidelines,
    RelevanceToQuery,
    Safety,
    ToolCallCorrectness,
    ToolCallEfficiency,
    scorer,
)

# COMMAND ----------

# Hardcoded notebook config keeps Module 03 self-contained and easy to debug.
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"
LAKEBASE_INSTANCE_NAME = None
LAKEBASE_AUTOSCALING_PROJECT = "knowledge-assistant-state"
LAKEBASE_AUTOSCALING_BRANCH = "production"
VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.policy_index"

EVALUATION_RUN_ID = uuid.uuid4().hex[:8]
EXPERIMENT_NAME = (
    f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/"
    "agent_bootcamp_evaluation"
)

DEMO_USER_MEMORIES: dict[str, dict[str, dict[str, str]]] = {
    "manager-alex": {
        "role": {"role": "manager"},
        "department": {"department": "Engineering"},
    },
    "hybrid-casey": {
        "work_style": {"work_style": "hybrid"},
    },
}


def lakebase_target() -> str:
    if LAKEBASE_INSTANCE_NAME:
        return LAKEBASE_INSTANCE_NAME
    return f"{LAKEBASE_AUTOSCALING_PROJECT}/{LAKEBASE_AUTOSCALING_BRANCH}"


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def first_results_table(results):
    if not getattr(results, "tables", None):
        return None, None
    table_name = list(results.tables.keys())[0]
    return table_name, results.tables[table_name]


def feedback_columns(results_df: pd.DataFrame) -> list[str]:
    return [column for column in results_df.columns if "value" in column.lower()]


mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.langchain.autolog()
mlflow.tracing.enable()

print("✓ Configuration loaded")
print(f"  Experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Rebuild the Same Bot Shape from Module 02
# MAGIC
# MAGIC We keep the same graph shape used in the memory section:
# MAGIC one prompted LangGraph agent with document retrieval and memory tools.
# MAGIC
# MAGIC In Module 02 those memory tools can be backed by DatabricksStore. Here we keep
# MAGIC the same tool interface but use a small notebook memory map so the focus stays
# MAGIC on evaluation rather than storage setup.

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

print("✓ Memory-capable bot rebuilt for evaluation")
print(f"  Tools: {[tool.name for tool in all_tools]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create the Evaluation Dataset
# MAGIC
# MAGIC We will evaluate several policy questions at once. Each row contains the input
# MAGIC plus a reference answer we can use later when custom judges become necessary.
# MAGIC
# MAGIC We include both `expected_answer` (a plain string for custom judges) and
# MAGIC `expected_facts` (a list for the built-in `Correctness` scorer). Keeping both
# MAGIC formats lets us use built-in and custom scorers on the same dataset.
# MAGIC
# MAGIC > **Production note**: for versioned, reusable evaluation datasets, MLflow 3.x
# MAGIC > provides `mlflow.genai.datasets.create_dataset()` and `get_dataset()`. Inline
# MAGIC > DataFrames are fine for teaching, but the Datasets API is worth knowing about
# MAGIC > when your eval set grows.

# COMMAND ----------

eval_data = pd.DataFrame(
    [
        {
            "user_id": "manager-alex",
            "question": "How much vacation time do full-time employees get?",
            "expected_answer": "Full-time employees accrue 15 days of vacation per year (1.25 days per month).",
            "expected_facts": ["Full-time employees accrue 15 days of vacation per year."],
        },
        {
            "user_id": "hybrid-casey",
            "question": "What are the core in-office days for hybrid workers?",
            "expected_answer": "Tuesday and Thursday are designated as the core in-office days.",
            "expected_facts": ["Tuesday and Thursday are core in-office days."],
        },
        {
            "user_id": "manager-alex",
            "question": "What's the annual learning budget for managers?",
            "expected_answer": "Managers receive $2,500 per year for professional development.",
            "expected_facts": ["Managers receive $2,500 per year for professional development."],
        },
        {
            "user_id": "hybrid-casey",
            "question": "Can employees work remotely full-time?",
            "expected_answer": "No. Hybrid employees must be in the office at least 2 days per week, with Tuesday and Thursday as core days.",
            "expected_facts": ["Hybrid employees must be in the office at least 2 days per week.", "Tuesday and Thursday are core days."],
        },
        {
            "user_id": "manager-alex",
            "question": "Do unused sick days carry over?",
            "expected_answer": "No, unused sick leave does not carry over to the next year.",
            "expected_facts": ["Unused sick leave does not carry over to the next year."],
        },
    ]
)

display(eval_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Define `predict_fn` and Format Evaluation Records
# MAGIC
# MAGIC Instead of generating predictions in a manual loop, we define a `predict_fn`
# MAGIC that MLflow calls during `mlflow.genai.evaluate()`. This is the idiomatic
# MAGIC MLflow 3.x pattern because:
# MAGIC
# MAGIC - traces are automatically linked to evaluation results
# MAGIC - MLflow handles parallelism (configurable via `MLFLOW_GENAI_EVAL_MAX_WORKERS`)
# MAGIC - you get a single call that generates, traces, and scores
# MAGIC
# MAGIC The `predict_fn` parameter names must match the keys in `inputs`.

# COMMAND ----------

def predict_fn(question: str, user_id: str) -> str:
    config = {
        "configurable": {
            "thread_id": f"eval-{EVALUATION_RUN_ID}-{hash(question) % 10000:04d}",
            "user_id": user_id,
        }
    }
    result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    return result["messages"][-1].content


eval_records = [
    {
        "inputs": {"question": row["question"], "user_id": row["user_id"]},
        "expectations": {
            "expected_answer": row["expected_answer"],
            "expected_facts": row["expected_facts"],
        },
    }
    for _, row in eval_data.iterrows()
]

print("✓ predict_fn defined and eval records formatted")
print(f"  Total eval records: {len(eval_records)}")
print(f"  Sample inputs: {eval_records[0]['inputs']}")
print(f"  Sample expectations keys: {list(eval_records[0]['expectations'].keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Use a More Advanced Built-In Scorer Set
# MAGIC
# MAGIC In `00_first_eval_loop.py` we kept the starter scorer set intentionally small.
# MAGIC Here we make it more realistic for a memory-enabled HR policy bot by splitting
# MAGIC broad quality into several dimensions:
# MAGIC
# MAGIC - question relevance
# MAGIC - **correctness** against expected facts (new in MLflow 3.x)
# MAGIC - professional tone
# MAGIC - policy grounding
# MAGIC - source-aware behavior
# MAGIC - relevant personalization
# MAGIC - response structure
# MAGIC - safety
# MAGIC
# MAGIC `Correctness()` is the bridge between broad quality checks and custom judges.
# MAGIC It compares the agent's output against `expected_facts` in your dataset, so it
# MAGIC catches factual mismatches without needing a custom judge.

# COMMAND ----------

relevance_scorer = RelevanceToQuery()
correctness_scorer = Correctness()
safety_scorer = Safety()

professional_tone_scorer = Guidelines(
    name="professional_tone",
    guidelines=[
        "The response must use a professional, helpful tone.",
        "The response must not include slang, sarcasm, or emojis.",
    ],
)

policy_grounding_scorer = Guidelines(
    name="policy_grounding",
    guidelines=[
        "The response should stay focused on company policy information.",
        "The response must not invent policy numbers, benefits, eligibility conditions, or approval requirements.",
    ],
)

source_awareness_scorer = Guidelines(
    name="source_awareness",
    guidelines=[
        "When retrieved policy context includes source file names, the response should reference the relevant source file names.",
        "If the policy context is incomplete or uncertain, the response should avoid sounding overly certain.",
    ],
)

relevant_personalization_scorer = Guidelines(
    name="relevant_personalization",
    guidelines=[
        "Use remembered user details only when they materially help answer the current question.",
        "Do not introduce unrelated personal details or personalize the answer when memory is not relevant.",
    ],
)

response_structure_scorer = Guidelines(
    name="response_structure",
    guidelines=[
        "The response should answer in 2 short bullet points.",
        "The response should end with one practical next step.",
    ],
)

print("✓ Built-in scorers created")
print("  • RelevanceToQuery")
print("  • Correctness (compares output against expected_facts)")
print("  • Safety")
print("  • Guidelines(name='professional_tone')")
print("  • Guidelines(name='policy_grounding')")
print("  • Guidelines(name='source_awareness')")
print("  • Guidelines(name='relevant_personalization')")
print("  • Guidelines(name='response_structure')")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Run Built-In Evaluation First
# MAGIC
# MAGIC This is where the `predict_fn` pattern pays off: one call generates predictions,
# MAGIC creates traces, and scores everything. Each prediction automatically gets a linked
# MAGIC trace you can inspect later in the MLflow UI.
# MAGIC
# MAGIC This answers the question: "Are the responses relevant, correct against known
# MAGIC facts, safe, well-grounded, well-structured, and appropriately personalized?"

# COMMAND ----------

built_in_results = mlflow.genai.evaluate(
    data=eval_records,
    predict_fn=predict_fn,
    scorers=[
        relevance_scorer,
        correctness_scorer,
        safety_scorer,
        professional_tone_scorer,
        policy_grounding_scorer,
        source_awareness_scorer,
        relevant_personalization_scorer,
        response_structure_scorer,
    ],
)

print("✓ Built-in evaluation complete")
print("\nAggregate metrics:")
for metric_name, value in built_in_results.metrics.items():
    print(f"  {metric_name}: {value}")

built_in_table_name, built_in_df = first_results_table(built_in_results)
if built_in_df is not None:
    print(f"\nDetailed built-in results table: {built_in_table_name}")
    display(built_in_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Notice Where Built-Ins Stop Being Enough
# MAGIC
# MAGIC Built-ins tell us whether answers are safe and well-behaved. They usually do
# MAGIC **not** tell us whether:
# MAGIC - the answer used the correct HR policy number
# MAGIC - the answer included an important eligibility condition
# MAGIC - the answer omitted a key detail from the reference answer
# MAGIC
# MAGIC There is one more nuance to watch for in real projects: if your policy corpus
# MAGIC changes, a mismatch might mean either:
# MAGIC - the answer is factually weak, or
# MAGIC - the reference answer is stale and your eval dataset needs updating
# MAGIC
# MAGIC Built-ins usually do not separate those two cases. That is the point where
# MAGIC custom judges and periodic dataset review become useful.
# MAGIC
# MAGIC ### Code-based scorers: not all scoring needs an LLM
# MAGIC
# MAGIC The `@scorer` decorator lets you write deterministic Python checks that plug
# MAGIC directly into `mlflow.genai.evaluate()`. The `policy_detail_check` scorer below
# MAGIC is a lightweight example: it checks whether the expected answer appears in the
# MAGIC agent's output. No LLM call required, instant feedback.

# COMMAND ----------

@scorer
def policy_detail_check(outputs: str, expectations: dict) -> Feedback:
    """Deterministic check: does the output contain the expected policy detail?"""
    expected_text = normalize_text(expectations.get("expected_answer", ""))
    answer_text = normalize_text(str(outputs))
    hedging_signals = [
        "best to verify", "check with hr", "ask your hr team", "ask hr",
        "not explicitly covered", "depends on",
    ]

    if expected_text and expected_text in answer_text:
        return Feedback(value="yes", rationale="Expected policy detail appears in the answer.")

    if any(signal in answer_text for signal in hedging_signals):
        return Feedback(
            value="no",
            rationale=(
                "The answer hedges or defers instead of clearly answering. "
                "A custom LLM judge would catch this better than broad built-ins."
            ),
        )

    return Feedback(
        value="no",
        rationale=(
            "The answer may score well on relevance and tone, but it does not "
            "contain the exact expected policy detail."
        ),
    )


if built_in_df is not None:
    comparison_rows = []
    for idx in range(len(built_in_df)):
        result_row = built_in_df.iloc[idx]
        agent_answer = str(result_row.get("outputs", ""))
        comparison_row = {
            "question": eval_data.iloc[idx]["question"],
            "expected_answer": eval_data.iloc[idx]["expected_answer"],
            "agent_answer": agent_answer[:200],
        }
        for column in [
            "relevance_to_query/value",
            "correctness/value",
            "professional_tone/value",
            "policy_grounding/value",
            "response_structure/value",
        ]:
            if column in built_in_df.columns:
                comparison_row[column] = result_row[column]
        comparison_rows.append(comparison_row)

    comparison_df = pd.DataFrame(comparison_rows)
    display(comparison_df)
    print("\nNotice: rows can look good on broad checks but still miss exact policy details.")
    print("The @scorer-based policy_detail_check and Correctness scorer help catch these.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Add Custom Judges Only for the Domain Gaps
# MAGIC
# MAGIC Now we add custom judges for the things that are specific to this HR bot:
# MAGIC factual accuracy and policy completeness.
# MAGIC
# MAGIC `make_judge()` supports these template variables:
# MAGIC - `{{ inputs }}` - the question/user_id sent to the agent
# MAGIC - `{{ outputs }}` - the agent's response
# MAGIC - `{{ expectations }}` - ground truth
# MAGIC - `{{ trace }}` - the full execution trace (requires setting `model=`)
# MAGIC
# MAGIC We use `{{ outputs }}` and `{{ expectations }}` here. When you need to inspect
# MAGIC tool calls or intermediate steps inside the trace, `{{ trace }}` lets the judge
# MAGIC LLM explore the full span tree autonomously.

# COMMAND ----------

hr_accuracy_judge = make_judge(
    name="hr_accuracy",
    instructions=(
        "Evaluate whether {{ outputs }} is factually accurate with respect to the HR policy answer in {{ expectations }}.\n"
        "Focus on numbers, eligibility conditions, and policy details.\n"
        "Return one of: excellent, good, fair, poor, very_poor."
    ),
    feedback_value_type=Literal["excellent", "good", "fair", "poor", "very_poor"],
    model=f"databricks:/{LLM_ENDPOINT}",
)

policy_completeness_judge = make_judge(
    name="policy_completeness",
    instructions=(
        "Evaluate whether {{ outputs }} covers the key information in {{ expectations }}.\n"
        "Reward answers that include important caveats, required days, rates, or next steps.\n"
        "Return one of: excellent, good, fair, poor, very_poor."
    ),
    feedback_value_type=Literal["excellent", "good", "fair", "poor", "very_poor"],
    model=f"databricks:/{LLM_ENDPOINT}",
)

print("✓ Custom judges created")
print("  • hr_accuracy")
print("  • policy_completeness")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Re-Run Evaluation With the Full Scorer Set
# MAGIC
# MAGIC This pass keeps the built-ins, adds the custom LLM judges, and includes the
# MAGIC code-based `policy_detail_check` scorer from Step 8. All three scorer types
# MAGIC (built-in, custom LLM, code-based) work together in a single `evaluate()` call.

# COMMAND ----------

full_results = mlflow.genai.evaluate(
    data=eval_records,
    predict_fn=predict_fn,
    scorers=[
        relevance_scorer,
        correctness_scorer,
        safety_scorer,
        professional_tone_scorer,
        policy_grounding_scorer,
        source_awareness_scorer,
        relevant_personalization_scorer,
        response_structure_scorer,
        policy_detail_check,
        hr_accuracy_judge,
        policy_completeness_judge,
    ],
)

print("✓ Full evaluation complete")
print("\nAggregate metrics:")
for metric_name, value in full_results.metrics.items():
    print(f"  {metric_name}: {value}")

table_name, results_df = first_results_table(full_results)
if results_df is not None:
    print(f"\nDetailed full results table: {table_name}")
    display(results_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Interpret the Final Results
# MAGIC
# MAGIC Look first at the custom HR judges, because those tell you whether the answer
# MAGIC was actually right for your domain.

# COMMAND ----------

if results_df is not None:
    print("Feedback columns:")
    print(feedback_columns(results_df))

    print("\nSample rows with feedback:")
    for _, row in results_df.head(2).iterrows():
        print("=" * 80)
        if "inputs" in row:
            print(f"Inputs: {row['inputs']}")
        if "outputs" in row:
            print(f"Output: {str(row['outputs'])[:220]}...")

        for column in feedback_columns(results_df):
            scorer_name = column.split("/")[0]
            print(f"  {scorer_name}: {row[column]}")
            rationale_column = f"{scorer_name}/rationale"
            if rationale_column in results_df.columns and pd.notna(row[rationale_column]):
                print(f"    Rationale: {str(row[rationale_column])[:160]}...")

    print("\nFeedback distribution:")
    for column in feedback_columns(results_df):
        print(f"\n{column}")
        print(results_df[column].value_counts(dropna=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 12: Evaluate Agent Behavior With Tool-Call Scorers
# MAGIC
# MAGIC So far we have scored **output quality**. But for a tool-calling agent, we also
# MAGIC want to score **agent behavior**: did it call `search_policy_documents` before
# MAGIC answering a policy question? Did it avoid redundant tool calls?
# MAGIC
# MAGIC MLflow provides two built-in scorers that analyze traces to answer these:
# MAGIC
# MAGIC - `ToolCallCorrectness` — did the agent call the right tools with correct arguments?
# MAGIC - `ToolCallEfficiency` — were tool calls efficient without redundancy?
# MAGIC
# MAGIC These scorers require traces, which are automatically generated by `predict_fn`.
# MAGIC This ties back to the tracing concepts from notebook `01_tracing.py`.

# COMMAND ----------

agent_behavior_results = mlflow.genai.evaluate(
    data=eval_records,
    predict_fn=predict_fn,
    scorers=[
        ToolCallCorrectness(),
        ToolCallEfficiency(),
    ],
)

print("✓ Agent behavior evaluation complete")
print("\nAggregate metrics:")
for metric_name, value in agent_behavior_results.metrics.items():
    print(f"  {metric_name}: {value}")

agent_table_name, agent_df = first_results_table(agent_behavior_results)
if agent_df is not None:
    display(agent_df)
    print("\nLook for rows where the agent skipped search_policy_documents or made")
    print("redundant tool calls. These are behavioral issues that output-only scorers miss.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 13: Test Multi-Turn Memory With Conversation-Level Evaluation
# MAGIC
# MAGIC You built memory in Module 02. But does memory actually help across conversation
# MAGIC turns? Single-turn evaluation cannot answer that.
# MAGIC
# MAGIC Here we run a short multi-turn conversation, tag it with a session ID, and
# MAGIC evaluate the full session with conversation-level scorers:
# MAGIC
# MAGIC - `ConversationCompleteness` — were all user questions answered by the end?
# MAGIC - `KnowledgeRetention` — did the agent remember facts from earlier turns?
# MAGIC
# MAGIC These scorers are experimental in MLflow 3.10 and require traces with
# MAGIC `mlflow.trace.session` metadata.

# COMMAND ----------

session_id = f"multi-turn-eval-{EVALUATION_RUN_ID}"

multi_turn_questions = [
    ("I'm a manager in the Engineering department.", "manager-alex"),
    ("What's my annual learning budget?", "manager-alex"),
    ("Does unused sick leave carry over to next year?", "manager-alex"),
]

for turn_idx, (question, user_id) in enumerate(multi_turn_questions):
    with mlflow.start_span(name=f"multi_turn_{turn_idx}") as span:
        mlflow.update_current_trace(
            metadata={"mlflow.trace.session": session_id}
        )
        config = {
            "configurable": {
                "thread_id": f"multi-turn-{EVALUATION_RUN_ID}",
                "user_id": user_id,
            }
        }
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]}, config=config
        )
        answer = result["messages"][-1].content
        span.set_inputs({"question": question})
        span.set_outputs(answer)
        print(f"Turn {turn_idx + 1}: {question}")
        print(f"  → {answer[:120]}...\n")

print(f"✓ Multi-turn conversation complete (session: {session_id})")
print("  The second turn should reflect knowledge from the first (manager role).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 14: Conversation Simulation (Brief)
# MAGIC
# MAGIC You do not have to hand-write every multi-turn test case. MLflow 3.10 introduces
# MAGIC `ConversationSimulator`, which generates reproducible conversations from a
# MAGIC persona and goal. This is especially useful for regression testing when you
# MAGIC change prompts or tools.
# MAGIC
# MAGIC The example below shows the pattern. Adapt `predict_fn` to accept the Chat
# MAGIC Completions message format (`list[dict]`) if you want to run the simulator
# MAGIC end-to-end.
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.simulators import ConversationSimulator
# MAGIC
# MAGIC simulator = ConversationSimulator(
# MAGIC     test_cases=[
# MAGIC         {
# MAGIC             "goal": "Find out vacation and sick leave policies as a new employee.",
# MAGIC             "persona": "A recently hired engineer who has never read the employee handbook.",
# MAGIC         },
# MAGIC         {
# MAGIC             "goal": "Understand the remote work policy and in-office requirements.",
# MAGIC             "persona": "A hybrid worker who wants to maximize remote days.",
# MAGIC         },
# MAGIC     ],
# MAGIC     max_turns=4,
# MAGIC )
# MAGIC
# MAGIC from mlflow.genai.scorers import ConversationCompleteness, Safety
# MAGIC
# MAGIC results = mlflow.genai.evaluate(
# MAGIC     data=simulator,
# MAGIC     predict_fn=chat_predict_fn,  # must accept list[dict] messages
# MAGIC     scorers=[ConversationCompleteness(), Safety()],
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **When to use this**: before deploying prompt or tool changes, generate synthetic
# MAGIC conversations to check for regressions without manual testing.

# COMMAND ----------

print("Step 14 is a reference pattern — no code to run here.")
print("Use ConversationSimulator when you want to generate reproducible")
print("multi-turn test cases from personas and goals.")
print("\nKey scorers for conversation evaluation:")
print("  • ConversationCompleteness — were all user questions answered?")
print("  • KnowledgeRetention — did the agent remember facts from earlier turns?")
print("  • UserFrustration — did the user become frustrated?")
print("  • ConversationalGuidelines — custom natural-language rules for conversations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 15: Add Human Feedback to Improve the Judge Set
# MAGIC
# MAGIC Human feedback helps you refine custom judges when they disagree with what your
# MAGIC domain experts care about.
# MAGIC
# MAGIC > **Production next step**: MLflow 3.10+ supports `judge.align()` with
# MAGIC > SIMBA and MemAlign optimizers to automatically tune judge instructions based
# MAGIC > on human feedback. The **Judge Builder UI** (MLflow >= 3.9) provides a no-code
# MAGIC > alternative for creating and testing judges visually. Both are beyond bootcamp
# MAGIC > scope but worth knowing about for production use.

# COMMAND ----------

human_feedback_example = pd.DataFrame(
    [
        {
            "question": "How much vacation time do employees get?",
            "judge_name": "policy_completeness",
            "judge_rating": "good",
            "human_rating": "fair",
            "human_note": "The answer should include the accrual rate of 1.25 days per month.",
        },
        {
            "question": "What are the core in-office days?",
            "judge_name": "hr_accuracy",
            "judge_rating": "excellent",
            "human_rating": "excellent",
            "human_note": "The answer is accurate and complete.",
        },
        {
            "question": "Can employees work remotely full-time?",
            "judge_name": "hr_accuracy",
            "judge_rating": "fair",
            "human_rating": "poor",
            "human_note": "The answer should explicitly mention the 2-day in-office requirement.",
        },
    ]
)

display(human_feedback_example)

rating_order = ["very_poor", "poor", "fair", "good", "excellent"]
print("Judge vs human alignment:")
for _, row in human_feedback_example.iterrows():
    judge_idx = rating_order.index(row["judge_rating"])
    human_idx = rating_order.index(row["human_rating"])
    diff = abs(judge_idx - human_idx)
    aligned = "✓" if diff <= 1 else "✗"
    print(
        f"{aligned} {row['judge_name']} on '{row['question']}' -> "
        f"judge={row['judge_rating']}, human={row['human_rating']}"
    )
    if diff > 1:
        print(f"   Note: {row['human_note']}")

print("\nImprovement loop:")
print("  1. Collect 10-20 expert ratings")
print("  2. Compare human labels with judge outputs")
print("  3. Tighten judge instructions around disagreement patterns")
print("  4. Re-run evaluation and confirm alignment improves")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 16: Turn the Final Scorer Set Into Quality Gates
# MAGIC
# MAGIC Once you know which scorers matter, turn them into simple deployment checks.

# COMMAND ----------

def check_quality_gates(results_df: pd.DataFrame) -> bool:
    print("Quality Gate Checks")
    print("=" * 80)

    passed = True

    if "hr_accuracy/value" in results_df.columns:
        failing_accuracy = results_df[
            results_df["hr_accuracy/value"].astype(str).str.lower().isin(["poor", "very_poor"])
        ]
        if not failing_accuracy.empty:
            passed = False
            print(f"❌ hr_accuracy gate failed on {len(failing_accuracy)} rows")

    if "policy_completeness/value" in results_df.columns:
        failing_completeness = results_df[
            results_df["policy_completeness/value"].astype(str).str.lower().isin(["poor", "very_poor"])
        ]
        if not failing_completeness.empty:
            passed = False
            print(f"❌ policy_completeness gate failed on {len(failing_completeness)} rows")

    tone_columns = [column for column in results_df.columns if column.startswith("professional_tone/") and "value" in column]
    for column in tone_columns:
        failing_tone = results_df[results_df[column].astype(str).str.lower().str.contains("no|fail", regex=True)]
        if not failing_tone.empty:
            passed = False
            print(f"❌ professional_tone gate failed on {len(failing_tone)} rows")

    if passed:
        print("✅ All quality gates passed")
    else:
        print("❌ Quality gates failed")

    return passed


gates_passed = check_quality_gates(results_df) if results_df is not None else False

print("\nCI/CD pattern:")
print("```python")
print("results = mlflow.genai.evaluate(data=eval_records, scorers=final_scorers)")
print("table_name, results_df = first_results_table(results)")
print("if not check_quality_gates(results_df):")
print("    raise Exception('Quality gates failed - block deployment')")
print("```")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What We Built
# MAGIC - ✅ Reused the same memory-capable bot shape from Module 02
# MAGIC - ✅ Used `predict_fn` to let MLflow handle prediction, tracing, and parallelism
# MAGIC - ✅ Started with built-in scorers including `Correctness` for fact-checking
# MAGIC - ✅ Wrote code-based scorers with the `@scorer` decorator
# MAGIC - ✅ Added custom LLM judges only for HR-specific accuracy and completeness
# MAGIC - ✅ Evaluated agent behavior with `ToolCallCorrectness` and `ToolCallEfficiency`
# MAGIC - ✅ Tested multi-turn memory with conversation-level evaluation
# MAGIC - ✅ Connected human feedback to judge refinement
# MAGIC - ✅ Turned the final scorer set into quality gates
# MAGIC
# MAGIC ### The Evaluation Flow to Remember
# MAGIC
# MAGIC 1. Start with a toy Q&A loop so the pattern feels concrete
# MAGIC 2. Use traces to understand one run deeply
# MAGIC 3. Use built-in scorers (including `Correctness`) to measure many runs quickly
# MAGIC 4. Write code-based `@scorer` functions for deterministic checks
# MAGIC 5. Add custom LLM judges only for the domain-specific gaps
# MAGIC 6. Evaluate agent behavior (tool calls) and conversation quality (memory)
# MAGIC
# MAGIC ### Three Types of Scorers
# MAGIC
# MAGIC | Type | Example | When to Use |
# MAGIC | --- | --- | --- |
# MAGIC | Built-in LLM | `Correctness()`, `Safety()`, `Guidelines()` | Quick start, broad quality |
# MAGIC | Code-based `@scorer` | `policy_detail_check` | Deterministic checks, no LLM cost |
# MAGIC | Custom LLM `make_judge()` | `hr_accuracy`, `policy_completeness` | Domain-specific nuance |
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC Continue to **Module 04: MCP Tool Integration** to expand this bot with more
# MAGIC tools such as Genie and UC Functions.

# COMMAND ----------

print("✓ Batch evaluation notebook complete!")
print("Your evaluation stack now includes:")
print("  ✓ predict_fn pattern for integrated generation + scoring")
print("  ✓ Built-in scorers including Correctness for fact-checking")
print("  ✓ Code-based @scorer for deterministic checks")
print("  ✓ Custom LLM judges for HR-specific gaps")
print("  ✓ ToolCallCorrectness + ToolCallEfficiency for agent behavior")
print("  ✓ Multi-turn conversation evaluation for memory testing")
print("  ✓ Human feedback and quality gates")
print("\nNext: continue to Module 04 for multi-tool agents.")
