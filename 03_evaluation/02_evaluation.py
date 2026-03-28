# Databricks notebook source
# MAGIC %md
# MAGIC # Agent Evaluation: Systematic Quality Assessment
# MAGIC
# MAGIC ## What You'll Build
# MAGIC Add systematic evaluation to your agent with custom judges for domain-specific quality assessment.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Create evaluation datasets with ground truth
# MAGIC - Build custom judges with make_judge() (MLflow 3.x)
# MAGIC - Run comprehensive evaluation with mlflow.genai.evaluate()
# MAGIC - Interpret scores and identify issues
# MAGIC - Implement human feedback loops
# MAGIC - Set up production quality gates
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed [01_tracing_and_evaluation.py](01_tracing_and_evaluation.py) - learned tracing
# MAGIC - Understanding of memory-enabled agents from Module 02
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

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Setup and Imports

# COMMAND ----------

import mlflow
import uuid
from mlflow.genai.judges import make_judge
from typing import Literal
from databricks_langchain import ChatDatabricks, CheckpointSaver
from databricks.vector_search.client import VectorSearchClient
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
import pandas as pd

# COMMAND ----------

# Configuration
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"
LAKEBASE_INSTANCE_NAME = None
LAKEBASE_AUTOSCALING_PROJECT = "knowledge-assistant-state"
LAKEBASE_AUTOSCALING_BRANCH = "production"

VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.policy_index"
ENDPOINT_NAME = "one-env-shared-endpoint-15"

# Set experiment
experiment_name = f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/agent_bootcamp_evaluation"
mlflow.set_experiment(experiment_name)
mlflow.tracing.enable()


def lakebase_target() -> str:
    if LAKEBASE_INSTANCE_NAME:
        return LAKEBASE_INSTANCE_NAME
    return f"{LAKEBASE_AUTOSCALING_PROJECT}/{LAKEBASE_AUTOSCALING_BRANCH}"

print(f"✓ Configuration loaded")
print(f"  Experiment: {experiment_name}")

EVALUATION_RUN_ID = uuid.uuid4().hex[:8]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Build Agent (Quick Recap)

# COMMAND ----------

# Create Vector Search tool
@tool
def search_policy_documents(query: str) -> str:
    """Search company policy documents."""
    vsc = VectorSearchClient(disable_notice=True)
    index = vsc.get_index(ENDPOINT_NAME, VECTOR_INDEX)
    results = index.similarity_search(query_text=query, columns=["source_file", "chunk_text"], num_results=3)
    formatted = []
    for row in results.get("result", {}).get("data_array", []):
        formatted.append(f"[Source: {row[0]}]\n{row[1]}\n")
    return "\n".join(formatted) if formatted else "No relevant documents found."

# Build agent
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1)
tools = [search_policy_documents]
llm_with_tools = llm.bind_tools(tools)

def call_agent(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def should_continue(state: MessagesState):
    last_msg = state["messages"][-1]
    return "tools" if hasattr(last_msg, "tool_calls") and last_msg.tool_calls else END

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_agent)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

# Add CheckpointSaver
checkpointer = CheckpointSaver(
    instance_name=LAKEBASE_INSTANCE_NAME,
    project=LAKEBASE_AUTOSCALING_PROJECT,
    branch=LAKEBASE_AUTOSCALING_BRANCH,
)
try:
    checkpointer.setup()
except Exception as e:
    if "already exists" not in str(e).lower():
        raise Exception(
            f"CheckpointSaver setup failed for {lakebase_target()}. "
            "Verify the Lakebase target exists and your role has CREATE access on schema public."
        ) from e

agent = workflow.compile(checkpointer=checkpointer)

print("✓ Agent rebuilt with memory")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create Evaluation Dataset
# MAGIC
# MAGIC Build a dataset with diverse questions covering different policy areas.

# COMMAND ----------

eval_data = pd.DataFrame([
    {
        "question": "How much vacation time do full-time employees get?",
        "expected_answer": "Full-time employees accrue 15 days of vacation per year (1.25 days per month)"
    },
    {
        "question": "What are the core in-office days for hybrid workers?",
        "expected_answer": "Tuesday and Thursday are designated as core in-office days"
    },
    {
        "question": "What's the annual learning budget for managers?",
        "expected_answer": "Managers receive $2,500 per year for professional development"
    },
    {
        "question": "Can employees work remotely full-time?",
        "expected_answer": "No, hybrid employees must be in office at least 2 days per week (Tuesday and Thursday)"
    },
    {
        "question": "Do unused sick days carry over?",
        "expected_answer": "No, unused sick leave does not carry over to the next year"
    },
])

print(f"✓ Evaluation dataset created")
print(f"  Total questions: {len(eval_data)}")
print("\nSample questions:")
display(eval_data.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Create Custom Judges
# MAGIC
# MAGIC Build domain-specific judges using make_judge() from MLflow 3.x.

# COMMAND ----------

# Judge 1: HR Policy Accuracy
hr_accuracy_judge = make_judge(
    name="hr_accuracy",
    instructions=(
        "Evaluate if the response in {{ outputs }} is factually accurate according to HR policies.\n\n"
        "Compare the predicted answer to the expected answer: {{ expectations }}\n\n"
        "Check for:\n"
        "- Correct numbers and dates\n"
        "- Accurate policy details\n"
        "- Appropriate caveats and eligibility requirements\n\n"
        "Rate as:\n"
        "- 'excellent' = Perfectly accurate with all details correct\n"
        "- 'good' = Accurate with minor details missing\n"
        "- 'fair' = Mostly accurate but missing important details\n"
        "- 'poor' = Partially accurate with significant errors\n"
        "- 'very_poor' = Incorrect or misleading"
    ),
    feedback_value_type=Literal["excellent", "good", "fair", "poor", "very_poor"],
    model=f"databricks:/{LLM_ENDPOINT}",
)

# Judge 2: Response Completeness
completeness_judge = make_judge(
    name="completeness",
    instructions=(
        "Evaluate if the response in {{ outputs }} is complete and provides all necessary information.\n\n"
        "Check if important details are included:\n"
        "- Eligibility requirements\n"
        "- Specific rates or amounts\n"
        "- Important caveats or conditions\n"
        "- Next steps or how to take action\n\n"
        "Rate as:\n"
        "- 'excellent' = Complete with all necessary details\n"
        "- 'good' = Mostly complete with one minor detail missing\n"
        "- 'fair' = Adequate but missing some important information\n"
        "- 'poor' = Incomplete, missing several key details\n"
        "- 'very_poor' = Very incomplete or too vague"
    ),
    feedback_value_type=Literal["excellent", "good", "fair", "poor", "very_poor"],
    model=f"databricks:/{LLM_ENDPOINT}",
)

# Judge 3: Professional Tone
tone_judge = make_judge(
    name="professional_tone",
    instructions=(
        "Evaluate if the response in {{ outputs }} uses appropriate professional tone.\n\n"
        "Check for:\n"
        "- Courteous and respectful language\n"
        "- Appropriate formality (not too casual, not overly formal)\n"
        "- Clear and direct communication\n"
        "- Empathy and helpfulness\n\n"
        "Rate as:\n"
        "- 'excellent' = Excellent professional tone, clear and courteous\n"
        "- 'good' = Good tone with minor improvements possible\n"
        "- 'fair' = Acceptable but somewhat casual or unclear\n"
        "- 'poor' = Poor tone, too casual or unclear\n"
        "- 'very_poor' = Inappropriate or unprofessional tone"
    ),
    feedback_value_type=Literal["excellent", "good", "fair", "poor", "very_poor"],
    model=f"databricks:/{LLM_ENDPOINT}",
)

print("✓ Created 3 custom judges:")
print("  • hr_accuracy - factual correctness")
print("  • completeness - information completeness")
print("  • professional_tone - communication quality")

# Test a judge to verify it works
print("\nTesting judge creation...")
try:
    test_feedback = hr_accuracy_judge(
        inputs={"question": "Test question"},
        outputs="Test response",
        expectations={"expected_answer": "Test expected"}
    )
    print(f"✓ Judge test successful: {test_feedback.value}")
except Exception as e:
    print(f"⚠️ Judge test failed: {str(e)}")
    print("This may indicate endpoint configuration issues")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Generate Predictions
# MAGIC
# MAGIC Run the agent on each question to generate predictions.

# COMMAND ----------

# Prediction function with tracing
@mlflow.trace
def predict(question: str, question_idx: int) -> str:
    """Generate agent response for a question."""
    thread_id = f"eval-{EVALUATION_RUN_ID}-{question_idx:03d}"
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config
    )
    return result["messages"][-1].content

# Generate predictions
print("Generating predictions...")
predictions = []
for idx, row in eval_data.iterrows():
    pred = predict(row["question"], idx)
    predictions.append(pred)
    print(f"  {idx+1}/{len(eval_data)} complete")

# Transform data to MLflow GenAI format
eval_data_with_predictions = []
for idx, row in eval_data.iterrows():
    eval_data_with_predictions.append({
        "inputs": {"question": row["question"]},
        "outputs": predictions[idx],
        "expectations": {"expected_answer": row["expected_answer"]}
    })

print("\n✓ Predictions generated")
print(f"  Total predictions: {len(predictions)}")
print("\nSample prediction:")
print(f"  Q: {eval_data_with_predictions[0]['inputs']['question']}")
print(f"  A: {eval_data_with_predictions[0]['outputs'][:100]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Run Evaluation
# MAGIC
# MAGIC Evaluate all predictions using the custom judges.

# COMMAND ----------

# Run evaluation
results = mlflow.genai.evaluate(
    data=eval_data_with_predictions,
    scorers=[hr_accuracy_judge, completeness_judge, tone_judge],
)

print("✓ Evaluation complete!")
print("\nAggregate Metrics:")
print("=" * 80)

if hasattr(results, 'metrics'):
    for metric_name, value in results.metrics.items():
        print(f"  {metric_name}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Interpret Results
# MAGIC
# MAGIC Analyze detailed results to understand agent quality.

# COMMAND ----------

# View detailed results
print("Available result tables:")
print(list(results.tables.keys()))

# Get the results table (key might vary)
table_key = list(results.tables.keys())[0] if results.tables else None
if table_key:
    results_df = results.tables[table_key]

    print("\nDetailed Results by Question:")
    print("=" * 80)
    display(results_df)

    # Show sample of results with feedback
    print("\nSample Results with Rationales:")
    print("=" * 80)
    for idx, row in results_df.head(2).iterrows():
        if 'inputs' in row:
            print(f"\nQuestion: {row['inputs']}")
        if 'outputs' in row:
            print(f"Response: {row['outputs'][:150]}...")

        # Show judge scores and rationales
        for col in results_df.columns:
            if 'value' in col.lower():
                judge_name = col.split('/')[0]
                print(f"\n  {judge_name}: {row[col]}")
                # Look for rationale
                rationale_col = f"{judge_name}/rationale"
                if rationale_col in results_df.columns and pd.notna(row[rationale_col]):
                    print(f"    Rationale: {row[rationale_col][:150]}...")
else:
    print("No detailed results table found")
    print("Aggregate metrics only:")
    print(results.metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Analyze Feedback
# MAGIC
# MAGIC Review judge feedback to understand quality issues.

# COMMAND ----------

# Analyze judge feedback patterns
print("Judge Feedback Summary:")
print("=" * 80)

if table_key and results_df is not None:
    # Count feedback values per judge
    for col in results_df.columns:
        if 'value' in col.lower():
            judge_name = col.split('/')[0]
            print(f"\n{judge_name}:")
            value_counts = results_df[col].value_counts()
            for value, count in value_counts.items():
                pct = (count / len(results_df)) * 100
                print(f"  {value}: {count} ({pct:.1f}%)")

    # Identify issues (poor or very_poor ratings)
    print("\n\nResponses Needing Improvement:")
    print("=" * 80)

    found_issues = False
    for col in results_df.columns:
        if 'value' in col.lower():
            poor_responses = results_df[results_df[col].isin(['poor', 'very_poor'])]
            if len(poor_responses) > 0:
                found_issues = True
                judge_name = col.split('/')[0]
                print(f"\n{judge_name} - {len(poor_responses)} issues found")
                for idx, row in poor_responses.iterrows():
                    if 'inputs' in row:
                        print(f"\n  Question: {row['inputs']}")
                    if 'outputs' in row:
                        print(f"  Response: {row['outputs'][:150]}...")
                    print(f"  Rating: {row[col]}")
                    rationale_col = f"{judge_name}/rationale"
                    if rationale_col in results_df.columns and pd.notna(row[rationale_col]):
                        print(f"  Rationale: {row[rationale_col][:150]}...")

    if not found_issues:
        print("✅ All responses rated 'fair' or better!")
else:
    print("Cannot analyze feedback - no detailed results available")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Human Feedback Loop
# MAGIC
# MAGIC Improve judges based on human feedback.

# COMMAND ----------

# Simulate human feedback on a few examples
print("Human Feedback Collection Pattern:")
print("=" * 80)

# In production, collect real human ratings like this:
human_feedback_example = pd.DataFrame([
    {
        "question": "How much vacation time do employees get?",
        "agent_response": "Full-time employees get 15 days per year.",
        "judge_rating": "good",
        "human_rating": "fair",
        "human_note": "Accurate but should mention accrual rate (1.25 days/month)"
    },
    {
        "question": "What are core in-office days?",
        "agent_response": "Tuesday and Thursday are core days.",
        "judge_rating": "excellent",
        "human_rating": "excellent",
        "human_note": "Perfect - concise and accurate"
    },
    {
        "question": "Can I work remotely full-time?",
        "agent_response": "No, you need to be in the office sometimes.",
        "judge_rating": "fair",
        "human_rating": "poor",
        "human_note": "Too vague - should specify Tuesday/Thursday requirement"
    },
])

print("\nExample Human Feedback:")
display(human_feedback_example)

# Analyze judge vs human alignment
print("\nJudge Alignment Analysis:")
print("=" * 80)

rating_order = ["very_poor", "poor", "fair", "good", "excellent"]
for idx, row in human_feedback_example.iterrows():
    judge_idx = rating_order.index(row['judge_rating'])
    human_idx = rating_order.index(row['human_rating'])
    diff = abs(judge_idx - human_idx)
    aligned = "✓" if diff <= 1 else "✗"
    print(f"{aligned} Question {idx+1}: Judge={row['judge_rating']}, Human={row['human_rating']}")
    if diff > 1:
        print(f"   Note: {row['human_note']}")

print("\nImprovement Actions:")
print("  1. Collect 10-20 human ratings on diverse examples")
print("  2. Identify patterns where judge disagrees with humans")
print("  3. Refine judge instructions to align with human judgment")
print("  4. Re-run evaluation to confirm improved alignment")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Production Quality Gates
# MAGIC
# MAGIC Set up automated quality checks for deployments.

# COMMAND ----------

print("Production Quality Gate Pattern:")
print("=" * 80)

# Define quality requirements based on categorical ratings
# For production, require high percentage of "excellent" or "good" ratings
def check_quality_gates(results):
    """Return True if agent passes quality gates based on judge feedback."""
    print("\nQuality Gate Checks:")
    print("-" * 80)

    passed = True
    if not hasattr(results, 'metrics'):
        print("❌ No metrics available")
        return False

    print("\nAvailable metrics:")
    for metric_name, value in results.metrics.items():
        print(f"  {metric_name}: {value}")

    # Define requirements: judges should mostly rate as "good" or "excellent"
    # This is illustrative - adjust thresholds based on your requirements
    print("\nQuality Requirements:")
    print("  - Majority of responses should be rated 'good' or 'excellent'")
    print("  - Maximum 20% 'fair' ratings acceptable")
    print("  - No 'poor' or 'very_poor' ratings for production")

    # Check if we have categorical distribution in metrics
    # The actual metric structure depends on how MLflow aggregates categorical feedback
    for metric_name, value in results.metrics.items():
        if 'poor' in str(value).lower() or 'very_poor' in str(value).lower():
            passed = False
            print(f"  ❌ {metric_name}: Contains poor ratings")

    if passed:
        print("\n  ✅ All quality gates passed")
    else:
        print("\n  ❌ Quality gates failed")

    return passed

# Run quality gates check
print("\n" + "=" * 80)
gates_passed = check_quality_gates(results)

print("\n" + "=" * 80)
if gates_passed:
    print("✅ AGENT PASSES QUALITY GATES - Ready for deployment")
else:
    print("❌ AGENT FAILS QUALITY GATES - Improvements required before deployment")

# CI/CD integration example
print("\n\nCI/CD Integration Pattern:")
print("```python")
print("# In your deployment pipeline:")
print("results = mlflow.genai.evaluate(...)")
print("if not check_quality_gates(results):")
print("    raise Exception('Quality gates failed - blocking deployment')")
print("```")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What We Built
# MAGIC - ✅ Created evaluation dataset with ground truth
# MAGIC - ✅ Built 3 custom judges (accuracy, completeness, tone)
# MAGIC - ✅ Generated predictions with tracing
# MAGIC - ✅ Ran comprehensive evaluation
# MAGIC - ✅ Interpreted scores and identified issues
# MAGIC - ✅ Implemented human feedback loop
# MAGIC - ✅ Set up production quality gates
# MAGIC
# MAGIC ### Key Concepts
# MAGIC
# MAGIC **Custom Judges with make_genai_metric:**
# MAGIC ```python
# MAGIC judge = make_genai_metric(
# MAGIC     name="judge_name",
# MAGIC     definition="What to evaluate",
# MAGIC     grading_prompt="How to score (1-5 scale)",
# MAGIC     examples=[...],  # Few-shot examples
# MAGIC     model=f"endpoints:/{LLM_ENDPOINT}",
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **Evaluation Flow:**
# MAGIC 1. Create dataset with questions + expected answers
# MAGIC 2. Generate predictions (with tracing)
# MAGIC 3. Create custom judges for your domain
# MAGIC 4. Run `mlflow.genai.evaluate()` with judges
# MAGIC 5. Analyze results and identify issues
# MAGIC 6. Improve agent or judges based on feedback
# MAGIC
# MAGIC **Score Interpretation (1-5 scale):**
# MAGIC - 4.5-5.0: Excellent (90-100%)
# MAGIC - 3.5-4.5: Good (70-90%)
# MAGIC - 2.5-3.5: Fair (50-70%) - needs improvement
# MAGIC - <2.5: Poor (<50%) - major issues
# MAGIC
# MAGIC **Production Checklist:**
# MAGIC - ✅ Define quality thresholds per judge
# MAGIC - ✅ Run evaluation on every deployment
# MAGIC - ✅ Block deployment if gates fail
# MAGIC - ✅ Track scores over time in MLflow
# MAGIC - ✅ Collect human feedback regularly (10-20 samples)
# MAGIC - ✅ Refine judges based on human alignment
# MAGIC
# MAGIC ### Complete Evaluation Architecture
# MAGIC
# MAGIC Your agent now has:
# MAGIC - ✅ Tracing for observability (Module 03.01)
# MAGIC - ✅ Custom judges for quality assessment (Module 03.02)
# MAGIC - ✅ Production quality gates
# MAGIC - ✅ Human feedback integration
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC Continue to **Module 04: Multi-Tool Agents** to add Genie for structured data queries and UC Functions for custom tools.

# COMMAND ----------

print("✓ Evaluation tutorial complete!")
print("\nYour agent now has:")
print("  ✓ MLflow tracing for observability")
print("  ✓ Custom judges for domain-specific evaluation")
print("  ✓ Production quality gates")
print("  ✓ Human feedback integration")
print("\nNext: Continue to Module 04 for multi-tool agents")
