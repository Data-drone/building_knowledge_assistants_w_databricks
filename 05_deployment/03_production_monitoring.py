# Databricks notebook source
# MAGIC %md
# MAGIC # Production Monitoring for Databricks Apps
# MAGIC
# MAGIC ## What You'll Do
# MAGIC Build a production-grade monitoring pipeline: create scorers, run batch
# MAGIC evaluation, register scorers for continuous monitoring on live traces, and
# MAGIC manage their lifecycle.
# MAGIC
# MAGIC **Estimated time:** 20-25 minutes
# MAGIC
# MAGIC **Where this fits in Module 5:** After `02_deploy_and_validate.py`. This notebook
# MAGIC covers observability, quality evaluation, and continuous monitoring.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC - Completed `01_packaging_for_apps.py` and `02_deploy_and_validate.py`
# MAGIC - OAuth token saved in `my-secrets/apps_oauth_token`
# MAGIC - App experiment exists at `/Shared/knowledge_assistant_agent_app`
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Verify app health through `/invocations` calls and MLflow traces
# MAGIC - Create scorers: built-in LLM judges, custom LLM judges, and code-based scorers
# MAGIC - Run batch evaluation with `mlflow.genai.evaluate()`
# MAGIC - Register scorers and start continuous production monitoring with sampling
# MAGIC - Manage production scorers: list, update, stop, delete

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies
# MAGIC
# MAGIC Only MLflow and requests are needed here — the app does the heavy lifting.

# COMMAND ----------

%pip install -q --upgrade \
  "mlflow[databricks]>=3.10,<3.11" \
  requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configuration
# MAGIC
# MAGIC Points at the same experiment the app writes traces to. All scorers and
# MAGIC evaluation results land in this experiment.

# COMMAND ----------

import sys
import requests
import uuid
import mlflow
from typing import Literal

sys.path.append("/Workspace" + "/".join(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[:-2]))

from config import APP_EXPERIMENT, APP_NAME, get_app_url

APP_URL = get_app_url(APP_NAME)
EXPERIMENT_NAME = APP_EXPERIMENT
OAUTH_TOKEN = dbutils.secrets.get("my-secrets", "apps_oauth_token")

mlflow.set_experiment(EXPERIMENT_NAME)

print("App URL:", APP_URL)
print("Experiment:", EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Generate Production-Like Traffic
# MAGIC
# MAGIC Each query gets its own `thread_id` (isolated conversations) and a shared
# MAGIC `user_id` so the agent's long-term memory tools are active.

# COMMAND ----------

queries = [
    "What is the vacation policy for full-time employees?",
    "How many employees are in Engineering?",
    "What equipment is provided for remote work?",
    "Give me a one-sentence summary of Databricks MCP.",
]

user_id = "bootcamp-monitor"

results = []
for q in queries:
    resp = requests.post(
        f"{APP_URL}/invocations",
        headers={
            "Authorization": f"Bearer {OAUTH_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "input": [{"role": "user", "content": q}],
            "custom_inputs": {"thread_id": f"monitor-{uuid.uuid4()}", "user_id": user_id},
        },
        timeout=60,
        allow_redirects=False,
    )
    row = {
        "query": q,
        "status_code": resp.status_code,
        "content_type": resp.headers.get("content-type", ""),
        "body": resp.text,
    }
    results.append(row)
    print(f"{resp.status_code} | {q}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Quick Health Checks
# MAGIC
# MAGIC Before scoring quality, confirm that responses came back as JSON (not an
# MAGIC HTML login page) and returned 200.

# COMMAND ----------

success = [r for r in results if r["status_code"] == 200 and "text/html" not in r["content_type"]]
failures = [r for r in results if r not in success]

print(f"Successful JSON responses: {len(success)}/{len(results)}")
if failures:
    print("\nFailures:")
    for f in failures:
        print("-", f["status_code"], f["query"], "|", f["content_type"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Inspect Recent MLflow Traces
# MAGIC
# MAGIC Confirm the app is writing traces before we score them. If no traces
# MAGIC appear, the app experiment path may be misconfigured.

# COMMAND ----------

traces = mlflow.search_traces(
    experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id],
    max_results=20,
    order_by=["timestamp_ms DESC"],
)

print(f"Recent traces found: {len(traces)}")
if len(traces) > 0:
    preferred_cols = ["trace_id", "timestamp_ms", "execution_duration", "status"]
    available_cols = [c for c in preferred_cols if c in traces.columns]
    display(traces[available_cols].head(10) if available_cols else traces.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Understanding Scorers
# MAGIC
# MAGIC Module 03 introduced these three scorer types for development-time evaluation.
# MAGIC The same scorers also work for **production monitoring** on live traffic —
# MAGIC you register them once and MLflow runs them automatically on new traces.
# MAGIC
# MAGIC | Type | When to use | Example |
# MAGIC |------|-------------|---------|
# MAGIC | **Built-in LLM judge** | Quick start with research-validated criteria | `Safety`, `RelevanceToQuery`, `Guidelines` |
# MAGIC | **Custom LLM judge** | Domain-specific evaluation with custom prompts | `make_judge(name=..., instructions=...)` |
# MAGIC | **Code-based scorer** | Deterministic checks (length, format, keywords) | `@scorer` decorated function |
# MAGIC
# MAGIC We create one of each type below, then register them for production
# MAGIC monitoring.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6a: Built-in LLM Judges
# MAGIC
# MAGIC Built-in judges are pre-validated and require minimal configuration.
# MAGIC
# MAGIC - `Safety` checks for harmful or toxic content
# MAGIC - `Guidelines` checks custom natural-language rules (pass/fail)

# COMMAND ----------

from mlflow.genai.scorers import Safety, Guidelines

safety_scorer = Safety()

guidelines_scorer = Guidelines(
    name="professional_tone",
    guidelines=[
        "The response must use a professional, helpful tone",
        "The response must not include casual slang or emojis",
    ],
)

print("Built-in scorers created: Safety, Guidelines (professional_tone)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6b: Custom LLM Judge with `make_judge`
# MAGIC
# MAGIC `make_judge` lets you write your own grading criteria and choose a
# MAGIC feedback type (categorical, boolean, numeric). This judge evaluates
# MAGIC whether responses are clear and directly answer the question.

# COMMAND ----------

from mlflow.genai.judges import make_judge

clarity_judge = make_judge(
    name="response_clarity",
    instructions=(
        "Evaluate if {{ outputs }} is clear and directly answers {{ inputs }}.\n"
        "Return one of: excellent, good, fair, poor, very_poor."
    ),
    feedback_value_type=Literal["excellent", "good", "fair", "poor", "very_poor"],
    model="endpoints:/databricks-claude-sonnet-4-6",
)

print("Custom LLM judge created: response_clarity")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6c: Code-Based Scorer with `@scorer`
# MAGIC
# MAGIC Code-based scorers run deterministic Python logic — fast, free (no LLM
# MAGIC calls), and useful for format/length/keyword checks.
# MAGIC
# MAGIC **Production monitoring requirement:** all imports must be inside the
# MAGIC function body so the monitoring service can serialize the scorer.

# COMMAND ----------

from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

@scorer
def response_length_check(outputs) -> Feedback:
    import json
    text = str(outputs) if not isinstance(outputs, str) else outputs
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "output" in parsed:
            text = str(parsed["output"])
    except (json.JSONDecodeError, TypeError):
        pass
    word_count = len(text.split())
    ok = word_count >= 10
    return Feedback(
        value="yes" if ok else "no",
        rationale=f"Response has {word_count} words ({'sufficient' if ok else 'too short'})",
    )

print("Code-based scorer created: response_length_check")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Batch Evaluation During Development
# MAGIC
# MAGIC Use `mlflow.genai.evaluate()` to run all scorers against collected
# MAGIC responses. This is the development-time workflow — you run the same
# MAGIC scorers manually to verify quality before enabling continuous monitoring.

# COMMAND ----------

eval_rows = []
for r in success:
    eval_rows.append(
        {
            "inputs": {"question": r["query"]},
            "outputs": r["body"],
        }
    )

if eval_rows:
    all_scorers = [safety_scorer, guidelines_scorer, clarity_judge, response_length_check]
    eval_results = mlflow.genai.evaluate(data=eval_rows, scorers=all_scorers)
    print("Batch evaluation complete.")
    if hasattr(eval_results, "metrics"):
        print(eval_results.metrics)
else:
    print("Skipped evaluation: no successful JSON responses.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Register Scorers for Production Monitoring
# MAGIC
# MAGIC Once scorers work in batch evaluation, register them for **continuous
# MAGIC monitoring**. Registered scorers run automatically on new traces logged
# MAGIC to the experiment — no manual intervention needed.
# MAGIC
# MAGIC Each scorer gets a `ScorerSamplingConfig` that controls what fraction
# MAGIC of traces it evaluates:
# MAGIC - `sample_rate=1.0` — evaluate every trace (use for critical or cheap checks)
# MAGIC - `sample_rate=0.3` — evaluate 30% of traces (use for expensive LLM judges)
# MAGIC
# MAGIC Up to 20 scorers can be registered per experiment.

# COMMAND ----------

from mlflow.genai.scorers import ScorerSamplingConfig

registered_safety = safety_scorer.register(name="bootcamp_safety")
registered_safety = registered_safety.start(
    sampling_config=ScorerSamplingConfig(sample_rate=1.0),
)
print("Registered: bootcamp_safety (sample_rate=1.0)")

registered_guidelines = guidelines_scorer.register(name="bootcamp_professional_tone")
registered_guidelines = registered_guidelines.start(
    sampling_config=ScorerSamplingConfig(sample_rate=0.5),
)
print("Registered: bootcamp_professional_tone (sample_rate=0.5)")

registered_clarity = clarity_judge.register(name="bootcamp_response_clarity")
registered_clarity = registered_clarity.start(
    sampling_config=ScorerSamplingConfig(sample_rate=0.3),
)
print("Registered: bootcamp_response_clarity (sample_rate=0.3)")

registered_length = response_length_check.register(name="bootcamp_response_length")
registered_length = registered_length.start(
    sampling_config=ScorerSamplingConfig(sample_rate=1.0),
)
print("Registered: bootcamp_response_length (sample_rate=1.0)")

print("\n4 scorers now monitoring new traces in this experiment.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Manage Production Scorers
# MAGIC
# MAGIC ### List All Registered Scorers

# COMMAND ----------

from mlflow.genai.scorers import list_scorers

all_registered = list_scorers()
print(f"Registered scorers: {len(all_registered)}\n")
for s in all_registered:
    name = getattr(s, "_server_name", getattr(s, "name", "unknown"))
    rate = getattr(s, "sample_rate", "N/A")
    print(f"  {name}: sample_rate={rate}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Update a Scorer's Sampling Rate
# MAGIC
# MAGIC Scorer operations are immutable — `update()` returns a new instance with
# MAGIC the changed config.

# COMMAND ----------

updated_clarity = registered_clarity.update(
    sampling_config=ScorerSamplingConfig(sample_rate=0.5),
)
print(f"bootcamp_response_clarity updated: sample_rate {registered_clarity.sample_rate} -> {updated_clarity.sample_rate}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### (Optional) Clean Up Tutorial Scorers
# MAGIC
# MAGIC **Skip this cell if you want scorers to keep monitoring your app.**
# MAGIC The four scorers registered in Step 8 will continue scoring new traces
# MAGIC at their configured sampling rates until you stop or delete them.
# MAGIC
# MAGIC Run this cell only when you want to remove them entirely.

# COMMAND ----------

# from mlflow.genai.scorers import delete_scorer
#
# for name in [
#     "bootcamp_safety",
#     "bootcamp_professional_tone",
#     "bootcamp_response_clarity",
#     "bootcamp_response_length",
# ]:
#     try:
#         delete_scorer(name=name)
#         print(f"Deleted: {name}")
#     except Exception as e:
#         print(f"Skipped {name}: {e}")
#
# print("\nAll tutorial scorers cleaned up.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC - Generated synthetic production traffic against `/invocations` (with `thread_id` + `user_id`)
# MAGIC - Validated response health and inspected MLflow traces
# MAGIC - Created three types of scorers:
# MAGIC   - **Built-in LLM judges**: `Safety` (harmful content) and `Guidelines` (custom rules)
# MAGIC   - **Custom LLM judge**: `make_judge` with categorical feedback (clarity grading)
# MAGIC   - **Code-based scorer**: `@scorer` for deterministic checks (response length)
# MAGIC - Ran batch evaluation with `mlflow.genai.evaluate()` using all four scorers
# MAGIC - Registered scorers for **continuous production monitoring** with per-scorer sampling rates
# MAGIC - Managed scorer lifecycle: `list_scorers`, `update`, `delete_scorer` (cleanup is opt-in)
# MAGIC
# MAGIC **Next step:** `04_end_to_end_deployment.py` combines deployment, validation, and monitoring into a single release gate.
