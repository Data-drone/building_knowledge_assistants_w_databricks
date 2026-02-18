# Databricks notebook source
# MAGIC %md
# MAGIC # Production Monitoring for Databricks Apps
# MAGIC
# MAGIC ## What You'll Do
# MAGIC Monitor the deployed Databricks App using MLflow traces and online evaluation checks.
# MAGIC
# MAGIC **Estimated time:** 15-20 minutes
# MAGIC
# MAGIC **Where this fits in Module 5:** After `02_apps_deployment.py`. This notebook verifies runtime behavior and observability signals for the deployed app.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC - Completed `01_apps_dev_loop.py` and `02_apps_deployment.py`
# MAGIC - OAuth token saved in `my-secrets/apps_oauth_token`
# MAGIC - App experiment exists at `/Shared/knowledge_assistant_agent_app`
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Verify app health through repeated `/invocations` calls
# MAGIC - Inspect recent MLflow traces for the app experiment
# MAGIC - Run lightweight online quality checks with custom judges

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies

# COMMAND ----------

%pip install -q --upgrade mlflow[databricks]>=3.1.0 requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configuration

# COMMAND ----------

import requests
import mlflow
from typing import Literal
from mlflow.genai.judges import make_judge

APP_URL = "https://knowledge-assistant-agent-app-984752964297111.11.azure.databricksapps.com"
EXPERIMENT_NAME = "/Shared/knowledge_assistant_agent_app"
OAUTH_TOKEN = dbutils.secrets.get("my-secrets", "apps_oauth_token")

mlflow.set_experiment(EXPERIMENT_NAME)

print("App URL:", APP_URL)
print("Experiment:", EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Generate Production-Like Traffic

# COMMAND ----------

queries = [
    "What is the vacation policy for full-time employees?",
    "How many employees are in Engineering?",
    "What equipment is provided for remote work?",
    "Give me a one-sentence summary of Databricks MCP.",
]

results = []
for q in queries:
    resp = requests.post(
        f"{APP_URL}/invocations",
        headers={
            "Authorization": f"Bearer {OAUTH_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={"input": [{"role": "user", "content": q}]},
        timeout=60,
        allow_redirects=False,
    )
    row = {
        "query": q,
        "status_code": resp.status_code,
        "content_type": resp.headers.get("content-type", ""),
        "body_preview": resp.text[:220],
    }
    results.append(row)
    print(f"{resp.status_code} | {q}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Quick Health Checks

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
# MAGIC ## Step 6: Online Quality Check with a Judge

# COMMAND ----------

judge = make_judge(
    name="response_clarity",
    instructions=(
        "Evaluate if {{ outputs }} is clear and directly answers {{ inputs }}.\n"
        "Return one of: excellent, good, fair, poor, very_poor."
    ),
    feedback_value_type=Literal["excellent", "good", "fair", "poor", "very_poor"],
    model="databricks:/databricks-claude-sonnet-4-5",
)

eval_rows = []
for r in success:
    eval_rows.append(
        {
            "inputs": {"question": r["query"]},
            "outputs": r["body_preview"],
            "expectations": {"style": "clear_and_direct"},
        }
    )

if eval_rows:
    eval_results = mlflow.genai.evaluate(data=eval_rows, scorers=[judge])
    print("Evaluation complete.")
    if hasattr(eval_results, "metrics"):
        print(eval_results.metrics)
else:
    print("Skipped evaluation: no successful JSON responses.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC - Generated synthetic production traffic against Databricks App `/invocations`
# MAGIC - Validated response health (status + non-HTML payload checks)
# MAGIC - Queried MLflow traces from app experiment
# MAGIC - Ran an online quality pass using an MLflow judge
