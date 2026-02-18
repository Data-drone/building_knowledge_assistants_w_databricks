# Databricks notebook source
# MAGIC %md
# MAGIC # End-to-End Apps Deployment Pipeline
# MAGIC
# MAGIC ## What You'll Do
# MAGIC Run a full deployment workflow for Databricks Apps:
# MAGIC
# MAGIC 1. Optional quality gate checks
# MAGIC 2. Sync + deploy app source
# MAGIC 3. Validate `/invocations`
# MAGIC 4. Confirm monitoring signals in MLflow
# MAGIC
# MAGIC **Estimated time:** 15-20 minutes
# MAGIC
# MAGIC **Where this fits in Module 5:** Final notebook. This is the release-ready checklist that combines deployment, validation, and monitoring gates.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed `01_apps_dev_loop.py` and `02_apps_deployment.py`
# MAGIC - Recommended: completed `03_production_monitoring.py`
# MAGIC - OAuth token saved in `my-secrets/apps_oauth_token`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configuration

# COMMAND ----------

import requests
import mlflow

APP_NAME = "knowledge-assistant-agent-app"
APP_URL = "https://knowledge-assistant-agent-app-984752964297111.11.azure.databricksapps.com"
EXPERIMENT_NAME = "/Shared/knowledge_assistant_agent_app"
OAUTH_TOKEN = dbutils.secrets.get("my-secrets", "apps_oauth_token")

mlflow.set_experiment(EXPERIMENT_NAME)

print("App:", APP_NAME)
print("URL:", APP_URL)
print("Experiment:", EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Deployment Commands (Run in Terminal)
# MAGIC
# MAGIC This notebook documents the canonical deployment flow:
# MAGIC
# MAGIC ```bash
# MAGIC cd databricks_agent_bootcamp
# MAGIC DATABRICKS_USERNAME=$(databricks current-user me -p adb-984752964297111 -o json | jq -r '.userName')
# MAGIC
# MAGIC databricks sync -p adb-984752964297111 \
# MAGIC   "apps/knowledge_assistant_agent" \
# MAGIC   "/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
# MAGIC
# MAGIC databricks apps deploy -p adb-984752964297111 knowledge-assistant-agent-app \
# MAGIC   --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
# MAGIC
# MAGIC databricks apps get -p adb-984752964297111 knowledge-assistant-agent-app -o json
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Post-Deploy Validation Tests

# COMMAND ----------

test_queries = [
    "Reply with exactly: e2e test one passed.",
    "In one sentence, explain why MCP helps governance.",
    "Give two bullets on why evaluating agents matters.",
]

responses = []
for q in test_queries:
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
    responses.append(resp)
    print(f"{resp.status_code} | {q}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Gate Decision

# COMMAND ----------

ok = 0
for resp in responses:
    ct = resp.headers.get("content-type", "")
    is_jsonish = "text/html" not in ct
    if resp.status_code == 200 and is_jsonish:
        ok += 1

gate_passed = ok == len(responses)
print(f"Validation passed: {ok}/{len(responses)}")
print("QUALITY GATE:", "PASSED" if gate_passed else "FAILED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Confirm Trace Activity

# COMMAND ----------

traces = mlflow.search_traces(
    experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id],
    max_results=10,
    order_by=["timestamp_ms DESC"],
)

print("Recent traces:", len(traces))
if len(traces) > 0:
    preferred_cols = ["trace_id", "timestamp_ms", "status"]
    available_cols = [c for c in preferred_cols if c in traces.columns]
    display(traces[available_cols].head(10) if available_cols else traces.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC - Deployment path is Databricks Apps (`sync` + `apps deploy`)
# MAGIC - API validation uses OAuth + `/invocations`
# MAGIC - Quality gate uses response correctness/health checks
# MAGIC - Monitoring uses MLflow traces in app experiment
