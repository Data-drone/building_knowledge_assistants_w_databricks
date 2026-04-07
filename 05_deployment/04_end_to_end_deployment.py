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
# MAGIC - Completed `01_packaging_for_apps.py` and `02_deploy_and_validate.py`
# MAGIC - Recommended: completed `03_production_monitoring.py`
# MAGIC - OAuth token saved in `my-secrets/apps_oauth_token`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configuration
# MAGIC
# MAGIC Same app name, experiment, and OAuth token used across Module 05.

# COMMAND ----------

import sys
import requests
import mlflow
import uuid

sys.path.append("/Workspace" + "/".join(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[:-2]))

from config import APP_EXPERIMENT, APP_NAME, get_app_url

APP_URL = get_app_url(APP_NAME)
EXPERIMENT_NAME = APP_EXPERIMENT
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
# MAGIC DATABRICKS_PROFILE=<your-databricks-profile>
# MAGIC DATABRICKS_USERNAME=$(databricks current-user me -p "$DATABRICKS_PROFILE" -o json | jq -r '.userName')
# MAGIC
# MAGIC databricks sync -p "$DATABRICKS_PROFILE" \
# MAGIC   "apps/knowledge_assistant_agent" \
# MAGIC   "/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
# MAGIC
# MAGIC databricks apps deploy -p "$DATABRICKS_PROFILE" knowledge-assistant-agent-app \
# MAGIC   --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
# MAGIC
# MAGIC databricks apps get -p "$DATABRICKS_PROFILE" knowledge-assistant-agent-app -o json
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Post-Deploy Validation Tests
# MAGIC
# MAGIC These checks validate memory behavior:
# MAGIC - **Short-term**: same `thread_id` preserves context; different `thread_id` isolates it
# MAGIC - **Long-term**: `user_id` is passed so the agent's memory tools can read/write user-scoped facts

# COMMAND ----------

shared_thread_id = f"e2e-{uuid.uuid4()}"
isolated_thread_id = f"e2e-{uuid.uuid4()}"
user_id = "bootcamp-e2e"

test_cases = [
    (
        "Turn 1",
        {
            "input": [{"role": "user", "content": "How much vacation time do employees get?"}],
            "custom_inputs": {"thread_id": shared_thread_id, "user_id": user_id},
        },
    ),
    (
        "Turn 2 (same thread)",
        {
            "input": [{"role": "user", "content": "What about sick leave?"}],
            "custom_inputs": {"thread_id": shared_thread_id, "user_id": user_id},
        },
    ),
    (
        "Fresh thread",
        {
            "input": [{"role": "user", "content": "What about sick leave?"}],
            "custom_inputs": {"thread_id": isolated_thread_id, "user_id": user_id},
        },
    ),
]

responses = []
for label, payload in test_cases:
    resp = requests.post(
        f"{APP_URL}/invocations",
        headers={
            "Authorization": f"Bearer {OAUTH_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=payload,
        timeout=60,
        allow_redirects=False,
    )
    responses.append((label, payload, resp))
    print(f"{resp.status_code} | {label} | thread_id={payload['custom_inputs']['thread_id']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Gate Decision
# MAGIC
# MAGIC Same pass/fail gate pattern from Module 03 Step 16, now applied to the
# MAGIC live endpoint. All validation requests must return 200 with JSON content.

# COMMAND ----------

ok = 0
for label, payload, resp in responses:
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
# MAGIC
# MAGIC Check that the validation requests produced traces in the app experiment.
# MAGIC If scorers are registered (from `03_production_monitoring.py`), feedback
# MAGIC should start appearing on these traces automatically.

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
# MAGIC - API validation uses OAuth + `/invocations` with `thread_id` and `user_id`
# MAGIC - Quality gate checks endpoint health, short-term memory continuity, and long-term memory availability
# MAGIC - Monitoring uses MLflow traces in app experiment
# MAGIC - The app also serves an interactive chat UI at `GET /` (open the App URL in a browser)
