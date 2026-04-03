# Databricks notebook source
# MAGIC %md
# MAGIC # Apps Dev Loop: Local Changes to Live Agent
# MAGIC
# MAGIC ## What You'll Do
# MAGIC Learn the day-to-day development loop for a stateful agent deployed with Databricks Apps.
# MAGIC
# MAGIC **Estimated time:** 10-15 minutes
# MAGIC
# MAGIC **Where this fits in Module 5:** Start here. This notebook establishes the inner development loop you will reuse in the rest of the module.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC - Databricks CLI configured for a profile that can deploy Apps
# MAGIC - Databricks App created once as `knowledge-assistant-agent-app`
# MAGIC - App source code present under `apps/knowledge_assistant_agent`
# MAGIC
# MAGIC **Why `ResponsesAgent` shows up here**
# MAGIC - The early notebooks use plain LangGraph flows to teach core agent ideas quickly
# MAGIC - The deployed app uses MLflow's `ResponsesAgent` interface because it provides the request,
# MAGIC   streaming, and tool-event shape expected by the production serving layer
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Understand where app runtime code lives in this repo
# MAGIC - Sync local changes to workspace files
# MAGIC - Redeploy quickly and verify `/invocations`
# MAGIC - Pass `thread_id` and `user_id` so short-term and long-term memory work as intended
# MAGIC - Configure OAuth token usage for notebook testing

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: App Source Layout
# MAGIC
# MAGIC We deploy the app from:
# MAGIC
# MAGIC - `apps/knowledge_assistant_agent/app.yaml` (runtime command + env vars)
# MAGIC - `apps/knowledge_assistant_agent/requirements.txt`
# MAGIC - `apps/knowledge_assistant_agent/agent_server/start_server.py`
# MAGIC - `apps/knowledge_assistant_agent/agent_server/agent.py`
# MAGIC - `apps/knowledge_assistant_agent/agent_server/langgraph_agent.py`
# MAGIC - `apps/knowledge_assistant_agent/agent_server/utils_memory.py` (long-term memory tools)
# MAGIC - `apps/knowledge_assistant_agent/static/index.html` (chat UI)
# MAGIC
# MAGIC The app exposes:
# MAGIC
# MAGIC - `POST {APP_URL}/invocations` — Responses API-compatible endpoint
# MAGIC - `GET {APP_URL}/` — Interactive chat UI (open in browser)
# MAGIC
# MAGIC **Memory** is controlled by `ENABLE_MEMORY` in `app.yaml` (currently `true`):
# MAGIC - **Short-term** — pass `custom_inputs.thread_id` per request. Same `thread_id` -> same conversation. New `thread_id` -> fresh conversation.
# MAGIC - **Long-term** — pass `custom_inputs.user_id` to persist facts across sessions. The agent has `get_user_memory`, `save_user_memory`, and `delete_user_memory` tools backed by Lakebase.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: One-Time App Setup (Terminal)
# MAGIC
# MAGIC Run this from your local terminal (outside this notebook):
# MAGIC
# MAGIC ```bash
# MAGIC cd databricks_agent_bootcamp
# MAGIC DATABRICKS_PROFILE=<your-databricks-profile>
# MAGIC
# MAGIC # Sync local app source to workspace files
# MAGIC DATABRICKS_USERNAME=$(databricks current-user me -p "$DATABRICKS_PROFILE" -o json | jq -r '.userName')
# MAGIC databricks sync -p "$DATABRICKS_PROFILE" \
# MAGIC   "apps/knowledge_assistant_agent" \
# MAGIC   "/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
# MAGIC
# MAGIC # Create app once
# MAGIC databricks apps create -p "$DATABRICKS_PROFILE" knowledge-assistant-agent-app
# MAGIC
# MAGIC # Deploy
# MAGIC databricks apps deploy -p "$DATABRICKS_PROFILE" knowledge-assistant-agent-app \
# MAGIC   --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Fast Iteration Loop (Terminal)
# MAGIC
# MAGIC For each code change, run:
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
# MAGIC databricks apps logs -p "$DATABRICKS_PROFILE" knowledge-assistant-agent-app --tail-lines 120
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Store OAuth Token in a Secret
# MAGIC
# MAGIC Databricks Apps require OAuth token auth. Save one into a secret:
# MAGIC
# MAGIC ```bash
# MAGIC DATABRICKS_PROFILE=<your-databricks-profile>
# MAGIC TOKEN=$(databricks auth token --profile "$DATABRICKS_PROFILE" -o json | jq -r '.access_token // .token_value // .token')
# MAGIC databricks secrets create-scope my-secrets --profile "$DATABRICKS_PROFILE" || true
# MAGIC databricks secrets put-secret my-secrets apps_oauth_token \
# MAGIC   --string-value "$TOKEN" \
# MAGIC   --profile "$DATABRICKS_PROFILE"
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Notebook Smoke Test Against the App
# MAGIC
# MAGIC This test exercises both memory layers:
# MAGIC - **Short-term**: reuse the same `thread_id` across two turns, then use a fresh
# MAGIC   thread to confirm isolation.
# MAGIC - **Long-term**: pass `user_id` so the agent's memory tools can read/write
# MAGIC   user-scoped facts in Lakebase.

# COMMAND ----------

import sys
import requests
import uuid

sys.path.append("/Workspace" + "/".join(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[:-2]))

from config import APP_NAME, get_app_url

APP_URL = get_app_url(APP_NAME)
OAUTH_TOKEN = dbutils.secrets.get("my-secrets", "apps_oauth_token")
THREAD_ID = f"apps-dev-loop-{uuid.uuid4()}"
FRESH_THREAD_ID = f"apps-dev-loop-{uuid.uuid4()}"
USER_ID = "bootcamp-tester"

turn_1 = {
    "input": [
        {"role": "user", "content": "How much vacation time do employees get?"}
    ],
    "custom_inputs": {"thread_id": THREAD_ID, "user_id": USER_ID},
}

turn_2 = {
    "input": [{"role": "user", "content": "What about sick leave?"}],
    "custom_inputs": {"thread_id": THREAD_ID, "user_id": USER_ID},
}

fresh_thread = {
    "input": [{"role": "user", "content": "What about sick leave?"}],
    "custom_inputs": {"thread_id": FRESH_THREAD_ID, "user_id": USER_ID},
}

for label, payload in [
    ("Turn 1", turn_1),
    ("Turn 2 (same thread)", turn_2),
    ("Fresh thread", fresh_thread),
]:
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
    print(f"\n{label}")
    print("thread_id:", payload["custom_inputs"]["thread_id"])
    print("user_id:", payload["custom_inputs"].get("user_id", ""))
    print("Status:", resp.status_code)
    print("Content-Type:", resp.headers.get("content-type"))
    print(resp.text[:700])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Troubleshooting
# MAGIC
# MAGIC - If you get HTML login content, your token is not valid OAuth for Apps.
# MAGIC - If you get `403` from `/invocations`, refresh your OAuth token and update the secret.
# MAGIC - If app returns 502, check logs:
# MAGIC   `databricks apps logs -p <your-databricks-profile> knowledge-assistant-agent-app --tail-lines 200`
# MAGIC - If startup fails after code changes, redeploy and inspect the first stack trace in logs.
