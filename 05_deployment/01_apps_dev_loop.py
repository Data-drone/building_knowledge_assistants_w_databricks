# Databricks notebook source
# MAGIC %md
# MAGIC # Apps Dev Loop: Local Changes to Live Agent
# MAGIC
# MAGIC ## What You'll Do
# MAGIC Learn the day-to-day development loop for an agent deployed with Databricks Apps.
# MAGIC
# MAGIC **Estimated time:** 10-15 minutes
# MAGIC
# MAGIC **Where this fits in Module 5:** Start here. This notebook establishes the inner development loop you will reuse in the rest of the module.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC - Databricks CLI configured for profile `adb-984752964297111`
# MAGIC - Databricks App created once as `knowledge-assistant-agent-app`
# MAGIC - App source code present under `apps/knowledge_assistant_agent`
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Understand where app runtime code lives in this repo
# MAGIC - Sync local changes to workspace files
# MAGIC - Redeploy quickly and verify `/invocations`
# MAGIC - Configure OAuth token usage for notebook testing

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: App Source Layout
# MAGIC
# MAGIC We deploy the app from:
# MAGIC
# MAGIC - `apps/knowledge_assistant_agent/app.yaml` (runtime command + env)
# MAGIC - `apps/knowledge_assistant_agent/requirements.txt`
# MAGIC - `apps/knowledge_assistant_agent/agent_server/start_server.py`
# MAGIC - `apps/knowledge_assistant_agent/agent_server/agent.py`
# MAGIC - `apps/knowledge_assistant_agent/agent_server/langgraph_agent.py`
# MAGIC
# MAGIC This app exposes a Responses API-compatible endpoint:
# MAGIC
# MAGIC `POST {APP_URL}/invocations`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: One-Time App Setup (Terminal)
# MAGIC
# MAGIC Run this from your local terminal (outside this notebook):
# MAGIC
# MAGIC ```bash
# MAGIC cd databricks_agent_bootcamp
# MAGIC
# MAGIC # Sync local app source to workspace files
# MAGIC DATABRICKS_USERNAME=$(databricks current-user me -p adb-984752964297111 -o json | jq -r '.userName')
# MAGIC databricks sync -p adb-984752964297111 \
# MAGIC   "apps/knowledge_assistant_agent" \
# MAGIC   "/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
# MAGIC
# MAGIC # Create app once
# MAGIC databricks apps create -p adb-984752964297111 knowledge-assistant-agent-app
# MAGIC
# MAGIC # Deploy
# MAGIC databricks apps deploy -p adb-984752964297111 knowledge-assistant-agent-app \
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
# MAGIC DATABRICKS_USERNAME=$(databricks current-user me -p adb-984752964297111 -o json | jq -r '.userName')
# MAGIC
# MAGIC databricks sync -p adb-984752964297111 \
# MAGIC   "apps/knowledge_assistant_agent" \
# MAGIC   "/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
# MAGIC
# MAGIC databricks apps deploy -p adb-984752964297111 knowledge-assistant-agent-app \
# MAGIC   --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
# MAGIC
# MAGIC databricks apps logs -p adb-984752964297111 knowledge-assistant-agent-app --tail-lines 120
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Store OAuth Token in a Secret
# MAGIC
# MAGIC Databricks Apps require OAuth token auth. Save one into a secret:
# MAGIC
# MAGIC ```bash
# MAGIC TOKEN=$(databricks auth token --profile adb-984752964297111 -o json | jq -r '.access_token // .token_value // .token')
# MAGIC databricks secrets create-scope my-secrets --profile adb-984752964297111 || true
# MAGIC databricks secrets put-secret my-secrets apps_oauth_token \
# MAGIC   --string-value "$TOKEN" \
# MAGIC   --profile adb-984752964297111
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Notebook Smoke Test Against the App

# COMMAND ----------

import requests

APP_URL = "https://knowledge-assistant-agent-app-984752964297111.11.azure.databricksapps.com"
OAUTH_TOKEN = dbutils.secrets.get("my-secrets", "apps_oauth_token")

payload = {
    "input": [
        {"role": "user", "content": "Reply with: apps dev loop smoke test passed."}
    ]
}

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

print("Status:", resp.status_code)
print("Content-Type:", resp.headers.get("content-type"))
print(resp.text[:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Troubleshooting
# MAGIC
# MAGIC - If you get HTML login content, your token is not valid OAuth for Apps.
# MAGIC - If you get `403` from `/invocations`, refresh your OAuth token and update the secret.
# MAGIC - If app returns 502, check logs:
# MAGIC   `databricks apps logs -p adb-984752964297111 knowledge-assistant-agent-app --tail-lines 200`
# MAGIC - If startup fails after code changes, redeploy and inspect the first stack trace in logs.
