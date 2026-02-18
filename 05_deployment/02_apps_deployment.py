# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy to Databricks Apps
# MAGIC
# MAGIC ## What You'll Do
# MAGIC Deploy the bootcamp agent as a Databricks App and validate the `/invocations` API.
# MAGIC
# MAGIC **Estimated time:** 10-15 minutes
# MAGIC
# MAGIC **Where this fits in Module 5:** After `01_apps_dev_loop.py`. This notebook covers repeatable deployment validation before adding production monitoring.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC - Completed `01_apps_dev_loop.py`
# MAGIC - OAuth token saved in `my-secrets/apps_oauth_token`
# MAGIC - App name `knowledge-assistant-agent-app` already exists
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Deploy app source code to Databricks Apps
# MAGIC - Validate endpoint health and response schema
# MAGIC - Run production-style API checks from notebook and terminal

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: App Runtime Summary
# MAGIC
# MAGIC The deployment artifact is the app source folder:
# MAGIC
# MAGIC `apps/knowledge_assistant_agent`
# MAGIC
# MAGIC Runtime command is defined in `app.yaml` and starts:
# MAGIC
# MAGIC `python -m agent_server.start_server`
# MAGIC
# MAGIC The server exposes:
# MAGIC
# MAGIC `POST /invocations` (Responses API format)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Deploy (Terminal Commands)
# MAGIC
# MAGIC Run the commands below from your local terminal:
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
# MAGIC ## Step 3: Configure Notebook Auth
# MAGIC
# MAGIC Use OAuth token stored in secrets (from `01_apps_dev_loop.py`):

# COMMAND ----------

APP_URL = "https://knowledge-assistant-agent-app-984752964297111.11.azure.databricksapps.com"
OAUTH_TOKEN = dbutils.secrets.get("my-secrets", "apps_oauth_token")

print("App URL:", APP_URL)
print("Token loaded:", bool(OAUTH_TOKEN))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Non-Streaming Request Test

# COMMAND ----------

import requests

request_body = {
    "input": [
        {"role": "user", "content": "What is Databricks MCP in one sentence?"}
    ]
}

response = requests.post(
    f"{APP_URL}/invocations",
    headers={
        "Authorization": f"Bearer {OAUTH_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    },
    json=request_body,
    timeout=60,
    allow_redirects=False,
)

print("Status:", response.status_code)
print("Content-Type:", response.headers.get("content-type"))
print(response.text[:1500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Multi-Request Validation

# COMMAND ----------

test_queries = [
    "Say hi in one short sentence.",
    "What is MLflow tracing in one sentence?",
    "Give me two bullets on why agent evaluation matters.",
]

passed = 0
for i, query in enumerate(test_queries, 1):
    resp = requests.post(
        f"{APP_URL}/invocations",
        headers={
            "Authorization": f"Bearer {OAUTH_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={"input": [{"role": "user", "content": query}]},
        timeout=60,
        allow_redirects=False,
    )
    ok = resp.status_code == 200 and "text/html" not in (resp.headers.get("content-type") or "")
    print(f"{i}. status={resp.status_code}, app_json={ok}")
    if ok:
        passed += 1

print(f"\nValidation passed: {passed}/{len(test_queries)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC
# MAGIC - If response body is HTML sign-in page, your token is invalid for Apps.
# MAGIC - If you get repeated `403` responses, refresh the OAuth token in `my-secrets/apps_oauth_token`.
# MAGIC - If request fails with 5xx, inspect app logs from terminal:
# MAGIC   `databricks apps logs -p adb-984752964297111 knowledge-assistant-agent-app --tail-lines 200`
# MAGIC - This tutorial path replaces `agents.deploy()` with Databricks Apps deployment.
