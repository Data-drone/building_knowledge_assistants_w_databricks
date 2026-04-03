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
# MAGIC - Validate short-term memory (`thread_id`) and long-term memory (`user_id`)
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
# MAGIC - `POST /invocations` (Responses API format)
# MAGIC - `GET /` (interactive chat UI — open the App URL in a browser)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Deploy (Terminal Commands)
# MAGIC
# MAGIC Run the commands below from your local terminal:
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
# MAGIC ## Step 3: Configure Notebook Auth
# MAGIC
# MAGIC Use OAuth token stored in secrets (from `01_apps_dev_loop.py`):

# COMMAND ----------

import sys

sys.path.append("/Workspace" + "/".join(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[:-2]))

from config import APP_NAME, get_app_url

APP_URL = get_app_url(APP_NAME)
OAUTH_TOKEN = dbutils.secrets.get("my-secrets", "apps_oauth_token")

print("App URL:", APP_URL)
print("Token loaded:", bool(OAUTH_TOKEN))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Memory Request Test
# MAGIC
# MAGIC Pass both `thread_id` (short-term) and `user_id` (long-term) so the agent
# MAGIC can use conversation history and user-scoped memory tools.

# COMMAND ----------

import requests
import uuid

thread_id = f"deploy-validation-{uuid.uuid4()}"
user_id = "bootcamp-tester"

request_turn_1 = {
    "input": [
        {"role": "user", "content": "How much vacation time do employees get?"}
    ],
    "custom_inputs": {"thread_id": thread_id, "user_id": user_id},
}
request_turn_2 = {
    "input": [{"role": "user", "content": "What about sick leave?"}],
    "custom_inputs": {"thread_id": thread_id, "user_id": user_id},
}

for label, request_body in [("Turn 1", request_turn_1), ("Turn 2", request_turn_2)]:
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
    print(f"\n{label}")
    print("thread_id:", thread_id)
    print("user_id:", user_id)
    print("Status:", response.status_code)
    print("Content-Type:", response.headers.get("content-type"))
    print(response.text[:1500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Multi-Request Validation
# MAGIC
# MAGIC Reuse one thread to validate conversation continuity, then use a fresh
# MAGIC thread to verify isolation.

# COMMAND ----------

multi_thread_id = f"deploy-validation-{uuid.uuid4()}"

test_cases = [
    (
        "Turn 1",
        {
            "input": [{"role": "user", "content": "How much vacation time do employees get?"}],
            "custom_inputs": {"thread_id": multi_thread_id, "user_id": user_id},
        },
    ),
    (
        "Turn 2 (same thread)",
        {
            "input": [{"role": "user", "content": "What about sick leave?"}],
            "custom_inputs": {"thread_id": multi_thread_id, "user_id": user_id},
        },
    ),
    (
        "Fresh thread",
        {
            "input": [{"role": "user", "content": "What about sick leave?"}],
            "custom_inputs": {"thread_id": f"deploy-validation-{uuid.uuid4()}", "user_id": user_id},
        },
    ),
]

passed = 0
for i, (label, payload) in enumerate(test_cases, 1):
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
    ok = resp.status_code == 200 and "text/html" not in (resp.headers.get("content-type") or "")
    print(f"{i}. {label}: status={resp.status_code}, app_json={ok}, thread_id={payload['custom_inputs']['thread_id']}")
    if ok:
        passed += 1

print(f"\nValidation passed: {passed}/{len(test_cases)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes
# MAGIC
# MAGIC - If response body is HTML sign-in page, your token is invalid for Apps.
# MAGIC - If you get repeated `403` responses, refresh the OAuth token in `my-secrets/apps_oauth_token`.
# MAGIC - If request fails with 5xx, inspect app logs from terminal:
# MAGIC   `databricks apps logs -p <your-databricks-profile> knowledge-assistant-agent-app --tail-lines 200`
# MAGIC - This bootcamp uses Databricks Apps as the only deployment path.
# MAGIC - Short-term memory requires `custom_inputs.thread_id` on each request.
# MAGIC - Long-term memory requires `custom_inputs.user_id` to scope per-user facts in Lakebase.
