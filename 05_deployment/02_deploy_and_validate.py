# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy and Validate the Agent App
# MAGIC
# MAGIC ## What You'll Do
# MAGIC Deploy the bootcamp agent as a Databricks App, configure authentication,
# MAGIC and validate the `/invocations` API including memory behaviour.
# MAGIC
# MAGIC **Estimated time:** 15-20 minutes
# MAGIC
# MAGIC **Where this fits in Module 5:** After `01_packaging_for_apps.py`, which
# MAGIC walked through the app code. This notebook deploys it and confirms
# MAGIC everything works end to end.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC - Completed `01_packaging_for_apps.py` (understand the app architecture)
# MAGIC - Databricks CLI configured for a profile that can deploy Apps
# MAGIC - App source code present under `apps/knowledge_assistant_agent`
# MAGIC
# MAGIC **Reference:** [Author an agent on Databricks Apps](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent-db-app)
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Create and deploy a Databricks App from the CLI
# MAGIC - Use the fast sync-and-redeploy iteration loop
# MAGIC - Configure OAuth token auth for notebook testing
# MAGIC - Validate endpoint health, response schema, and memory behaviour

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: One-Time App Setup (Terminal)
# MAGIC
# MAGIC Run this once from your local terminal:
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
# MAGIC # Create app (only needed once)
# MAGIC databricks apps create -p "$DATABRICKS_PROFILE" knowledge-assistant-agent-app
# MAGIC
# MAGIC # Deploy
# MAGIC databricks apps deploy -p "$DATABRICKS_PROFILE" knowledge-assistant-agent-app \
# MAGIC   --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Fast Iteration Loop (Terminal)
# MAGIC
# MAGIC After the initial setup, each code change follows a two-command cycle:
# MAGIC
# MAGIC ```bash
# MAGIC cd databricks_agent_bootcamp
# MAGIC DATABRICKS_PROFILE=<your-databricks-profile>
# MAGIC DATABRICKS_USERNAME=$(databricks current-user me -p "$DATABRICKS_PROFILE" -o json | jq -r '.userName')
# MAGIC
# MAGIC # Sync changed files
# MAGIC databricks sync -p "$DATABRICKS_PROFILE" \
# MAGIC   "apps/knowledge_assistant_agent" \
# MAGIC   "/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
# MAGIC
# MAGIC # Redeploy
# MAGIC databricks apps deploy -p "$DATABRICKS_PROFILE" knowledge-assistant-agent-app \
# MAGIC   --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
# MAGIC
# MAGIC # Check logs to verify startup
# MAGIC databricks apps logs -p "$DATABRICKS_PROFILE" knowledge-assistant-agent-app --tail-lines 120
# MAGIC ```
# MAGIC
# MAGIC To confirm the app is running:
# MAGIC
# MAGIC ```bash
# MAGIC databricks apps get -p "$DATABRICKS_PROFILE" knowledge-assistant-agent-app -o json
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Store OAuth Token in a Secret
# MAGIC
# MAGIC Databricks Apps require OAuth token auth (PATs are not supported). Store
# MAGIC one in a secret scope so notebooks can call the app:
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
# MAGIC ## Step 4: Configure Notebook Auth

# COMMAND ----------

import sys
import requests
import uuid

sys.path.append("/Workspace" + "/".join(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[:-2]))

from config import APP_NAME, get_app_url

APP_URL = get_app_url(APP_NAME)
OAUTH_TOKEN = dbutils.secrets.get("my-secrets", "apps_oauth_token")

print("App URL:", APP_URL)
print("Token loaded:", bool(OAUTH_TOKEN))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Memory Validation
# MAGIC
# MAGIC This test exercises both memory layers:
# MAGIC - **Short-term**: reuse the same `thread_id` across two turns to confirm
# MAGIC   context is preserved, then use a fresh thread to confirm isolation.
# MAGIC - **Long-term**: pass `user_id` so the agent's memory tools can read/write
# MAGIC   user-scoped facts in Lakebase.

# COMMAND ----------

thread_id = f"deploy-validation-{uuid.uuid4()}"
user_id = "bootcamp-tester"

test_cases = [
    (
        "Turn 1",
        {
            "input": [{"role": "user", "content": "How much vacation time do employees get?"}],
            "custom_inputs": {"thread_id": thread_id, "user_id": user_id},
        },
    ),
    (
        "Turn 2 (same thread)",
        {
            "input": [{"role": "user", "content": "What about sick leave?"}],
            "custom_inputs": {"thread_id": thread_id, "user_id": user_id},
        },
    ),
    (
        "Fresh thread (isolation check)",
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
    print(f"\n{i}. {label}")
    print(f"   thread_id: {payload['custom_inputs']['thread_id']}")
    print(f"   status: {resp.status_code}, valid_json: {ok}")
    print(f"   response: {resp.text[:500]}")
    if ok:
        passed += 1

print(f"\n{'='*50}")
print(f"Validation passed: {passed}/{len(test_cases)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: What to Look For
# MAGIC
# MAGIC - **Turn 1** should answer the vacation question directly.
# MAGIC - **Turn 2 (same thread)** should reference the earlier vacation context
# MAGIC   when answering about sick leave — this proves short-term memory works.
# MAGIC - **Fresh thread** should answer the sick leave question without any
# MAGIC   vacation context — this proves thread isolation works.
# MAGIC - All three should return `status=200` with JSON content (not an HTML
# MAGIC   login page).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Troubleshooting
# MAGIC
# MAGIC | Symptom | Cause | Fix |
# MAGIC |---|---|---|
# MAGIC | Response body is an HTML login page | OAuth token invalid for Apps | Refresh the token and update the secret (Step 3) |
# MAGIC | `403` from `/invocations` | Token expired | Re-run the token generation commands |
# MAGIC | `502` or `503` | App crashed on startup | Check logs: `databricks apps logs -p <profile> knowledge-assistant-agent-app --tail-lines 200` |
# MAGIC | Startup fails after code changes | Syntax or import error | Redeploy and inspect the first stack trace in logs |
# MAGIC | Memory not working | `ENABLE_MEMORY=false` in `app.yaml` | Set to `true` and redeploy |
# MAGIC
# MAGIC **Next step:** `03_production_monitoring.py` adds scorers and continuous
# MAGIC monitoring on the traces produced by this app.
