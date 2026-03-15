# Knowledge Assistant on Databricks Apps

This directory contains the canonical Databricks Apps deployment for the
bootcamp agent. It exposes a Responses API-compatible `/invocations` endpoint
using `mlflow.genai.agent_server.AgentServer`.

This is the only deployment path used in the bootcamp. The deployment notebooks
in `05_deployment` assume this app layout and runtime contract.

The app supports both **short-term** and **long-term** memory when
`ENABLE_MEMORY=true`, matching the Databricks stateful agent guidance:
<https://docs.databricks.com/aws/en/generative-ai/agent-framework/stateful-agents>

- **Short-term memory** — `AsyncCheckpointSaver` preserves conversation state
  within a session (`thread_id`).
- **Long-term memory** — `AsyncDatabricksStore` gives the agent three tools
  (`get_user_memory`, `save_user_memory`, `delete_user_memory`) that persist
  facts about a user across sessions using semantic search over Lakebase.

The current app configuration uses an autoscaling Lakebase target:

- `LAKEBASE_AUTOSCALING_PROJECT=knowledge-assistant-state`
- `LAKEBASE_AUTOSCALING_BRANCH=production`
- `EMBEDDING_ENDPOINT=databricks-gte-large-en`
- `EMBEDDING_DIMS=1024`
- `ENABLE_MEMORY=true`

## Request Contract

When memory is enabled, clients should pass `custom_inputs.thread_id` for
short-term (conversation) memory and `custom_inputs.user_id` for long-term
(cross-session) memory.

- Same `thread_id` = same conversation state
- Different `thread_id` = isolated conversation
- If `thread_id` is omitted, the server generates a new thread ID for that request
- `user_id` ties long-term memories to a specific user; if omitted, memory tools
  gracefully return "not available"

Example request body:

```json
{
  "input": [
    {
      "role": "user",
      "content": "How much vacation time do employees get?"
    }
  ],
  "custom_inputs": {
    "thread_id": "user:demo/session:memory-smoke-test",
    "user_id": "demo@example.com"
  }
}
```

## Deploy

```bash
DATABRICKS_USERNAME=$(databricks current-user me | jq -r .userName)
databricks sync . "/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
databricks apps create knowledge-assistant-agent-app
databricks apps deploy knowledge-assistant-agent-app \
  --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
```

## Lakebase Permissions

Both short-term (checkpoint) and long-term (store) memory need permission to
create and update tables in the Lakebase branch. Grant the app's Postgres role
access to schema `public` before testing `/invocations`.

```sql
GRANT USAGE, CREATE ON SCHEMA public TO "<app-service-principal-client-id>";
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO "<app-service-principal-client-id>";
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "<app-service-principal-client-id>";
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE ON TABLES TO "<app-service-principal-client-id>";
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO "<app-service-principal-client-id>";
```

## Query (CLI)

```bash
APP_URL=$(databricks apps get knowledge-assistant-agent-app -o json | jq -r '.url')
TOKEN=$(databricks auth token -o json | jq -r '.access_token')
THREAD_ID="user:demo/session:memory-smoke-test"
USER_ID="demo@example.com"

# First message — the agent checks long-term memory for the user
curl -sS -X POST "$APP_URL/invocations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": [{\"role\": \"user\", \"content\": \"How much vacation time do employees get?\"}],
    \"custom_inputs\": {\"thread_id\": \"$THREAD_ID\", \"user_id\": \"$USER_ID\"}
  }" | jq .

# Follow-up in the same thread — short-term memory keeps context
curl -sS -X POST "$APP_URL/invocations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"input\": [{\"role\": \"user\", \"content\": \"What about sick leave?\"}],
    \"custom_inputs\": {\"thread_id\": \"$THREAD_ID\", \"user_id\": \"$USER_ID\"}
  }" | jq .
```

## Query (Notebook)

If you want to call the deployed app from a Databricks notebook, use OAuth auth
(not `dbutils.notebook...apiToken()`).

### 1) Store an OAuth token in Databricks Secrets (run in terminal or `%sh`)

```bash
TOKEN=$(databricks auth token -o json | jq -r '.access_token // .token_value // .token')
databricks secrets create-scope my-secrets || true
databricks secrets put-secret my-secrets apps_oauth_token \
  --string-value "$TOKEN"
```

### 2) Invoke `/invocations` from a notebook cell

```python
import requests

APP_URL = "<your-app-url>"  # e.g. from `databricks apps get <app-name> -o json | jq -r '.url'`
OAUTH_TOKEN = dbutils.secrets.get("my-secrets", "apps_oauth_token")

resp = requests.post(
    f"{APP_URL}/invocations",
    headers={
        "Authorization": f"Bearer {OAUTH_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    },
    json={"input": [{"role": "user", "content": "What is Databricks MCP in one sentence?"}]},
    timeout=60,
    allow_redirects=False,
)

print("Status:", resp.status_code)
print("Content-Type:", resp.headers.get("content-type"))
print(resp.text)
```

If you get HTML with a sign-in page, your token is not a valid OAuth token for
Databricks Apps.
