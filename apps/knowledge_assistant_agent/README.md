# Knowledge Assistant on Databricks Apps

This directory contains the first-pass Databricks Apps deployment for the bootcamp
agent. It exposes a Responses API-compatible `/invocations` endpoint using
`mlflow.genai.agent_server.AgentServer`.

## Deploy

```bash
DATABRICKS_USERNAME=$(databricks current-user me | jq -r .userName)
databricks sync . "/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
databricks apps create knowledge-assistant-agent-app
databricks apps deploy knowledge-assistant-agent-app \
  --source-code-path "/Workspace/Users/$DATABRICKS_USERNAME/knowledge_assistant_agent_app"
```

## Query

```bash
APP_URL=$(databricks apps get knowledge-assistant-agent-app -o json | jq -r '.url')
TOKEN=$(databricks auth token -o json | jq -r '.access_token')

curl -sS -X POST "$APP_URL/invocations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"What is Databricks MCP?"}]}' | jq .
```
