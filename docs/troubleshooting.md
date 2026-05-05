# Troubleshooting

Practical runbook for common issues.

---

## Setup issues

??? question "Catalog/schema creation fails"
    - Verify you have `CREATE CATALOG` or `CREATE SCHEMA` permissions
    - Check if the catalog already exists: `SHOW CATALOGS`
    - Metastore must be assigned to your workspace

??? question "Lakebase project creation fails"
    - Confirm Lakebase is enabled for your workspace
    - Check region matches your workspace region in `config.py`
    - Verify your identity has Lakebase project creation rights

??? question "Endpoint not available"
    - Check endpoint status: Workspace → Serving → Endpoints
    - Must show `ONLINE` (not `PROVISIONING` or `FAILED`)
    - Some endpoints take a few minutes to provision on first use

---

## Vector Search

??? question "Index not syncing / stuck in PROVISIONING"
    - Source Delta table must have data (at least one row)
    - Embedding endpoint must be `ONLINE`
    - Wait up to 5 minutes for initial sync
    - Check UC permissions on source table and target index

??? question "Search returns no results"
    - Verify documents were chunked and inserted into the source table
    - Check that the embedding column matches what the index expects
    - Test with a very generic query to rule out relevance issues

---

## Lakebase & memory

??? question "Connection timeout on first call"
    Autoscaling Lakebase has a 20–30s cold start. This is normal for the first request after idle. Subsequent calls are fast.

??? question "CheckpointSaver setup fails on Apps"
    - Configure both `LAKEBASE_AUTOSCALING_PROJECT` and `LAKEBASE_AUTOSCALING_BRANCH` in `app.yaml`
    - App service principal needs `USAGE, CREATE` on schema `public`
    - Checkpoint tables need `SELECT, INSERT, UPDATE` granted to the app role

??? question "DatabricksStore setup fails on Apps"
    - Same Lakebase config as CheckpointSaver
    - App role needs create/update on: `public.store`, `public.store_vectors`, `public.store_migrations`, `public.vector_migrations`
    - Verify embedding endpoint and dimension count match store config

---

## MCP tools

??? question "Tools not found / agent doesn't call tools"
    - Verify MCP URL format: `{HOST}/api/2.0/mcp/vector-search/{CATALOG}/{SCHEMA}`
    - Check UC permissions on underlying assets
    - Confirm Genie Space ID is correct (from Module 04 setup)
    - Ensure tools are bound to the LLM: `llm.bind_tools(tools)`

??? question "Genie returns errors"
    - Tables referenced in the Genie Space must exist and be accessible
    - User must have `SELECT` on the underlying tables
    - Try the same question in the Genie UI to isolate agent vs Genie issues

---

## Evaluation

??? question "Traces don't appear"
    - Call `mlflow.tracing.enable()` before agent calls
    - Verify experiment name: `mlflow.set_experiment("...")`
    - Check that autologging is enabled: `mlflow.langchain.autolog()`

??? question "Scorers return unexpected results"
    - Start with built-in scorers to verify the pipeline works
    - For custom judges: tighten instructions around the specific mismatch
    - Collect 10–20 human ratings to calibrate judge accuracy

---

## Deployment

??? question "App returns HTML login page instead of JSON"
    OAuth token expired or invalid. Regenerate and update in secrets. Set `allow_redirects=False` on requests to catch this earlier.

??? question "App deploy succeeds but endpoint fails"
    - Check app logs: `databricks apps get <app-name>`
    - Verify all env vars are set in `app.yaml`
    - Check `requirements.txt` versions match MLR packages
    - Ensure source code was synced: `databricks sync "apps/..." "/Workspace/..."`

??? question "`databricks sync` hangs or fails"
    - Confirm CLI is authenticated: `databricks auth login`
    - Target path must start with `/Workspace/Users/`
    - Check network connectivity to workspace

---

## Deploy commands reference

```bash
# Sync source code
databricks sync "apps/knowledge_assistant_agent" \
  "/Users/$USERNAME/knowledge_assistant_agent_app"

# Deploy app
databricks apps deploy knowledge-assistant-agent-app \
  --source-code-path "/Workspace/Users/$USERNAME/knowledge_assistant_agent_app"

# Check app status
databricks apps get knowledge-assistant-agent-app

# View app logs
databricks apps get-logs knowledge-assistant-agent-app
```

---

## Where logs live

| Component | Location |
|---|---|
| App runtime | `databricks apps get-logs <app-name>` |
| MLflow traces | Experiment specified in `APP_EXPERIMENT` |
| Scorer results | MLflow evaluation tab in experiment |
| Agent decisions | Trace spans → LLM calls → tool calls |
| Lakebase | Workspace Lakebase console |
