# Troubleshooting

## Vector Search

### Index not syncing

- Check that the source Delta table has data
- Verify the embedding endpoint is available and `ONLINE`
- Wait for index state to reach `ONLINE` (initial sync can take a few minutes)
- Check Unity Catalog permissions on the source table

## Lakebase

### Connection timeout

- First connection after idle has a 20–30s cold start (autoscaling)
- Use connection pooling (`min_size >= 2`) to keep the connection warm
- Verify your Lakebase project and branch exist

### CheckpointSaver setup fails on Databricks Apps

- For autoscaling Lakebase, configure both `LAKEBASE_AUTOSCALING_PROJECT` and `LAKEBASE_AUTOSCALING_BRANCH`
- Ensure the app service principal has `USAGE, CREATE` on schema `public`
- Ensure checkpoint tables in `public` grant `SELECT, INSERT, UPDATE` to the app role

### DatabricksStore setup fails on Databricks Apps

- Use the same Lakebase autoscaling project/branch settings as short-term memory
- Ensure the app role can create and update:
    - `public.store`
    - `public.store_vectors`
    - `public.store_migrations`
    - `public.vector_migrations`
- Verify your embedding endpoint and dimensions match the store configuration

## MCP Tools

### Tools not found

- Verify MCP URL format is correct for your workspace
- Check Unity Catalog permissions on underlying assets
- Ensure authentication token has required scopes
- Confirm the Genie Space ID is set correctly (Module 04)

### Agent not using tools

- Check that tools are bound to the LLM: `llm.bind_tools(tools)`
- Verify tool descriptions are clear and specific
- Review MLflow traces to see the LLM's reasoning

## Evaluation

### Traces do not appear

- Call `mlflow.tracing.enable()` before generating answers
- Verify the experiment name is correct in `mlflow.set_experiment()`

### Built-in scorers look fine but answers are wrong

- Add domain-specific judges with `make_judge()` for factual accuracy
- Check if the issue is retrieval (wrong documents) or generation (wrong answer from right documents) using trace inspection

### Custom judge scores don't match expert review

- Collect 10–20 human ratings
- Tighten judge instructions around the disagreement patterns
- Consider using `feedback_value_type` with more granular categories

### Batch evaluation is slow

- Start with built-in scorers (fast and validated)
- Keep the eval dataset small while iterating
- Add custom LLM judges only for the gaps that matter

## Deployment

### OAuth token issues

- If you get an HTML login page instead of JSON, the token is expired or invalid
- Regenerate the OAuth token and update `my-secrets/apps_oauth_token`
- Check `allow_redirects=False` is set on requests to catch redirects

### App not responding

- Check `databricks apps get <app-name>` for status
- Verify the app source code was synced correctly
- Check app logs for startup errors
- Ensure all required environment variables are set in `app.yaml`
