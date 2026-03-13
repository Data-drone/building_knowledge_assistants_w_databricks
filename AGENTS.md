## Purpose
This repo contains examples and apps for building Databricks agents and memory-enabled assistants.
Agents should optimize for clarity, reproducibility, and safe Databricks operations.

## Working Style
- Make minimal, targeted changes.
- Prefer small diffs over broad refactors.
- Explain why a change is made, not only what changed.
- Keep examples beginner-friendly and instructional.
- Work primarily in Databricks Notebooks

## Python Standards
- Minimise use of functions.
- Add type hints where practical.
- Use clear variable names for teaching value.
- Avoid introducing new dependencies unless necessary.
- use `%pip install` in notebooks to install libraries followed by `%restart_python`

## Deployment
- use `databricks bundle deploy` command to deploy
## Diagnose `databricks bundle deploy` failures (e.g., RUN_EXECUTION_ERROR)

# 1) Deploy with debug + log file (captures job/run/pipeline IDs)
databricks bundle deploy -t <target> --debug --log-file /tmp/bundle.log

# 2) Extract identifiers + failure hint from the log
grep -E "RUN_EXECUTION_ERROR|run_id|job_id|pipeline_id" /tmp/bundle.log | tail -n 50

# 3) If you have a run_id, pull the real error/stacktrace
databricks jobs get-run-output <RUN_ID> --output json
databricks jobs get-run <RUN_ID> --output json

# 4) If get-run-output is thin (non-notebook / spark-submit / etc.)
# Open the failed task in the Jobs UI and read driver stdout/stderr + event logs.

## File Hygiene
- Do not commit generated artifacts (`__pycache__`, `.pyc`, local env files).
- Keep `.gitignore` updated for local/dev-only files.
- Preserve existing user changes in unrelated files.

## Safety
- Never expose or commit secrets/tokens.
- Redact sensitive values in logs, examples, and docs.

## Validation
After meaningful code edits:
- Run relevant checks/tests for touched code on Databricks via Databricks CLI
- Verify examples execute as documented.
- Note any limitations or unverified paths in the final response.

## Documentation
When behavior changes:
- Update nearby docs or README sections.
- Keep instructions copy-paste ready.