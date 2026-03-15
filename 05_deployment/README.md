# Deployment - Production on Databricks Apps

Deploy your agent to production with monitoring in under 45 minutes!

## What You'll Learn

This module teaches you to deploy and monitor your agent on Databricks Apps:
1. **Notebook 01** (10 min): Apps Dev Loop - Day-to-day development workflow
2. **Notebook 02** (15 min): Apps Deployment - Deploy and validate `/invocations`
3. **Notebook 03** (10 min): Production Monitoring - MLflow traces + online evaluation
4. **Notebook 04** (10 min): End-to-End Deployment - Full release pipeline

## Prerequisites

### Required
- **All prior modules completed** (or at minimum Modules 00–02)
- **App source code** present under `apps/knowledge_assistant_agent`
- Databricks CLI configured
- OAuth token for testing

### Recommended
- Completed [Module 03 (Evaluation)](../03_evaluation/README.md) for understanding scorers and judges
- Completed [Module 04 (MCP Tools)](../04_mcp_tool_integration/README.md) for the full multi-tool agent

## Learning Path

### Start Here: 01_apps_dev_loop.py

**What you'll build**: Inner development loop for iterating on a deployed agent

**Key concepts**:
- App source layout and runtime contract
- Syncing local changes to workspace files
- Redeploying and verifying `/invocations`
- Passing `thread_id` and `user_id` for memory
- OAuth token usage for notebook testing

**Time**: 10 minutes

---

### Next: 02_apps_deployment.py

**What you'll build**: Repeatable deployment validation workflow

**Key concepts**:
- Deploying app source code to Databricks Apps
- Validating endpoint health and response schema
- Validating short-term memory (`thread_id`) and long-term memory (`user_id`)
- Production-style API checks from notebook and terminal

**Time**: 15 minutes

---

### Next: 03_production_monitoring.py

**What you'll build**: Production monitoring pipeline with scorers

**Key concepts**:
- Verifying app health through traces
- Creating scorers: built-in LLM judges, custom LLM judges, code-based scorers
- Batch evaluation with `mlflow.genai.evaluate()`
- Registering scorers for continuous production monitoring with sampling
- Managing production scorers: list, update, stop, delete

**Time**: 10 minutes

---

### Next: 04_end_to_end_deployment.py

**What you'll build**: Release-ready deployment checklist

**Key concepts**:
- Quality gate checks before deployment
- Sync + deploy app source
- Validate `/invocations` end-to-end
- Confirm monitoring signals in MLflow

**Time**: 10 minutes

---

## What's Next?

After completing these notebooks, you'll understand:
- ✅ How to deploy agents as Databricks Apps
- ✅ Day-to-day development loop for iterating
- ✅ Endpoint validation and health checks
- ✅ Production monitoring with scorers and judges
- ✅ End-to-end release pipeline

---

## Quick Start

1. Open `01_apps_dev_loop.py`
2. Set up the development loop (sync, deploy, verify)
3. Continue to `02_apps_deployment.py` for deployment validation
4. Add monitoring in `03_production_monitoring.py`
5. Run the full pipeline in `04_end_to_end_deployment.py`

---

## Technical Stack

- **LLM**: Databricks LLM endpoints (Claude Sonnet 4.6)
- **Deployment**: Databricks Apps
- **Monitoring**: MLflow Tracing + online evaluation
- **Evaluation**: MLflow Evaluate with scorers and judges
- **Memory**: Lakebase (autoscaling project/branch)
- **App Runtime**: `mlflow.genai.agent_server.AgentServer`

---

## App Source

The deployment artifact is the app source folder at
[`apps/knowledge_assistant_agent`](../apps/knowledge_assistant_agent/). See its
[README](../apps/knowledge_assistant_agent/README.md) for the request contract,
deploy commands, Lakebase permissions, and query examples.

---

## Need Help?

- Check MLflow traces for debugging request flow
- Verify OAuth token is valid (HTML sign-in page means expired/invalid token)
- Ensure Lakebase permissions are granted to the app service principal
- See the [apps README](../apps/knowledge_assistant_agent/README.md) for Lakebase permission SQL

---

**Total Time**: 45 minutes from local agent to production deployment 🚀
