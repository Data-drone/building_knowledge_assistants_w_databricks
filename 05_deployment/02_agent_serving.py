# Databricks notebook source
# MAGIC %md
# MAGIC # Agent Deployment and the Review App
# MAGIC
# MAGIC ## What You'll Do
# MAGIC Deploy your logged agent to a Model Serving endpoint and enable the Review App
# MAGIC for stakeholder testing and human feedback collection.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Deploy an agent with `agents.deploy()`
# MAGIC - Understand what `agents.deploy()` does vs manual endpoint creation
# MAGIC - Test the production endpoint programmatically
# MAGIC - Enable the Review App for stakeholder feedback
# MAGIC - Access the agent from AI Playground, REST API, and SDK
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed [01_model_logging.py](01_model_logging.py) — model registered in Unity Catalog
# MAGIC
# MAGIC ## Estimated Time
# MAGIC 25 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies

# COMMAND ----------

%pip install -q --upgrade mlflow[databricks]>=3.1.0 databricks-sdk databricks-agents

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configuration

# COMMAND ----------

import mlflow
import os
import time
import requests

from databricks.sdk import WorkspaceClient
from databricks import agents

# Configuration
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.knowledge_assistant"

w = WorkspaceClient()
HOST = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Get the latest model version
versions = list(w.model_versions.list(full_name=MODEL_NAME))
if not versions:
    raise Exception(
        f"No model versions found for {MODEL_NAME}. "
        "Run 01_model_logging.py first to log and register the model."
    )

latest_version = max(versions, key=lambda v: int(v.version))
MODEL_VERSION = int(latest_version.version)

print(f"✓ Configuration loaded")
print(f"  Model: {MODEL_NAME}")
print(f"  Version: {MODEL_VERSION} (latest of {len(versions)})")
print(f"  Status: {latest_version.status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Understanding `agents.deploy()`
# MAGIC
# MAGIC | Feature | `agents.deploy()` | Manual `serving_endpoints.create()` |
# MAGIC |---------|-------------------|--------------------------------------|
# MAGIC | Endpoint creation | ✅ Automatic | Manual config with `ServedEntityInput` |
# MAGIC | Inference tables | ✅ Automatic | Must enable manually |
# MAGIC | MLflow tracing | ✅ Automatic | Must configure manually |
# MAGIC | Review App | ✅ One-line enable | Not available |
# MAGIC | Auth management | ✅ Managed credentials | Manual secret scopes |
# MAGIC | Scale-to-zero | ✅ Default | Must configure |
# MAGIC
# MAGIC `agents.deploy()` is a **high-level wrapper** that:
# MAGIC 1. Creates a Model Serving endpoint (or updates an existing one)
# MAGIC 2. Configures managed credentials so the agent can call MCP tools
# MAGIC 3. Enables inference tables for logging all requests/responses
# MAGIC 4. Enables MLflow tracing for internal execution visibility

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Deploy the Agent

# COMMAND ----------

print(f"Deploying {MODEL_NAME} v{MODEL_VERSION}...")
print("This creates a serving endpoint and enables inference tables.")
print("(First deployment may take 10-15 minutes to provision compute)\n")

# Check if the endpoint already exists and serves the correct version.
# agents.deploy() blocks internally and raises ValueError if the version
# is already deployed, so we short-circuit to avoid the long wait.
_expected_endpoint = f"agents_{MODEL_NAME.replace('.', '-')}"
_needs_deploy = True

try:
    _ep = w.serving_endpoints.get(_expected_endpoint)
    _served = _ep.config.served_entities
    if _served:
        _current_version = str(_served[0].entity_version)
        if _current_version == str(MODEL_VERSION):
            endpoint_name = _expected_endpoint
            _needs_deploy = False
            print(f"✓ Endpoint already exists and serves v{MODEL_VERSION}")
            print(f"  Endpoint name: {endpoint_name}")
        else:
            print(f"  Endpoint exists but serves v{_current_version}, updating to v{MODEL_VERSION}...")
except Exception:
    print(f"  No existing endpoint found — creating new deployment...")

if _needs_deploy:
    try:
        deployment = agents.deploy(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
        )
        endpoint_name = deployment.endpoint_name
        print(f"✓ Deployment complete")
        print(f"  Endpoint name: {endpoint_name}")
    except TimeoutError:
        endpoint_name = _expected_endpoint
        print(f"⏳ agents.deploy() timed out waiting for readiness (this is normal for first deploys)")
        print(f"  The deployment IS still running on the backend.")
        print(f"  Endpoint name: {endpoint_name}")
        print(f"  We'll poll for readiness in the next cell.")
    except ValueError as e:
        # Raised when the endpoint already serves this exact model version
        endpoint_name = _expected_endpoint
        print(f"✓ Endpoint already exists and serves this model version")
        print(f"  Endpoint name: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Wait for Endpoint to Be Ready
# MAGIC
# MAGIC The serving endpoint needs to:
# MAGIC 1. Provision compute resources
# MAGIC 2. Install the pip requirements from the logging notebook
# MAGIC 3. Execute `src/agent.py` — `ModelConfig()` reads config, `_build_graph()` runs
# MAGIC 4. Connect to MCP tools, Lakebase memory, etc.
# MAGIC
# MAGIC This is essentially the same startup sequence you tested locally with
# MAGIC `mlflow.pyfunc.load_model()` in the previous notebook, but on a dedicated container.

# COMMAND ----------

MAX_WAIT_SECONDS = 1200  # 20 minutes — generous for first deploy

print(f"Waiting for endpoint '{endpoint_name}' to be ready...")
print(f"(Checking every 60 seconds, timeout {MAX_WAIT_SECONDS // 60} min)\n")

start_time = time.time()

while True:
    endpoint = w.serving_endpoints.get(endpoint_name)
    state = endpoint.state
    elapsed = int(time.time() - start_time)
    print(f"  [{elapsed:>4}s] Ready: {state.ready} | Config update: {state.config_update}")

    if "NOT_READY" not in str(state.ready) and "READY" in str(state.ready):
        print(f"\n✓ Endpoint is ready! (took {elapsed}s)")
        break
    elif "NOT_READY" in str(state.ready) and "UPDATE_FAILED" in str(state.config_update):
        print(f"\n✗ Deployment failed after {elapsed}s")
        print("  Check the endpoint logs in the Serving UI:")
        print(f"  {HOST}/#mlflow/endpoints/{endpoint_name}")
        print("\n  Common failure causes:")
        print("    • Missing pip requirement (ModuleNotFoundError in container logs)")
        print("    • Invalid model_config values (MCP URL unreachable)")
        print("    • Lakebase instance not accessible from serving compute")
        break
    elif elapsed > MAX_WAIT_SECONDS:
        print(f"\n⏳ Still not ready after {elapsed}s — check the Serving UI:")
        print(f"  {HOST}/#mlflow/endpoints/{endpoint_name}")
        print("  The endpoint may still be provisioning. Re-run this cell to keep polling.")
        break

    time.sleep(60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Test the Production Endpoint
# MAGIC
# MAGIC The endpoint accepts the **Responses API** format. The `input` field is a list of
# MAGIC messages (same as OpenAI Responses API). This format is used by AI Playground,
# MAGIC the Databricks SDK, and the Review App.

# COMMAND ----------

test_queries = [
    {
        "query": "What is the vacation policy for full-time employees?",
        "expected_tool": "Vector Search (policy documents)",
    },
    {
        "query": "How many employees are in the Engineering department?",
        "expected_tool": "Genie (structured data query)",
    },
    {
        "query": "What equipment is provided for remote work?",
        "expected_tool": "Vector Search (policy documents)",
    },
]

print("Testing Production Endpoint:")
print("=" * 80)


def _extract_text(resp_json):
    """Extract text from Responses API format."""
    output = resp_json.get("output", [])
    for item in output:
        if item.get("type") == "message" and item.get("role") == "assistant":
            content = item.get("content", [])
            for c in content:
                if c.get("type") == "output_text":
                    return c.get("text", "")
    # Fallback: try ChatCompletions format for backward compatibility
    choices = resp_json.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return str(resp_json)


results = []
for test in test_queries:
    response = requests.post(
        f"{HOST}/serving-endpoints/{endpoint_name}/invocations",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "input": [{"role": "user", "content": test["query"]}]
        },
    )

    if response.status_code == 200:
        result = response.json()
        answer_preview = _extract_text(result)[:150]

        print(f"\n✓ Q: {test['query']}")
        print(f"  Expected tool: {test['expected_tool']}")
        print(f"  A: {answer_preview}...")
        results.append({"query": test["query"], "status": "success"})
    else:
        print(f"\n✗ Q: {test['query']}")
        print(f"  Error {response.status_code}: {response.text[:200]}")
        results.append({"query": test["query"], "status": "failed"})

passed = sum(1 for r in results if r["status"] == "success")
print(f"\n{'=' * 80}")
print(f"Results: {passed}/{len(results)} queries succeeded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Test Conversation Memory
# MAGIC
# MAGIC Since the agent has Lakebase CheckpointSaver enabled (Module 02), let's verify
# MAGIC multi-turn conversations work through the production endpoint.
# MAGIC
# MAGIC The `thread_id` is passed via `custom_inputs`, which the `ResponsesAgent` maps to
# MAGIC the LangGraph `configurable` dict for the `CheckpointSaver`.

# COMMAND ----------

thread_id = f"deploy-test-{username}"

print("Testing Multi-Turn Conversation:")
print("=" * 80)

# Turn 1
turn1_response = requests.post(
    f"{HOST}/serving-endpoints/{endpoint_name}/invocations",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "input": [{"role": "user", "content": "I work in the Engineering department. What training is available?"}],
        "custom_inputs": {"thread_id": thread_id}
    },
)

if turn1_response.status_code == 200:
    turn1_answer = _extract_text(turn1_response.json())
    print(f"Turn 1 - User: I work in Engineering. What training is available?")
    print(f"Turn 1 - Agent: {turn1_answer[:200]}...")
else:
    print(f"Turn 1 failed: {turn1_response.status_code}")
    print(f"  {turn1_response.text[:200]}")

# Turn 2: Follow-up that requires context from Turn 1
print()
turn2_response = requests.post(
    f"{HOST}/serving-endpoints/{endpoint_name}/invocations",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "input": [{"role": "user", "content": "How do I sign up for those?"}],
        "custom_inputs": {"thread_id": thread_id}
    },
)

if turn2_response.status_code == 200:
    turn2_answer = _extract_text(turn2_response.json())
    print(f"Turn 2 - User: How do I sign up for those?")
    print(f"Turn 2 - Agent: {turn2_answer[:200]}...")
    print(f"\n→ If the agent references training/Engineering, memory is working!")
else:
    print(f"Turn 2 failed: {turn2_response.status_code}")
    print(f"  {turn2_response.text[:200]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Enable the Review App
# MAGIC
# MAGIC The **Review App** provides a chat interface where stakeholders can:
# MAGIC - Test the agent with their own questions
# MAGIC - Rate responses with thumbs up/down
# MAGIC - Leave detailed written feedback
# MAGIC - Flag incorrect or inappropriate responses
# MAGIC
# MAGIC This feedback is stored in **assessment tables** (part of inference tables)
# MAGIC and feeds directly into the evaluation pipeline you built in Module 03.

# COMMAND ----------

try:
    review_url = agents.enable_trace_reviews(model_name=MODEL_NAME)
    print("✓ Review App enabled!")
    print()
    print(f"  Review App URL: {review_url}")
except ValueError as e:
    # Older traces in the payload table can cause format errors.
    # The Review App still works for new traces going forward.
    print(f"⚠ Could not enable trace reviews: {e}")
    print("  This typically means the payload table has traces in an older format.")
    print("  The Review App will work for new requests sent after redeployment.")
    review_url = f"{HOST}/ml/review/{MODEL_NAME.replace('.', '/')}"
    print(f"  Review App URL (direct): {review_url}")

print()
print("  Share this URL with stakeholders for testing.")
print("  All feedback is automatically logged to assessment tables.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Serving Architecture Overview
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    Consumers                                │
# MAGIC │   AI Playground  │  Review App  │  REST API  │  Your App   │
# MAGIC └───────┬──────────┬─────────────┬────────────┬──────────────┘
# MAGIC         │          │             │            │
# MAGIC         ▼          ▼             ▼            ▼
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                Model Serving Endpoint                       │
# MAGIC │  ┌───────────────────────────────────────────────────────┐  │
# MAGIC │  │      ResponsesAgent (from src/agent.py)               │  │
# MAGIC │  │  ┌────────────────────────────────────────────────┐   │  │
# MAGIC │  │  │          LangGraph (same code as notebooks)    │   │  │
# MAGIC │  │  │  ┌──────────┐  ┌────────────┐  ┌───────────┐  │   │  │
# MAGIC │  │  │  │   LLM    │  │ MCP Tools  │  │  Memory   │  │   │  │
# MAGIC │  │  │  │ (Claude) │  │ (VS,Genie) │  │(Lakebase) │  │   │  │
# MAGIC │  │  │  └──────────┘  └────────────┘  └───────────┘  │   │  │
# MAGIC │  │  └────────────────────────────────────────────────┘   │  │
# MAGIC │  └───────────────────────────────────────────────────────┘  │
# MAGIC │                                                             │
# MAGIC │  Inference Tables: requests/responses logged automatically  │
# MAGIC │  MLflow Tracing: internal spans captured automatically      │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: All Access Points

# COMMAND ----------

print("Your Agent is Live!")
print("=" * 80)

print(f"\n1. REST API (Responses format):")
print(f"   POST {HOST}/serving-endpoints/{endpoint_name}/invocations")
print(f"   Headers: Authorization: Bearer <token>")
print(f'   Body: {{"input": [{{"role": "user", "content": "..."}}]}}')

print(f"\n2. AI Playground:")
print(f"   Navigate to: Workspace → Machine Learning → Playground")
print(f"   Select endpoint: {endpoint_name}")

print(f"\n3. Review App:")
print(f"   {review_url}")

print(f"\n4. Databricks SDK:")
print(f"   from databricks.sdk import WorkspaceClient")
print(f"   w = WorkspaceClient()")
print(f"   response = w.serving_endpoints.query(")
print(f"       name='{endpoint_name}',")
print(f'       input=[{{"role": "user", "content": "..."}}]')
print(f"   )")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What We Did
# MAGIC - ✅ Deployed agent with `agents.deploy()` (endpoint + inference tables + tracing)
# MAGIC - ✅ Monitored endpoint startup and verified it reached READY state
# MAGIC - ✅ Tested single-turn queries covering Vector Search and Genie tools
# MAGIC - ✅ Tested multi-turn conversation to verify memory works in production
# MAGIC - ✅ Enabled Review App for stakeholder testing and feedback collection
# MAGIC - ✅ Identified all access points (REST, Playground, Review App, SDK)
# MAGIC
# MAGIC ### Key Concepts
# MAGIC
# MAGIC **`agents.deploy()` does 4 things in one call:**
# MAGIC 1. Creates/updates a Model Serving endpoint
# MAGIC 2. Configures managed authentication for MCP tools
# MAGIC 3. Enables inference tables (request/response logging)
# MAGIC 4. Enables MLflow tracing (internal execution spans)
# MAGIC
# MAGIC **Request format (Responses API):**
# MAGIC ```python
# MAGIC response = requests.post(
# MAGIC     f"{HOST}/serving-endpoints/{endpoint_name}/invocations",
# MAGIC     json={
# MAGIC         "input": [{"role": "user", "content": "..."}],
# MAGIC         "custom_inputs": {"thread_id": "abc123"}  # optional, for memory
# MAGIC     }
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC Continue to [03_production_monitoring.py](03_production_monitoring.py) to set up
# MAGIC production monitoring, online evaluation, and quality alerts.

# COMMAND ----------

print("✓ Deployment complete!")
print(f"\n  Endpoint: {endpoint_name}")
print(f"  Review App: enabled")
print("\nNext: Set up production monitoring and online evaluation")
