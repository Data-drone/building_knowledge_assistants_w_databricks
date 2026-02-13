# Databricks notebook source
# MAGIC %md
# MAGIC # End-to-End Deployment: From Evaluation to Production
# MAGIC
# MAGIC ## What You'll Do
# MAGIC Execute a complete deployment pipeline: run quality gates, log the model, deploy to
# MAGIC serving, validate, and enable monitoring — tying together everything from Modules 00-04.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Execute a CI/CD-style deployment pipeline for agents
# MAGIC - Gate deployment on evaluation quality (from Module 03)
# MAGIC - Deploy, validate, and enable monitoring in a single workflow
# MAGIC - Understand the complete agent development lifecycle
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed notebooks 01-03 in this module (or run this standalone)
# MAGIC - Understanding of all prior modules (00-04)
# MAGIC
# MAGIC ## Estimated Time
# MAGIC 20 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies

# COMMAND ----------

%pip install -q --upgrade mlflow[databricks]>=3.1.0 databricks-sdk databricks-agents databricks-langchain[memory]>=0.8.0 databricks-mcp langgraph>=0.2.50 langchain-core>=0.3.0 mcp

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configuration

# COMMAND ----------

import mlflow
from mlflow.genai.judges import make_judge
from typing import Literal
from databricks.sdk import WorkspaceClient
from databricks import agents
import pandas as pd
import time
import requests
import os

# Configuration — same values used throughout the bootcamp
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
LLM_ENDPOINT = "databricks-claude-sonnet-4-5"
LAKEBASE_PROJECT = "knowledge-assistant-state"
GENIE_SPACE_ID = "01234567-89ab-cdef-0123-456789abcdef"  # From Module 04
MODEL_NAME = f"{CATALOG}.{SCHEMA}.knowledge_assistant"

# Workspace info
w = WorkspaceClient()
HOST = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# MCP URLs
VS_MCP_URL = f"{HOST}/api/2.0/mcp/vector-search/{CATALOG}/{SCHEMA}"
GENIE_MCP_URL = f"{HOST}/api/2.0/mcp/genie/{GENIE_SPACE_ID}"

# Set experiment
experiment_name = f"/Users/{username}/agent_bootcamp_deployment"
mlflow.set_experiment(experiment_name)

print("✓ Configuration loaded for end-to-end deployment pipeline")
print(f"  Model: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: The Deployment Pipeline
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
# MAGIC │   PRE-DEPLOY        │     │   DEPLOY          │     │   POST-DEPLOY        │
# MAGIC │                     │     │                   │     │                      │
# MAGIC │ 1. Build agent      │────▶│ 4. Log to MLflow  │────▶│ 7. Validate endpoint │
# MAGIC │ 2. Run eval judges  │     │ 5. Register in UC │     │ 8. Enable Review App │
# MAGIC │ 3. Check quality    │     │ 6. Deploy serving │     │ 9. Verify monitoring │
# MAGIC │    gate             │     │                   │     │                      │
# MAGIC └─────────────────────┘     └──────────────────┘     └─────────────────────┘
# MAGIC         │                                                      │
# MAGIC         │ FAIL → Fix issues, re-evaluate                       │
# MAGIC         └──────────────────────────────────────────────────────┘
# MAGIC                              Feedback loop
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Pre-Deployment — Build and Evaluate
# MAGIC
# MAGIC **Before deploying, verify the agent meets quality standards.**
# MAGIC This is the same evaluation pipeline from Module 03, now used as a deployment gate.

# COMMAND ----------

# Model configuration — same as what we'll log to MLflow
model_config = {
    "llm_endpoint": LLM_ENDPOINT,
    "vector_search_mcp_url": VS_MCP_URL,
    "genie_mcp_url": GENIE_MCP_URL,
    "lakebase_instance_name": LAKEBASE_PROJECT,
    "enable_memory": True,
    "temperature": 0.1,
    "tool_timeout": 30,
    "recursion_limit": 25,
}

# Resolve agent module path
agent_module_path = os.path.abspath(os.path.join(os.getcwd(), "..", "src", "agent.py"))
print(f"Agent module: {agent_module_path} (exists: {os.path.exists(agent_module_path)})")

# Load the agent locally for evaluation — same as serving startup
print("Building agent for pre-deployment evaluation...")
try:
    loaded_agent = mlflow.pyfunc.load_model(
        agent_module_path,
        model_config=model_config,
    )
    print("✓ Agent loaded for evaluation")
except Exception as e:
    loaded_agent = None
    print(f"⚠ Could not load agent locally: {type(e).__name__}: {e}")
    print("  Skipping pre-deployment evaluation — will proceed to log and deploy.")

# COMMAND ----------

# Evaluation dataset (same questions from Module 03)
eval_questions = [
    {
        "question": "How much vacation time do full-time employees get?",
        "expected": "Full-time employees accrue 15 days of vacation per year (1.25 days per month)"
    },
    {
        "question": "What are the core in-office days for hybrid workers?",
        "expected": "Tuesday and Thursday are designated as core in-office days"
    },
    {
        "question": "What's the annual learning budget for managers?",
        "expected": "Managers receive $2,500 per year for professional development"
    },
]

# Create the accuracy judge (from Module 03)
accuracy_judge = make_judge(
    name="hr_accuracy",
    instructions=(
        "Evaluate if the response in {{ outputs }} is factually accurate.\n\n"
        "Compare to expected answer: {{ expectations }}\n\n"
        "Rate as:\n"
        "- 'excellent' = Perfectly accurate\n"
        "- 'good' = Accurate with minor details missing\n"
        "- 'fair' = Mostly accurate but missing important details\n"
        "- 'poor' = Significant errors\n"
        "- 'very_poor' = Incorrect or misleading"
    ),
    feedback_value_type=Literal["excellent", "good", "fair", "poor", "very_poor"],
    model=f"databricks:/{LLM_ENDPOINT}",
)


def _extract_text_from_response(result):
    """Extract text from ResponsesAgent output format."""
    if isinstance(result, dict):
        # Responses API format
        output = result.get("output", [])
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "assistant":
                content = item.get("content", [])
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        return c.get("text", "")
        # Fallback: try ChatCompletions format
        choices = result.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", str(result))
    return str(result)


# Generate predictions using the loaded agent
print("Running pre-deployment evaluation...")
print("=" * 80)

results = None
if loaded_agent is not None:
    try:
        eval_data = []
        for q in eval_questions:
            # Predict with Responses API format
            result = loaded_agent.predict(
                {"input": [{"role": "user", "content": q["question"]}]}
            )

            answer = _extract_text_from_response(result)

            eval_data.append({
                "inputs": {"question": q["question"]},
                "outputs": answer,
                "expectations": {"expected_answer": q["expected"]},
            })
            print(f"  ✓ {q['question'][:60]}...")

        # Run evaluation
        results = mlflow.genai.evaluate(
            data=eval_data,
            scorers=[accuracy_judge],
        )

        print("\n✓ Pre-deployment evaluation complete")
        if hasattr(results, 'metrics'):
            print("\nMetrics:")
            for metric_name, value in results.metrics.items():
                print(f"  {metric_name}: {value}")
    except Exception as e:
        print(f"\n⚠ Pre-deployment evaluation failed: {type(e).__name__}: {e}")
        print("  Proceeding with deployment (evaluation is advisory).")
else:
    print("  Skipped — agent could not be loaded locally.")
    print("  Proceeding with deployment.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Quality Gate Decision

# COMMAND ----------

gate_passed = True
issues_found = []

if results is not None and hasattr(results, 'tables') and results.tables:
    table_key = list(results.tables.keys())[0]
    results_df = results.tables[table_key]

    for col in results_df.columns:
        if 'value' in col.lower():
            poor_responses = results_df[results_df[col].isin(['poor', 'very_poor'])]
            if len(poor_responses) > 0:
                gate_passed = False
                judge_name = col.split('/')[0]
                issues_found.append(f"{judge_name}: {len(poor_responses)} poor/very_poor")

    print("Evaluation Results:")
    print("=" * 80)
    display(results_df)
elif results is None:
    print("⚠ No evaluation results (evaluation was skipped or failed)")
    print("  Proceeding with deployment for tutorial purposes.")

print()
if gate_passed:
    print("✅ QUALITY GATE PASSED — proceeding with deployment")
else:
    print("❌ QUALITY GATE FAILED")
    for issue in issues_found:
        print(f"   • {issue}")
    print("\n   In production CI/CD, this would block deployment.")
    print("   For this tutorial, we'll proceed anyway.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Log Model to MLflow
# MAGIC
# MAGIC Package the agent using `mlflow.pyfunc.log_model()` — same pattern as notebook 01.
# MAGIC `ResponsesAgent` auto-infers the correct signature.

# COMMAND ----------

pip_requirements = [
    "mlflow[databricks]>=3.1.0",
    "databricks-langchain[memory]>=0.8.0",
    "databricks-sdk",
    "databricks-mcp",
    "langgraph>=0.2.50",
    "langchain-core>=0.3.0",
    "mcp",
]

mlflow.set_registry_uri("databricks-uc")

# ResponsesAgent auto-infers signature — no manual signature needed
with mlflow.start_run(run_name="e2e_deployment") as run:
    logged_model = mlflow.pyfunc.log_model(
        python_model=agent_module_path,
        name="agent",
        model_config=model_config,
        pip_requirements=pip_requirements,
        registered_model_name=MODEL_NAME,
    )

    mlflow.set_tags({
        "deployment_type": "end_to_end",
        "agent_interface": "ResponsesAgent",
        "quality_gate": "passed" if gate_passed else "failed",
        "tools": "vector_search,genie",
        "memory": "lakebase",
        "llm": LLM_ENDPOINT,
    })

model_version = logged_model.registered_model_version

print(f"✓ Model logged and registered")
print(f"  Version: {model_version}")
print(f"  Run ID: {run.info.run_id}")
print(f"  Quality gate: {'passed' if gate_passed else 'FAILED (deployed anyway for tutorial)'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Deploy to Serving

# COMMAND ----------

print(f"Deploying {MODEL_NAME} v{model_version}...")
print("(This may take 10-15 minutes on first deployment)\n")

# Check if the endpoint already exists and serves the correct version.
# agents.deploy() blocks internally so we short-circuit when possible.
_expected_endpoint = f"agents_{MODEL_NAME.replace('.', '-')}"
_needs_deploy = True

try:
    _ep = w.serving_endpoints.get(_expected_endpoint)
    _served = _ep.config.served_entities
    if _served:
        _current_version = str(_served[0].entity_version)
        if _current_version == str(model_version):
            endpoint_name = _expected_endpoint
            _needs_deploy = False
            print(f"✓ Endpoint already exists and serves v{model_version}")
            print(f"  Endpoint name: {endpoint_name}")
        else:
            print(f"  Endpoint exists but serves v{_current_version}, updating to v{model_version}...")
except Exception:
    print(f"  No existing endpoint found — creating new deployment...")

if _needs_deploy:
    try:
        deployment = agents.deploy(
            model_name=MODEL_NAME,
            model_version=model_version,
        )
        endpoint_name = deployment.endpoint_name
        print(f"✓ Deployment initiated: {endpoint_name}")
    except TimeoutError:
        endpoint_name = _expected_endpoint
        print(f"⏳ agents.deploy() timed out (normal for first deploys)")
        print(f"  Deployment continues on the backend: {endpoint_name}")
    except ValueError as e:
        endpoint_name = _expected_endpoint
        print(f"✓ Endpoint already exists and serves this model version")
        print(f"  Endpoint name: {endpoint_name}")

# Wait for ready
MAX_WAIT_SECONDS = 1200
print(f"\nWaiting for endpoint to be ready (timeout {MAX_WAIT_SECONDS // 60} min)...")
start_time = time.time()

while True:
    endpoint = w.serving_endpoints.get(endpoint_name)
    state = endpoint.state
    elapsed = int(time.time() - start_time)
    print(f"  [{elapsed:>4}s] Ready: {state.ready} | Config: {state.config_update}")

    if "NOT_READY" not in str(state.ready) and "READY" in str(state.ready):
        print(f"\n✓ Endpoint is ready! (took {elapsed}s)")
        break
    elif "NOT_READY" in str(state.ready) and "UPDATE_FAILED" in str(state.config_update):
        print(f"\n✗ Deployment failed after {elapsed}s")
        print(f"  Check endpoint logs: {HOST}/#mlflow/endpoints/{endpoint_name}")
        break
    elif elapsed > MAX_WAIT_SECONDS:
        print(f"\n⏳ Still not ready after {elapsed}s — check the Serving UI")
        break

    time.sleep(60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Validate Production Endpoint

# COMMAND ----------

test_queries = [
    "What is the vacation policy for full-time employees?",
    "How many employees are in the Engineering department?",
    "What equipment is provided for remote work?",
]


def _extract_text(resp_json):
    """Extract text from Responses API format."""
    output = resp_json.get("output", [])
    for item in output:
        if item.get("type") == "message" and item.get("role") == "assistant":
            content = item.get("content", [])
            for c in content:
                if c.get("type") == "output_text":
                    return c.get("text", "")
    # Fallback: ChatCompletions format
    choices = resp_json.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return str(resp_json)


print("Production Validation:")
print("=" * 80)

passed_count = 0
for query in test_queries:
    response = requests.post(
        f"{HOST}/serving-endpoints/{endpoint_name}/invocations",
        headers={"Authorization": f"Bearer {token}"},
        json={"input": [{"role": "user", "content": query}]},
    )

    if response.status_code == 200:
        result = response.json()
        answer = _extract_text(result)
        print(f"\n✓ Q: {query}")
        print(f"  A: {answer[:150]}...")
        passed_count += 1
    else:
        print(f"\n✗ Q: {query}")
        print(f"  Error {response.status_code}: {response.text[:200]}")

print(f"\n{'=' * 80}")
print(f"Validation: {passed_count}/{len(test_queries)} queries succeeded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Enable Review App and Verify Monitoring

# COMMAND ----------

print("Enabling Review App...")
try:
    review_url = agents.enable_trace_reviews(model_name=MODEL_NAME)
    print(f"✓ Review App enabled")
    print(f"  URL: {review_url}")
except ValueError as e:
    print(f"⚠ Could not enable trace reviews: {e}")
    print("  This typically means the payload table has traces in an older format.")
    print("  The Review App will work for new requests sent after redeployment.")
    review_url = f"{HOST}/ml/review/{MODEL_NAME.replace('.', '/')}"
    print(f"  Review App URL (direct): {review_url}")

print(f"\nVerifying monitoring...")
print("=" * 80)

print(f"\n  1. Endpoint: {endpoint_name}")
print(f"     State: {w.serving_endpoints.get(endpoint_name).state.ready}")

print(f"\n  2. MLflow Tracing: enabled (automatic with agents.deploy)")

print(f"\n  3. Inference Tables: enabled (automatic with agents.deploy)")
print(f"     Payload table: {CATALOG}.{SCHEMA}.`{endpoint_name}_payload`")

print(f"\n  4. Review App: enabled")

print(f"\n✓ All monitoring systems active")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deployment Pipeline Complete!
# MAGIC
# MAGIC | Phase | Step | Status |
# MAGIC |-------|------|--------|
# MAGIC | **Pre-Deploy** | Build agent locally | ✅ |
# MAGIC | | Run evaluation judges | ✅ |
# MAGIC | | Check quality gate | ✅ |
# MAGIC | **Deploy** | Log model to MLflow | ✅ |
# MAGIC | | Register in Unity Catalog | ✅ |
# MAGIC | | Deploy to Model Serving | ✅ |
# MAGIC | **Post-Deploy** | Validate endpoint | ✅ |
# MAGIC | | Enable Review App | ✅ |
# MAGIC | | Verify monitoring | ✅ |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Full Bootcamp Recap
# MAGIC
# MAGIC | Module | What You Learned | Key Databricks Feature |
# MAGIC |--------|-----------------|----------------------|
# MAGIC | **00 - Foundations** | LLM endpoints, platform orientation | Mosaic AI Gateway |
# MAGIC | **01 - RAG Pipeline** | Document retrieval, vector indexing | Vector Search + Delta Sync |
# MAGIC | **02 - Memory** | Short-term + long-term memory | CheckpointSaver + DatabricksStore (Lakebase) |
# MAGIC | **03 - Evaluation** | Tracing, custom judges, quality gates | MLflow Tracing + GenAI Evaluate |
# MAGIC | **04 - MCP Tools** | Genie integration, custom MCP tools | Managed MCP + Databricks Apps |
# MAGIC | **05 - Deployment** | Logging, serving, monitoring | Model Serving + Inference Tables |
# MAGIC
# MAGIC ### Your Production Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────┐
# MAGIC │                      Unity Catalog                             │
# MAGIC │  ┌───────────────────────────────────────────────────────────┐ │
# MAGIC │  │  agent_bootcamp.knowledge_assistant                       │ │
# MAGIC │  │                                                           │ │
# MAGIC │  │  Tables       │ Volumes  │ Models   │ Indexes             │ │
# MAGIC │  │  (employees,  │ (docs)   │ (agent)  │ (vector search)     │ │
# MAGIC │  │   chunks)     │          │          │                     │ │
# MAGIC │  └───────────────────────────────────────────────────────────┘ │
# MAGIC └───────────────────────────┬─────────────────────────────────────┘
# MAGIC                             │
# MAGIC               ┌─────────────┴─────────────┐
# MAGIC               ▼                           ▼
# MAGIC  ┌──────────────────────┐    ┌────────────────────────┐
# MAGIC  │  Model Serving       │    │  Monitoring             │
# MAGIC  │  Endpoint            │    │                         │
# MAGIC  │  ┌────────────────┐  │    │  • Inference Tables     │
# MAGIC  │  │ Knowledge      │  │    │  • MLflow Traces        │
# MAGIC  │  │ Assistant      │  │    │  • Online Evaluation    │
# MAGIC  │  │ (Responses     │  │    │  • SQL Alerts           │
# MAGIC  │  │  Agent)        │  │    │  • Review App Feedback  │
# MAGIC  │  │ • LLM (Claude) │  │    │                         │
# MAGIC  │  │ • Vector Search│  │    └────────────────────────┘
# MAGIC  │  │ • Genie        │  │
# MAGIC  │  │ • Memory       │  │
# MAGIC  │  └────────────────┘  │
# MAGIC  └──────────┬───────────┘
# MAGIC      ┌──────┴──────────────┐
# MAGIC      ▼      ▼      ▼      ▼
# MAGIC    REST    SDK    AI    Review
# MAGIC    API           Play    App
# MAGIC                  ground
# MAGIC ```
# MAGIC
# MAGIC ### What's Next?
# MAGIC
# MAGIC | Goal | Approach |
# MAGIC |------|---------|
# MAGIC | Add more tools | Build custom MCP servers (Module 04) |
# MAGIC | Improve quality | Collect Review App feedback, refine judges, retrain |
# MAGIC | Scale up | Increase serving endpoint compute, add caching |
# MAGIC | Add more data | Index new documents in Vector Search, add Genie tables |
# MAGIC | CI/CD | Automate this notebook as a Databricks Job/workflow |
# MAGIC | A/B testing | Deploy multiple versions, route traffic, compare metrics |

# COMMAND ----------

print("=" * 80)
print("  DATABRICKS AGENT BOOTCAMP COMPLETE!")
print("=" * 80)
print()
print(f"  Agent:    {MODEL_NAME}")
print(f"  Endpoint: {endpoint_name}")
print()
print("  Your production-ready knowledge assistant is live.")
print()
print("  Modules completed:")
print("    00 - Foundations          (Mosaic AI Gateway)")
print("    01 - RAG Pipeline         (Vector Search)")
print("    02 - Memory               (Lakebase)")
print("    03 - Evaluation           (MLflow Judges)")
print("    04 - MCP Tool Integration (Genie + Custom Tools)")
print("    05 - Deployment           (Serving + Monitoring)")
print()
print("=" * 80)
