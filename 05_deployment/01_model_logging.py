# Databricks notebook source
# MAGIC %md
# MAGIC # From Notebook to Production: Logging Your Agent
# MAGIC
# MAGIC ## What You'll Do
# MAGIC Package the agent you've built across Modules 00-04 into a deployable MLflow model,
# MAGIC registered in Unity Catalog and ready for serving.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Understand models-from-code: how MLflow packages LangGraph agents
# MAGIC - Define model configuration and dependencies
# MAGIC - Log your agent to MLflow and register in Unity Catalog
# MAGIC - Validate the logged model locally before deployment
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed Modules 00-04
# MAGIC - Vector Search index, Genie space, and Lakebase instance created
# MAGIC
# MAGIC ## Estimated Time
# MAGIC 25 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies

# COMMAND ----------

%pip install -q --upgrade mlflow[databricks]>=3.1.0 databricks-sdk databricks-langchain[memory]>=0.8.0 databricks-mcp langgraph>=0.2.50 langchain-core>=0.3.0 mcp

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configuration

# COMMAND ----------

import mlflow
import os

# Configuration — same values from earlier modules
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
LLM_ENDPOINT = "databricks-claude-sonnet-4-5"
LAKEBASE_PROJECT = "knowledge-assistant-state"
GENIE_SPACE_ID = "01234567-89ab-cdef-0123-456789abcdef"  # From Module 04

# Workspace info
HOST = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# Model registry name (3-level Unity Catalog path)
MODEL_NAME = f"{CATALOG}.{SCHEMA}.knowledge_assistant"

# MCP endpoint URLs (same pattern as Module 04)
VS_MCP_URL = f"{HOST}/api/2.0/mcp/vector-search/{CATALOG}/{SCHEMA}"
GENIE_MCP_URL = f"{HOST}/api/2.0/mcp/genie/{GENIE_SPACE_ID}"

# Set experiment
experiment_name = f"/Users/{username}/agent_bootcamp_deployment"
mlflow.set_experiment(experiment_name)

print(f"✓ Configuration loaded")
print(f"  Model name: {MODEL_NAME}")
print(f"  Experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: The Notebook-to-Production Gap
# MAGIC
# MAGIC Throughout Modules 00-04, you built an agent **interactively** in notebooks:
# MAGIC
# MAGIC | Module | What You Built |
# MAGIC |--------|----------------|
# MAGIC | 00 - Foundations | LLM calls via Mosaic AI Gateway |
# MAGIC | 01 - RAG Pipeline | Vector Search tool + LangGraph agent |
# MAGIC | 02 - Memory | CheckpointSaver + DatabricksStore |
# MAGIC | 03 - Evaluation | Custom judges + quality gates |
# MAGIC | 04 - MCP Tools | Genie integration + custom tools |
# MAGIC
# MAGIC To **deploy** this agent, you need to package it so that:
# MAGIC 1. MLflow can **serialize** it (save to disk)
# MAGIC 2. Model Serving can **reconstruct** it (load from disk on a fresh container)
# MAGIC 3. All tools, memory, and config are available at serving time
# MAGIC
# MAGIC MLflow's **models-from-code** pattern solves this: you put your LangGraph agent
# MAGIC in a Python file, MLflow stores the file as text, and rebuilds the graph at
# MAGIC serving time from a config dict.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Understanding Models-from-Code
# MAGIC
# MAGIC The pattern is simple — your agent module is a **plain Python script** that:
# MAGIC 1. Reads config via `mlflow.models.ModelConfig()`
# MAGIC 2. Builds the LangGraph agent (same code as the notebooks)
# MAGIC 3. Wraps it in a `ResponsesAgent` (thin serving wrapper)
# MAGIC 4. Calls `mlflow.models.set_model(agent)` to register it
# MAGIC
# MAGIC ```python
# MAGIC # agent.py — this is just the notebook code in a file
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest, ResponsesAgentResponse,
# MAGIC     output_to_responses_items_stream, to_chat_completions_input,
# MAGIC )
# MAGIC from databricks_langchain import ChatDatabricks, CheckpointSaver
# MAGIC from langgraph.graph import StateGraph, MessagesState, END
# MAGIC from langgraph.prebuilt import ToolNode
# MAGIC
# MAGIC config = mlflow.models.ModelConfig()
# MAGIC
# MAGIC # --- Same LangGraph code from notebooks ---
# MAGIC llm = ChatDatabricks(endpoint=config.get("llm_endpoint"))
# MAGIC llm_with_tools = llm.bind_tools(tools)
# MAGIC workflow = StateGraph(MessagesState)
# MAGIC # ... same graph wiring ...
# MAGIC graph = workflow.compile(checkpointer=checkpointer)
# MAGIC
# MAGIC # --- Thin wrapper for serving ---
# MAGIC class KnowledgeAssistant(ResponsesAgent):
# MAGIC     def predict(self, request):
# MAGIC         msgs = to_chat_completions_input([i.model_dump() for i in request.input])
# MAGIC         result = graph.stream({"messages": msgs}, ...)
# MAGIC         yield from output_to_responses_items_stream(result_messages)
# MAGIC
# MAGIC agent = KnowledgeAssistant()
# MAGIC mlflow.models.set_model(agent)
# MAGIC ```
# MAGIC
# MAGIC **Why this works:**
# MAGIC
# MAGIC | Concern | How It's Handled |
# MAGIC |---------|-----------------|
# MAGIC | Serialization | Module file stored as text, not pickled |
# MAGIC | Configuration | `model_config` dict stored alongside model |
# MAGIC | Dependencies | `pip_requirements` installed on serving container |
# MAGIC | Authentication | Serving endpoint gets its own managed credentials |
# MAGIC | Signature | `ResponsesAgent` **auto-infers** the correct signature — compatible with `agents.deploy()` |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Examine the Agent Module
# MAGIC
# MAGIC The `src/agent.py` file contains:
# MAGIC 1. The **same LangGraph code** from the notebooks (`_build_graph()`)
# MAGIC 2. A thin `ResponsesAgent` wrapper that converts between the Responses API
# MAGIC    format and LangGraph's `MessagesState` format using MLflow helper functions
# MAGIC
# MAGIC ```
# MAGIC src/agent.py
# MAGIC ├── config = mlflow.models.ModelConfig()    ← Read config at load time
# MAGIC ├── _build_graph(config)                    ← Same code from Modules 01-04:
# MAGIC │   ├── llm = ChatDatabricks(...)           ←   Module 00
# MAGIC │   ├── tools = load_mcp_tools(...)         ←   Module 04
# MAGIC │   ├── checkpointer = CheckpointSaver(...) ←   Module 02
# MAGIC │   ├── call_agent(state: MessagesState)    ←   Same as every notebook
# MAGIC │   ├── should_continue(state)              ←   Same as every notebook
# MAGIC │   └── graph = workflow.compile(...)        ←   Same as every notebook
# MAGIC │
# MAGIC ├── class KnowledgeAssistant(ResponsesAgent):
# MAGIC │   ├── predict(request) → ResponsesAgentResponse
# MAGIC │   └── predict_stream(request) → yields ResponsesAgentStreamEvent
# MAGIC │       ├── to_chat_completions_input()     ← MLflow: Responses → messages
# MAGIC │       ├── graph.stream({"messages":...})  ← Run through LangGraph
# MAGIC │       └── output_to_responses_items_stream() ← MLflow: messages → Responses
# MAGIC │
# MAGIC └── mlflow.models.set_model(agent)          ← Register instance for serving
# MAGIC ```
# MAGIC
# MAGIC **You'll recognise every line** — the only new code is the `ResponsesAgent` wrapper
# MAGIC and MLflow's conversion helpers (`to_chat_completions_input` and
# MAGIC `output_to_responses_items_stream`).

# COMMAND ----------

# Let's verify the agent module exists
agent_path = os.path.join(os.getcwd(), "..", "src", "agent.py")
abs_agent_path = os.path.abspath(agent_path)

print(f"Agent module path: {abs_agent_path}")
print(f"Exists: {os.path.exists(abs_agent_path)}")

if os.path.exists(abs_agent_path):
    with open(abs_agent_path) as f:
        lines = f.readlines()

    print(f"Lines: {len(lines)}")
    print()
    print("Key patterns (matching notebooks):")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if any(kw in stripped for kw in [
            "ChatDatabricks(", "StateGraph(", "set_entry_point",
            "add_conditional_edges", "compile(", "set_model(",
            "call_agent", "should_continue", "ToolNode(",
            "CheckpointSaver(", "ModelConfig()",
            "ResponsesAgent", "to_chat_completions_input",
            "output_to_responses_items_stream",
        ]):
            print(f"  Line {i:>3}: {stripped}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Define Model Configuration
# MAGIC
# MAGIC The configuration dict is stored alongside the model in MLflow. When the serving
# MAGIC endpoint loads the agent module, `mlflow.models.ModelConfig()` reads these values.
# MAGIC
# MAGIC **Important:** Don't put secrets here — authentication is handled by Databricks'
# MAGIC managed credentials (OAuth) at serving time.

# COMMAND ----------

model_config = {
    # LLM configuration (Module 00)
    "llm_endpoint": LLM_ENDPOINT,
    "temperature": 0.1,

    # MCP tool URLs (Modules 01, 04)
    "vector_search_mcp_url": VS_MCP_URL,
    "genie_mcp_url": GENIE_MCP_URL,

    # Memory configuration (Module 02)
    "lakebase_instance_name": LAKEBASE_PROJECT,
    "enable_memory": True,

    # Agent behavior
    "tool_timeout": 30,
    "recursion_limit": 25,
}

print("Model Configuration:")
print("=" * 80)
for key, value in model_config.items():
    display_val = (str(value)[:60] + "...") if len(str(value)) > 60 else value
    print(f"  {key}: {display_val}")

print(f"\nThis config maps to your work across modules:")
print(f"  llm_endpoint          → Module 00 (Mosaic AI Gateway)")
print(f"  vector_search_mcp_url → Module 01 (RAG Pipeline)")
print(f"  lakebase_instance_name → Module 02 (Memory)")
print(f"  genie_mcp_url          → Module 04 (Genie Integration)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Define Pip Requirements
# MAGIC
# MAGIC The serving endpoint runs on a **fresh container** — it doesn't have your notebook's
# MAGIC packages. You must explicitly list every dependency your agent needs.

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

print("Pip Requirements for Serving Endpoint:")
print("=" * 80)
for req in pip_requirements:
    print(f"  • {req}")
print(f"\n  Total: {len(pip_requirements)} packages")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Log the Model to MLflow
# MAGIC
# MAGIC We use `mlflow.pyfunc.log_model()` with models-from-code:
# MAGIC - `python_model` points to the agent module containing the `ResponsesAgent` subclass
# MAGIC - `ResponsesAgent` **automatically infers the correct signature** — no manual
# MAGIC   signature or input_example needed
# MAGIC - `model_config` is stored and read by `mlflow.models.ModelConfig()` at serving time
# MAGIC - Compatible with `agents.deploy()`, AI Playground, and the Review App

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# Resolve the agent module path
agent_module_path = os.path.abspath(os.path.join(os.getcwd(), "..", "src", "agent.py"))
print(f"Agent module: {agent_module_path}")
print(f"Exists: {os.path.exists(agent_module_path)}")

# COMMAND ----------

# ResponsesAgent auto-infers the correct signature — no manual signature needed.
# This eliminates the schema compatibility issues with agents.deploy().
with mlflow.start_run(run_name="knowledge_assistant_v1") as run:
    logged_model = mlflow.pyfunc.log_model(
        python_model=agent_module_path,
        name="agent",
        model_config=model_config,
        pip_requirements=pip_requirements,
        registered_model_name=MODEL_NAME,
    )

    mlflow.set_tags({
        "agent_type": "knowledge_assistant",
        "agent_interface": "ResponsesAgent",
        "tools": "vector_search,genie",
        "memory": "lakebase_checkpointer",
        "llm": LLM_ENDPOINT,
    })

print(f"✓ Model logged successfully")
print(f"  Run ID: {run.info.run_id}")
print(f"  Model URI: {logged_model.model_uri}")
print(f"  Registry: {MODEL_NAME}")
if hasattr(logged_model, 'registered_model_version'):
    print(f"  Version: {logged_model.registered_model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Validate the Logged Model
# MAGIC
# MAGIC **Always validate before deploying.** Load the model back from MLflow and test it
# MAGIC locally. This catches configuration errors *before* they reach the serving endpoint.
# MAGIC
# MAGIC Note: `ResponsesAgent` uses the Responses API format — `input` (not `messages`).

# COMMAND ----------

print("Loading model from MLflow...")
print("(This simulates serving startup: load module → build graph → ready)")
print()

loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)

# Test with standard Responses API format
test_input = {
    "input": [{"role": "user", "content": "What is the vacation policy?"}]
}

try:
    response = loaded_model.predict(test_input)
    print(f"✓ Model validation PASSED")
    print(f"  Response preview: {str(response)[:200]}...")
except Exception as e:
    print(f"✗ Model validation FAILED: {e}")
    print("  Fix the issue before deploying to serving.")
    print("  Common causes:")
    print("    • Missing pip requirement")
    print("    • Invalid MCP URL in model_config")
    print("    • Lakebase instance not accessible")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Examine the Registered Model

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

print(f"Registered Model: {MODEL_NAME}")
print("=" * 80)

versions = list(w.model_versions.list(full_name=MODEL_NAME))
for v in versions[-3:]:
    print(f"\n  Version {v.version}:")
    print(f"    Status: {v.status}")
    print(f"    Created: {v.created_at}")
    if v.run_id:
        print(f"    Run ID: {v.run_id}")

print(f"\nTotal versions: {len(versions)}")
print(f"\nModel location in Unity Catalog:")
print(f"  Catalog: {CATALOG}")
print(f"  Schema:  {SCHEMA}")
print(f"  Model:   knowledge_assistant")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What We Did
# MAGIC - ✅ Understood models-from-code (agent module + config)
# MAGIC - ✅ Examined the agent module (`src/agent.py`) — same LangGraph code as notebooks
# MAGIC - ✅ Defined model configuration mapping to each module
# MAGIC - ✅ Specified pip requirements for the serving environment
# MAGIC - ✅ Logged with `mlflow.pyfunc.log_model()` (ResponsesAgent auto-infers signature)
# MAGIC - ✅ Validated the model locally before deployment
# MAGIC - ✅ Registered in Unity Catalog for governance
# MAGIC
# MAGIC ### Key Concepts
# MAGIC
# MAGIC **Models-from-code logging pattern:**
# MAGIC ```python
# MAGIC # In agent.py — graph is 100% LangChain, ResponsesAgent is just the serving wrapper:
# MAGIC config = mlflow.models.ModelConfig()
# MAGIC graph, checkpointer = _build_graph(config)  # Same LangGraph code
# MAGIC
# MAGIC class KnowledgeAssistant(ResponsesAgent):
# MAGIC     def predict_stream(self, request):
# MAGIC         msgs = to_chat_completions_input([i.model_dump() for i in request.input])
# MAGIC         for _, events in graph.stream({"messages": msgs}, ...):
# MAGIC             yield from output_to_responses_items_stream(events["messages"])
# MAGIC
# MAGIC agent = KnowledgeAssistant()
# MAGIC mlflow.models.set_model(agent)
# MAGIC
# MAGIC # In logging notebook — no signature or input_example needed:
# MAGIC mlflow.pyfunc.log_model(
# MAGIC     python_model="src/agent.py",
# MAGIC     model_config={...},
# MAGIC     pip_requirements=[...],
# MAGIC     registered_model_name="catalog.schema.model",
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **What happens at serving time:**
# MAGIC
# MAGIC | Step | What Happens |
# MAGIC |------|-------------|
# MAGIC | 1 | Serving container starts, `pip_requirements` installed |
# MAGIC | 2 | `agent.py` module loaded, `ModelConfig()` reads stored config |
# MAGIC | 3 | `_build_graph()` — LLM, MCP tools, memory, graph all initialized |
# MAGIC | 4 | `KnowledgeAssistant()` instance created, `set_model()` registers it |
# MAGIC | 5 | Endpoint ready — `predict()` accepts `input`, returns `output` items |
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC Continue to [02_agent_serving.py](02_agent_serving.py) to deploy the model to a
# MAGIC serving endpoint and enable the Review App.

# COMMAND ----------

print("✓ Model logging complete!")
print(f"\nYour agent is registered at: {MODEL_NAME}")
print("Next: Deploy to a serving endpoint with agents.deploy()")
