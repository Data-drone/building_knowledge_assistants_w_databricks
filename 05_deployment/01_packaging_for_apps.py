# Databricks notebook source
# MAGIC %md
# MAGIC # Packaging Your Agent for Databricks Apps
# MAGIC
# MAGIC You have a working LangGraph agent from the previous modules. This notebook
# MAGIC shows how that same agent gets packaged into a Databricks App — no rewrite
# MAGIC required.
# MAGIC
# MAGIC **Estimated time:** 15-20 minutes
# MAGIC
# MAGIC **Where this fits:** Start here. Read this before deploying in `02_deploy_and_validate.py`.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC - Completed Modules 01-04
# MAGIC - App source code present under `apps/knowledge_assistant_agent`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Why Databricks Apps
# MAGIC
# MAGIC Model Serving for agents is being sunsetted. Databricks Apps is the
# MAGIC replacement — and it is a better fit for agent workloads anyway:
# MAGIC
# MAGIC - **Fast iteration** — sync files and redeploy in seconds, not minutes
# MAGIC - **Git-based** — standard Python project, version it like anything else
# MAGIC - **Async** — native `async` Python, so the server handles concurrent requests properly
# MAGIC - **Flexible** — any framework, custom routes, middleware if you need it
# MAGIC - **Observable** — MLflow tracing wired in automatically
# MAGIC - **Chat UI included** — ships with a conversational frontend out of the box
# MAGIC
# MAGIC Docs:
# MAGIC - [Author an agent on Databricks Apps](https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent-db-app)
# MAGIC - [Migrate from Model Serving to Apps](https://docs.databricks.com/aws/en/generative-ai/agent-framework/migrate-agent-to-apps)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Official Templates
# MAGIC
# MAGIC Databricks maintains starter templates at
# MAGIC [databricks/app-templates](https://github.com/databricks/app-templates).
# MAGIC Three LangGraph variants map directly to what you built in this bootcamp:
# MAGIC
# MAGIC | Template | Maps to | What's in it |
# MAGIC |---|---|---|
# MAGIC | [`agent-langgraph`](https://github.com/databricks/app-templates/tree/main/agent-langgraph) | Module 01 | Base agent + `AgentServer` + chat UI |
# MAGIC | [`agent-langgraph-short-term-memory`](https://github.com/databricks/app-templates/tree/main/agent-langgraph-short-term-memory) | Module 02 | Adds `AsyncCheckpointSaver` with `thread_id` |
# MAGIC | [`agent-langgraph-long-term-memory`](https://github.com/databricks/app-templates/tree/main/agent-langgraph-long-term-memory) | Module 02 | Adds `AsyncDatabricksStore` for per-user facts |
# MAGIC
# MAGIC Our bootcamp app is the combined version — base agent, both memory
# MAGIC layers, and optional MCP tools from Module 04 all in one.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: App Source Layout
# MAGIC
# MAGIC Everything lives under `apps/knowledge_assistant_agent/`:
# MAGIC
# MAGIC ```
# MAGIC apps/knowledge_assistant_agent/
# MAGIC   app.yaml                          # Env vars, start command
# MAGIC   requirements.txt                  # Python deps
# MAGIC   agent_server/
# MAGIC     start_server.py                 # HTTP server entry point
# MAGIC     agent.py                        # @invoke / @stream handlers
# MAGIC     langgraph_agent.py              # The LangGraph graph — same pattern as earlier modules
# MAGIC     utils_memory.py                 # Async memory tools
# MAGIC   static/
# MAGIC     index.html                      # Chat UI
# MAGIC ```
# MAGIC
# MAGIC We will walk through each file below, outside-in.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3a: `app.yaml`
# MAGIC
# MAGIC Think of this as the config cell at the top of every notebook, but as YAML.
# MAGIC It tells Databricks Apps what command to run and which env vars to set.
# MAGIC
# MAGIC ```yaml
# MAGIC command: ["python", "-m", "agent_server.start_server"]
# MAGIC
# MAGIC env:
# MAGIC   - name: MLFLOW_TRACKING_URI
# MAGIC     value: "databricks"
# MAGIC   - name: MLFLOW_EXPERIMENT_NAME
# MAGIC     value: "/Shared/knowledge_assistant_agent_app"
# MAGIC   - name: LLM_ENDPOINT
# MAGIC     value: "databricks-claude-sonnet-4-6"
# MAGIC   - name: CATALOG
# MAGIC     value: "agent_bootcamp"
# MAGIC   - name: SCHEMA
# MAGIC     value: "knowledge_assistant"
# MAGIC   - name: LAKEBASE_AUTOSCALING_PROJECT
# MAGIC     value: "knowledge-assistant-state"
# MAGIC   - name: LAKEBASE_AUTOSCALING_BRANCH
# MAGIC     value: "production"
# MAGIC   - name: ENABLE_MEMORY
# MAGIC     value: "true"
# MAGIC   - name: ENABLE_MCP_TOOLS
# MAGIC     value: "false"
# MAGIC ```
# MAGIC
# MAGIC Two flags worth noting:
# MAGIC
# MAGIC - **`ENABLE_MEMORY=true`** — turns on `AsyncCheckpointSaver` and
# MAGIC   `AsyncDatabricksStore` (the Lakebase-backed memory from Module 02)
# MAGIC - **`ENABLE_MCP_TOOLS=true`** — connects to the Vector Search, Genie,
# MAGIC   and UC functions MCP endpoints (the tools from Module 04)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b: `start_server.py`
# MAGIC
# MAGIC Entry point. Creates the `AgentServer`, mounts the chat UI, starts serving:
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.agent_server import AgentServer
# MAGIC
# MAGIC import agent_server.agent  # registers @invoke and @stream handlers
# MAGIC
# MAGIC agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)
# MAGIC app = agent_server.app
# MAGIC
# MAGIC @app.get("/", response_class=HTMLResponse)
# MAGIC async def chat_ui():
# MAGIC     return (_STATIC_DIR / "index.html").read_text()
# MAGIC ```
# MAGIC
# MAGIC `AgentServer` is MLflow's FastAPI wrapper. It gives you `POST /invocations`
# MAGIC plus request routing, error handling, and trace logging out of the box.
# MAGIC
# MAGIC The `import agent_server.agent` line matters — importing the module is
# MAGIC what registers the `@invoke` and `@stream` handlers with the server.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3c: `agent.py`
# MAGIC
# MAGIC Thin delegation layer. Two functions, both just call into `KnowledgeAssistant`:
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.agent_server import invoke, stream
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC )
# MAGIC from agent_server.langgraph_agent import KnowledgeAssistant
# MAGIC
# MAGIC _agent = KnowledgeAssistant()
# MAGIC
# MAGIC @invoke()
# MAGIC async def invoke_agent(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC     return await _agent.predict(request)
# MAGIC
# MAGIC @stream()
# MAGIC async def stream_agent(request: ResponsesAgentRequest):
# MAGIC     async for event in _agent.predict_stream(request):
# MAGIC         yield event
# MAGIC ```
# MAGIC
# MAGIC `@invoke` and `@stream` register these as the `POST /invocations` handlers.
# MAGIC The types use the [OpenAI Responses API](https://platform.openai.com/docs/guides/responses-vs-chat-completions)
# MAGIC schema, which is what AI Playground, Agent Evaluation, and Agent
# MAGIC Monitoring all expect.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3d: `langgraph_agent.py`
# MAGIC
# MAGIC This is the actual agent. Strip away the serving plumbing and the graph
# MAGIC construction looks exactly like what you wrote in the notebooks:
# MAGIC
# MAGIC ```python
# MAGIC from langgraph.graph import END, MessagesState, StateGraph
# MAGIC from langgraph.prebuilt import ToolNode
# MAGIC from databricks_langchain import ChatDatabricks
# MAGIC
# MAGIC def _build_workflow(config):
# MAGIC     llm = ChatDatabricks(endpoint=config["llm_endpoint"])
# MAGIC
# MAGIC     # Collect tools (MCP-discovered + memory tools)
# MAGIC     tools = []
# MAGIC     if config["enable_mcp_tools"]:
# MAGIC         tools.extend(load_mcp_tools(...))   # Module 04
# MAGIC     if config["enable_memory"]:
# MAGIC         tools.extend(memory_tools())          # Module 02
# MAGIC
# MAGIC     llm_with_tools = llm.bind_tools(tools) if tools else llm
# MAGIC
# MAGIC     def call_agent(state: MessagesState):
# MAGIC         return {"messages": [llm_with_tools.invoke(state["messages"])]}
# MAGIC
# MAGIC     def should_continue(state: MessagesState):
# MAGIC         if state["messages"][-1].tool_calls:
# MAGIC             return "tools"
# MAGIC         return END
# MAGIC
# MAGIC     workflow = StateGraph(MessagesState)
# MAGIC     workflow.add_node("agent", call_agent)
# MAGIC     workflow.add_node("tools", ToolNode(tools))
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
# MAGIC     workflow.add_edge("tools", "agent")
# MAGIC     return workflow
# MAGIC ```
# MAGIC
# MAGIC `StateGraph`, `MessagesState`, `ToolNode`, `ChatDatabricks`, `bind_tools`,
# MAGIC `add_conditional_edges` — all familiar. The only real difference is that
# MAGIC tool selection is driven by env vars instead of being hardcoded in a cell.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: The ResponsesAgent Bridge
# MAGIC
# MAGIC The graph produces LangChain messages. The `/invocations` endpoint speaks
# MAGIC Responses API. Something needs to translate between the two. That is what
# MAGIC `KnowledgeAssistant` does — a three-step conversion:
# MAGIC
# MAGIC ```
# MAGIC Responses API request
# MAGIC        |
# MAGIC        v
# MAGIC  to_chat_completions_input()     -- convert to LangChain messages
# MAGIC        |
# MAGIC        v
# MAGIC  graph.astream(messages)          -- run the LangGraph workflow
# MAGIC        |
# MAGIC        v
# MAGIC  output_to_responses_items_stream()  -- convert back to Responses API events
# MAGIC        |
# MAGIC        v
# MAGIC Responses API stream events
# MAGIC ```
# MAGIC
# MAGIC Here is the relevant bit of `predict_stream`:
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.types.responses import (
# MAGIC     to_chat_completions_input,
# MAGIC     output_to_responses_items_stream,
# MAGIC )
# MAGIC
# MAGIC async def predict_stream(self, request):
# MAGIC     # 1. Convert incoming Responses API items to LangChain messages
# MAGIC     cc_msgs = to_chat_completions_input(
# MAGIC         [item.model_dump() for item in request.input]
# MAGIC     )
# MAGIC
# MAGIC     # 2. Run the LangGraph workflow (the part you already know)
# MAGIC     graph = self._workflow.compile(checkpointer=checkpointer)
# MAGIC     async for _, events in graph.astream(
# MAGIC         {"messages": cc_msgs}, config=graph_config, stream_mode=["updates"]
# MAGIC     ):
# MAGIC         for node_data in events.values():
# MAGIC             if "messages" in node_data:
# MAGIC                 # 3. Convert LangChain output to Responses API events
# MAGIC                 for event in output_to_responses_items_stream(node_data["messages"]):
# MAGIC                     yield event
# MAGIC ```
# MAGIC
# MAGIC That is the glue. AI Playground, Agent Evaluation, and Agent Monitoring
# MAGIC all expect Responses API format — these two converter functions handle it
# MAGIC so the agent logic can stay as plain LangGraph.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Memory — Notebook vs App
# MAGIC
# MAGIC In the notebooks you wrote this:
# MAGIC
# MAGIC ```python
# MAGIC # Notebook pattern (Module 02)
# MAGIC from databricks_langchain import CheckpointSaver, DatabricksStore
# MAGIC
# MAGIC checkpointer = CheckpointSaver(instance_name="my-lakebase")
# MAGIC store = DatabricksStore(instance_name="my-lakebase", ...)
# MAGIC
# MAGIC graph = workflow.compile(checkpointer=checkpointer)
# MAGIC graph.invoke(
# MAGIC     {"messages": [...]},
# MAGIC     config={"configurable": {"thread_id": "my-thread", "store": store}}
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC In the app, same idea — just async, and `thread_id` comes from the HTTP
# MAGIC request instead of a hardcoded string:
# MAGIC
# MAGIC ```python
# MAGIC # App pattern (langgraph_agent.py)
# MAGIC from databricks_langchain import AsyncCheckpointSaver, AsyncDatabricksStore
# MAGIC
# MAGIC checkpointer = AsyncCheckpointSaver(**lakebase_kwargs)
# MAGIC store = AsyncDatabricksStore(**lakebase_kwargs, ...)
# MAGIC
# MAGIC graph = self._workflow.compile(checkpointer=checkpointer)
# MAGIC # thread_id comes from the HTTP request's custom_inputs
# MAGIC thread_id = _resolve_thread_id(request)
# MAGIC graph_config = {"configurable": {"thread_id": thread_id, "store": store}}
# MAGIC ```
# MAGIC
# MAGIC Side by side:
# MAGIC
# MAGIC | Aspect | Notebook (Module 02) | App |
# MAGIC |---|---|---|
# MAGIC | Sync model | `CheckpointSaver` (sync) | `AsyncCheckpointSaver` (async) |
# MAGIC | Store | `DatabricksStore` | `AsyncDatabricksStore` |
# MAGIC | `thread_id` source | Hardcoded string | Extracted from `request.custom_inputs` |
# MAGIC | `user_id` source | Hardcoded string | Extracted from `request.custom_inputs` |
# MAGIC | Lakebase config | `instance_name` parameter | Environment variables in `app.yaml` |
# MAGIC
# MAGIC The memory tools themselves (`get_user_memory`, `save_user_memory`,
# MAGIC `delete_user_memory`) in `utils_memory.py` are the same `@tool` functions
# MAGIC from Module 02 — just `async def` instead of `def`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: MCP Tools
# MAGIC
# MAGIC In Module 04 you set up MCP endpoints for Vector Search, Genie, and UC
# MAGIC functions. Set `ENABLE_MCP_TOOLS=true` in `app.yaml` and the app picks
# MAGIC them up at startup:
# MAGIC
# MAGIC ```python
# MAGIC # langgraph_agent.py — tool discovery
# MAGIC if config["enable_mcp_tools"]:
# MAGIC     for key in ["vector_search_mcp_url", "genie_mcp_url", "uc_functions_mcp_url"]:
# MAGIC         url = config[key]
# MAGIC         if url:
# MAGIC             tools.extend(load_mcp_tools(url))
# MAGIC ```
# MAGIC
# MAGIC `load_mcp_tools()` connects to each endpoint, calls `list_tools()`, and
# MAGIC wraps each tool as a LangChain `StructuredTool` — then `llm.bind_tools()`
# MAGIC makes them available to the graph, same as Module 04.
# MAGIC
# MAGIC Default MCP URLs point at the workspace's managed endpoints:
# MAGIC - `/api/2.0/mcp/vector-search/{catalog}/{schema}`
# MAGIC - `/api/2.0/mcp/genie/{genie_space_id}`
# MAGIC - `/api/2.0/mcp/functions/{catalog}/{schema}`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Verify App Source Exists

# COMMAND ----------

import os

app_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(""))),
    "apps", "knowledge_assistant_agent",
)

# When running in a Databricks notebook the above heuristic may not resolve.
# Fall back to a workspace-relative path.
if not os.path.isdir(app_dir):
    import sys
    nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    repo_root = "/Workspace" + "/".join(nb_path.split("/")[:-2])
    app_dir = os.path.join(repo_root, "apps", "knowledge_assistant_agent")

expected_files = [
    "app.yaml",
    "requirements.txt",
    "agent_server/start_server.py",
    "agent_server/agent.py",
    "agent_server/langgraph_agent.py",
    "agent_server/utils_memory.py",
    "static/index.html",
]

print(f"App source directory: {app_dir}\n")
for f in expected_files:
    full = os.path.join(app_dir, f)
    status = "OK" if os.path.exists(full) else "MISSING"
    print(f"  [{status}] {f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The app is not a rewrite. It is the same LangGraph graph wrapped in three
# MAGIC thin layers:
# MAGIC
# MAGIC 1. **`app.yaml`** — config that lived in notebook cells, now YAML
# MAGIC 2. **`@invoke` / `@stream`** — registers handlers with `AgentServer`
# MAGIC 3. **`to_chat_completions_input` / `output_to_responses_items_stream`** —
# MAGIC    translates between Responses API and LangChain message formats
# MAGIC
# MAGIC The `StateGraph`, `ToolNode`, `ChatDatabricks`, checkpointer, store, MCP
# MAGIC discovery — all the same code from earlier modules.
# MAGIC
# MAGIC **Next:** `02_deploy_and_validate.py` deploys this app and confirms it
# MAGIC works.
