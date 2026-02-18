import asyncio
import functools
import logging
import os
from typing import Any, Dict, Generator, List

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks, CheckpointSaver
from databricks_mcp import DatabricksOAuthClientProvider
from langchain_core.tools import StructuredTool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

logger = logging.getLogger(__name__)
_ws = WorkspaceClient()


def _run_async(coro):
    """Run an async coroutine from sync code."""
    try:
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result(timeout=30)
    except RuntimeError:
        return asyncio.run(coro)


def _cfg(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    val = config.get(key)
    return default if val is None else val


def _build_mcp_url(host: str, path: str) -> str:
    return f"{host.rstrip('/')}{path}"


def _runtime_config() -> Dict[str, Any]:
    host = os.getenv("DATABRICKS_HOST") or _ws.config.host or ""
    if host and not host.startswith("http://") and not host.startswith("https://"):
        host = f"https://{host}"
    catalog = os.getenv("CATALOG", "agent_bootcamp")
    schema = os.getenv("SCHEMA", "knowledge_assistant")
    genie_space_id = os.getenv("GENIE_SPACE_ID", "")
    return {
        "host": host,
        "llm_endpoint": os.getenv("LLM_ENDPOINT", "databricks-claude-sonnet-4-5"),
        "temperature": float(os.getenv("TEMPERATURE", "0.1")),
        "enable_memory": os.getenv("ENABLE_MEMORY", "false").lower() == "true",
        "enable_mcp_tools": os.getenv("ENABLE_MCP_TOOLS", "false").lower() == "true",
        "lakebase_instance_name": os.getenv("LAKEBASE_INSTANCE_NAME", ""),
        "recursion_limit": int(os.getenv("RECURSION_LIMIT", "25")),
        "vector_search_mcp_url": os.getenv(
            "VECTOR_SEARCH_MCP_URL",
            _build_mcp_url(host, f"/api/2.0/mcp/vector-search/{catalog}/{schema}"),
        ),
        "genie_mcp_url": os.getenv(
            "GENIE_MCP_URL",
            _build_mcp_url(host, f"/api/2.0/mcp/genie/{genie_space_id}") if genie_space_id else "",
        ),
        "uc_functions_mcp_url": os.getenv(
            "UC_FUNCTIONS_MCP_URL",
            _build_mcp_url(host, f"/api/2.0/mcp/functions/{catalog}/{schema}"),
        ),
    }


def _configure_mlflow_experiment() -> None:
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "").strip()
    if not exp_name:
        return
    try:
        mlflow.set_experiment(exp_name)
    except Exception as e:
        logger.warning("Failed to set MLflow experiment '%s': %s", exp_name, e)


def load_mcp_tools(mcp_url: str) -> List[StructuredTool]:
    def _call_tool(url: str, name: str, **kwargs) -> Any:
        async def _invoke():
            auth = DatabricksOAuthClientProvider(_ws)
            async with streamablehttp_client(url, auth) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    result = await session.call_tool(name, kwargs)
                    return result.content

        return _run_async(_invoke())

    async def _discover():
        auth = DatabricksOAuthClientProvider(_ws)
        async with streamablehttp_client(mcp_url, auth) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                return (await session.list_tools()).tools

    try:
        mcp_tools = _run_async(_discover())
        return [
            StructuredTool.from_function(
                name=t.name,
                description=t.description or "",
                func=functools.partial(_call_tool, mcp_url, t.name),
            )
            for t in mcp_tools
        ]
    except Exception as e:
        logger.warning("Failed to load MCP tools from %s: %s", mcp_url, e)
        return []


def _build_graph(config: Dict[str, Any]):
    llm = ChatDatabricks(
        endpoint=_cfg(config, "llm_endpoint"),
        temperature=_cfg(config, "temperature", 0.1),
    )

    tools = []
    if _cfg(config, "enable_mcp_tools", False):
        for key in ["vector_search_mcp_url", "genie_mcp_url", "uc_functions_mcp_url"]:
            url = _cfg(config, key)
            if url:
                tools.extend(load_mcp_tools(url))

    llm_with_tools = llm.bind_tools(tools) if tools else llm

    checkpointer = None
    if _cfg(config, "enable_memory") and _cfg(config, "lakebase_instance_name"):
        try:
            checkpointer = CheckpointSaver(instance_name=_cfg(config, "lakebase_instance_name"))
            checkpointer.setup()
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning("Checkpoint setup failed, proceeding without memory: %s", e)
                checkpointer = None

    def call_agent(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def should_continue(state: MessagesState):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_agent)
    if tools:
        workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")

    if tools:
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")
    else:
        workflow.add_edge("agent", END)

    return workflow.compile(checkpointer=checkpointer), checkpointer


class KnowledgeAssistant:
    def __init__(self):
        _configure_mlflow_experiment()
        self._config = _runtime_config()
        self._graph, self._checkpointer = _build_graph(self._config)
        mlflow.langchain.autolog()

    def _graph_config(self, request: ResponsesAgentRequest) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"recursion_limit": _cfg(self._config, "recursion_limit", 25)}
        if self._checkpointer:
            thread_id = "default"
            if request.custom_inputs:
                thread_id = request.custom_inputs.get("thread_id", "default")
            cfg["configurable"] = {"thread_id": thread_id}
        return cfg

    @mlflow.trace(span_type="AGENT")
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    @mlflow.trace(span_type="AGENT")
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        cc_msgs = to_chat_completions_input([item.model_dump() for item in request.input])
        for _, events in self._graph.stream(
            {"messages": cc_msgs},
            config=self._graph_config(request),
            stream_mode=["updates"],
        ):
            for node_data in events.values():
                if "messages" in node_data:
                    yield from output_to_responses_items_stream(node_data["messages"])
