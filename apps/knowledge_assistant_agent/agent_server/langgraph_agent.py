import asyncio
import functools
import logging
import os
import uuid
from typing import Any, AsyncGenerator, Dict, List

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import AsyncCheckpointSaver, AsyncDatabricksStore, ChatDatabricks
from databricks_mcp import DatabricksOAuthClientProvider
from langchain_core.messages import SystemMessage
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

from agent_server.utils_memory import get_user_id, memory_tools

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


def _resolve_thread_id(request: ResponsesAgentRequest) -> str:
    custom_inputs = request.custom_inputs or {}
    if custom_inputs.get("thread_id"):
        thread_id = str(custom_inputs["thread_id"])
        logger.info("Using thread_id from request.custom_inputs")
        return thread_id

    conversation_id = None
    if getattr(request, "context", None):
        conversation_id = getattr(request.context, "conversation_id", None)
    if conversation_id is None:
        conversation_id = getattr(request, "conversation_id", None)
    if conversation_id:
        thread_id = str(conversation_id)
        logger.info("Using thread_id from request conversation context")
        return thread_id

    # Generate an isolated fallback thread to avoid leaking state across users.
    thread_id = str(uuid.uuid4())
    logger.info("No thread_id provided; generated a new isolated thread_id")
    return thread_id


def _runtime_config() -> Dict[str, Any]:
    host = os.getenv("DATABRICKS_HOST") or _ws.config.host or ""
    if host and not host.startswith("http://") and not host.startswith("https://"):
        host = f"https://{host}"
    catalog = os.getenv("CATALOG", "agent_bootcamp")
    schema = os.getenv("SCHEMA", "knowledge_assistant")
    genie_space_id = os.getenv("GENIE_SPACE_ID", "")
    return {
        "host": host,
        "llm_endpoint": os.getenv("LLM_ENDPOINT", "databricks-claude-sonnet-4-6"),
        "temperature": float(os.getenv("TEMPERATURE", "0.1")),
        "enable_memory": os.getenv("ENABLE_MEMORY", "false").lower() == "true",
        "enable_mcp_tools": os.getenv("ENABLE_MCP_TOOLS", "false").lower() == "true",
        "lakebase_instance_name": os.getenv("LAKEBASE_INSTANCE_NAME", ""),
        "lakebase_autoscaling_project": os.getenv("LAKEBASE_AUTOSCALING_PROJECT", ""),
        "lakebase_autoscaling_branch": os.getenv("LAKEBASE_AUTOSCALING_BRANCH", ""),
        "embedding_endpoint": os.getenv("EMBEDDING_ENDPOINT", "databricks-gte-large-en"),
        "embedding_dims": int(os.getenv("EMBEDDING_DIMS", "1024")),
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


def _has_autoscaling_config(config: Dict[str, Any]) -> bool:
    return bool(_cfg(config, "lakebase_autoscaling_project")) and bool(
        _cfg(config, "lakebase_autoscaling_branch")
    )


def _lakebase_target(config: Dict[str, Any]) -> str:
    if _cfg(config, "lakebase_instance_name"):
        return str(_cfg(config, "lakebase_instance_name"))
    if _has_autoscaling_config(config):
        return (
            f"{_cfg(config, 'lakebase_autoscaling_project')}/"
            f"{_cfg(config, 'lakebase_autoscaling_branch')}"
        )
    return "<missing Lakebase config>"


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


MEMORY_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to long-term memory tools.\n\n"
    "You have three memory tools to manage what you know about users:\n"
    "- get_user_memory: Search for previously saved information about the user\n"
    "- save_user_memory: Save important facts, preferences, or details the user shares\n"
    "- delete_user_memory: Remove specific information when asked\n\n"
    "At the start of each conversation, check for relevant memories to provide "
    "personalized responses. When a user shares important preferences or facts, "
    "save them for future reference."
)


def _build_workflow(config: Dict[str, Any]):
    llm = ChatDatabricks(
        endpoint=_cfg(config, "llm_endpoint"),
        temperature=_cfg(config, "temperature", 0.1),
    )

    if _cfg(config, "enable_memory") and not (
        _cfg(config, "lakebase_instance_name") or _has_autoscaling_config(config)
    ):
        raise ValueError(
            "ENABLE_MEMORY requires either LAKEBASE_INSTANCE_NAME or both "
            "LAKEBASE_AUTOSCALING_PROJECT and LAKEBASE_AUTOSCALING_BRANCH."
        )

    tools = []
    if _cfg(config, "enable_mcp_tools", False):
        for key in ["vector_search_mcp_url", "genie_mcp_url", "uc_functions_mcp_url"]:
            url = _cfg(config, key)
            if url:
                tools.extend(load_mcp_tools(url))

    use_memory = _cfg(config, "enable_memory", False)
    if use_memory:
        tools.extend(memory_tools())

    llm_with_tools = llm.bind_tools(tools) if tools else llm

    def call_agent(state: MessagesState):
        messages = list(state["messages"])
        if use_memory:
            messages = [SystemMessage(content=MEMORY_SYSTEM_PROMPT)] + messages
        return {"messages": [llm_with_tools.invoke(messages)]}

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

    return workflow


class KnowledgeAssistant:
    def __init__(self):
        _configure_mlflow_experiment()
        self._config = _runtime_config()
        self._workflow = _build_workflow(self._config)
        mlflow.langchain.autolog()

    def _graph_config(self, request: ResponsesAgentRequest) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"recursion_limit": _cfg(self._config, "recursion_limit", 25)}
        if _cfg(self._config, "enable_memory"):
            cfg["configurable"] = {"thread_id": _resolve_thread_id(request)}
            user_id = get_user_id(request)
            if user_id:
                cfg["configurable"]["user_id"] = user_id
        return cfg

    @mlflow.trace(span_type="AGENT")
    async def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            async for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    @mlflow.trace(span_type="AGENT")
    async def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
        cc_msgs = to_chat_completions_input([item.model_dump() for item in request.input])

        if _cfg(self._config, "enable_memory"):
            lakebase_kwargs = {
                "instance_name": _cfg(self._config, "lakebase_instance_name") or None,
                "project": _cfg(self._config, "lakebase_autoscaling_project") or None,
                "branch": _cfg(self._config, "lakebase_autoscaling_branch") or None,
            }

            try:
                checkpointer = AsyncCheckpointSaver(**lakebase_kwargs)
            except Exception as e:
                raise RuntimeError(
                    "ENABLE_MEMORY is true, but AsyncCheckpointSaver initialization failed. "
                    f"Verify Lakebase access for {_lakebase_target(self._config)}."
                ) from e

            async with checkpointer:
                try:
                    await checkpointer.setup()
                except Exception as e:
                    raise RuntimeError(
                        "ENABLE_MEMORY is true, but AsyncCheckpointSaver setup failed. "
                        f"Verify Lakebase access for {_lakebase_target(self._config)}."
                    ) from e

                async with AsyncDatabricksStore(
                    **lakebase_kwargs,
                    embedding_endpoint=_cfg(self._config, "embedding_endpoint"),
                    embedding_dims=_cfg(self._config, "embedding_dims"),
                ) as store:
                    await store.setup()

                    graph = self._workflow.compile(checkpointer=checkpointer)
                    graph_config = self._graph_config(request)
                    graph_config["configurable"]["store"] = store

                    async for _, events in graph.astream(
                        {"messages": cc_msgs},
                        config=graph_config,
                        stream_mode=["updates"],
                    ):
                        for node_data in events.values():
                            if "messages" in node_data:
                                for event in output_to_responses_items_stream(
                                    node_data["messages"]
                                ):
                                    yield event
            return

        graph = self._workflow.compile()
        async for _, events in graph.astream(
            {"messages": cc_msgs},
            config=self._graph_config(request),
            stream_mode=["updates"],
        ):
            for node_data in events.values():
                if "messages" in node_data:
                    for event in output_to_responses_items_stream(node_data["messages"]):
                        yield event
