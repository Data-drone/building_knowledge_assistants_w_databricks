"""
Knowledge Assistant — Production Agent Module

Uses the same LangChain/LangGraph primitives from the bootcamp notebooks:
  - ChatDatabricks           (Module 00 — Mosaic AI Gateway)
  - StateGraph + ToolNode    (Module 01 — RAG Pipeline)
  - MessagesState            (Module 01 — same state schema everywhere)
  - CheckpointSaver          (Module 02 — Short-Term Memory)
  - MCP tools via ToolNode   (Module 04 — Genie + Custom Tools)

ResponsesAgent is a thin serving wrapper that tells agents.deploy() the model
speaks the Responses API format. The graph code inside is identical to the
notebooks.

Architecture:
    ResponsesAgent.predict(request)
         → to_chat_completions_input(request.input)    ← convert to messages
         → graph.invoke({"messages": [...]})            ← same LangGraph from notebooks
              → call_agent → should_continue → tools → call_agent → END
              → CheckpointSaver (Lakebase)
         → output_to_responses_items_stream(messages)   ← convert to Responses format
         → ResponsesAgentResponse(output=[...])
"""

import mlflow
import asyncio
import logging
import functools
from typing import List, Any, Optional, Generator

from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from databricks_langchain import ChatDatabricks, CheckpointSaver
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksOAuthClientProvider
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

logger = logging.getLogger(__name__)

# Shared WorkspaceClient — uses managed credentials on serving endpoints,
# notebook credentials during local dev
_ws = WorkspaceClient()


# ==============================================================================
# MCP TOOL LOADER (same as Module 04)
# ==============================================================================

def _run_async(coro):
    """Run an async coroutine from sync context, handling nested event loops."""
    try:
        asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result(timeout=30)
    except RuntimeError:
        return asyncio.run(coro)


def load_mcp_tools(mcp_url: str) -> List[StructuredTool]:
    """Connect to a Databricks MCP endpoint and return LangChain tools."""
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
        logger.warning(f"Failed to load MCP tools from {mcp_url}: {e}")
        return []


# ==============================================================================
# HELPER: Safe config accessor
# ==============================================================================

def _cfg(config, key, default=None):
    """Get a config value with a default. Works with both dict and ModelConfig."""
    try:
        val = config.get(key)
        return val if val is not None else default
    except (KeyError, TypeError):
        return default


# ==============================================================================
# GRAPH BUILDER — same LangChain/LangGraph code from the notebooks
# ==============================================================================

def _build_graph(config):
    """
    Build the LangGraph agent from config. This is the same code
    you wrote in Modules 01-04, extracted into a function.
    """
    # --- Module 00: LLM ---
    llm = ChatDatabricks(
        endpoint=_cfg(config, "llm_endpoint"),
        temperature=_cfg(config, "temperature", 0.1),
    )

    # --- Modules 01 + 04: MCP tools ---
    tools = []
    for key in ["vector_search_mcp_url", "genie_mcp_url", "uc_functions_mcp_url"]:
        url = _cfg(config, key)
        if url:
            tools.extend(load_mcp_tools(url))
    logger.info(f"Total tools loaded: {len(tools)}")

    llm_with_tools = llm.bind_tools(tools) if tools else llm

    # --- Module 02: Memory ---
    checkpointer = None
    if _cfg(config, "enable_memory") and _cfg(config, "lakebase_instance_name"):
        try:
            checkpointer = CheckpointSaver(
                instance_name=_cfg(config, "lakebase_instance_name")
            )
            checkpointer.setup()
            logger.info("Memory enabled via CheckpointSaver")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(
                    f"Memory setup failed (will work on serving endpoint): {e}"
                )
                checkpointer = None

    # --- Same graph wiring from every notebook ---
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
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")
    else:
        workflow.add_edge("agent", END)

    return workflow.compile(checkpointer=checkpointer), checkpointer


# ==============================================================================
# READ CONFIG & BUILD GRAPH AT MODULE LEVEL (models-from-code pattern)
# ==============================================================================

_config = mlflow.models.ModelConfig()
_graph, _checkpointer = _build_graph(_config)


# ==============================================================================
# ResponsesAgent — thin serving wrapper
#
# This is the ONLY non-LangChain part. It tells agents.deploy() that the model
# speaks the Responses API format. predict() just converts messages, calls
# graph.invoke(), and converts the result using MLflow's built-in helpers.
# ==============================================================================

class KnowledgeAssistant(ResponsesAgent):

    def _graph_config(self, request: ResponsesAgentRequest) -> dict:
        cfg = {"recursion_limit": _cfg(_config, "recursion_limit", 25)}
        if _checkpointer:
            thread_id = "default"
            if request.custom_inputs:
                thread_id = request.custom_inputs.get("thread_id", "default")
            cfg["configurable"] = {"thread_id": thread_id}
        return cfg

    @mlflow.trace(span_type="AGENT")
    def predict(
        self,
        request: ResponsesAgentRequest,
    ) -> ResponsesAgentResponse:
        # Collect all completed output items from the stream
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(
            output=outputs, custom_outputs=request.custom_inputs
        )

    @mlflow.trace(span_type="AGENT")
    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        # Convert Responses API input → ChatCompletions messages (for LangGraph)
        cc_msgs = to_chat_completions_input(
            [i.model_dump() for i in request.input]
        )

        # Stream through the LangGraph — same as every notebook
        for _, events in _graph.stream(
            {"messages": cc_msgs},
            config=self._graph_config(request),
            stream_mode=["updates"],
        ):
            for node_data in events.values():
                if "messages" in node_data:
                    # Convert LangGraph messages → Responses API output items
                    yield from output_to_responses_items_stream(
                        node_data["messages"]
                    )


mlflow.langchain.autolog()
agent = KnowledgeAssistant()
mlflow.models.set_model(agent)
