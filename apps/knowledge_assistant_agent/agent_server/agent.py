from typing import AsyncGenerator

from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.langgraph_agent import KnowledgeAssistant

_agent = KnowledgeAssistant()


@invoke()
async def invoke_agent(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    return await _agent.predict(request)


@stream()
async def stream_agent(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    async for event in _agent.predict_stream(request):
        yield event
