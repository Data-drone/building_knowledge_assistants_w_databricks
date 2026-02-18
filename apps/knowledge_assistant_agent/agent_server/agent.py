from typing import Generator

from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.langgraph_agent import KnowledgeAssistant

_agent = KnowledgeAssistant()


@invoke()
def invoke_agent(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    return _agent.predict(request)


@stream()
def stream_agent(
    request: ResponsesAgentRequest,
) -> Generator[ResponsesAgentStreamEvent, None, None]:
    yield from _agent.predict_stream(request)
