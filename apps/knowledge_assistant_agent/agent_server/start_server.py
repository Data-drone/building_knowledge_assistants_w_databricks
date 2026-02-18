from dotenv import load_dotenv
from mlflow.genai.agent_server import AgentServer

# Load local env vars first for local development parity.
load_dotenv(dotenv_path=".env", override=True)

# Import registers @invoke and @stream handlers with AgentServer.
import agent_server.agent  # noqa: E402,F401

agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=False)
app = agent_server.app  # noqa: F841


def main():
    agent_server.run(app_import_string="agent_server.start_server:app")


if __name__ == "__main__":
    main()
