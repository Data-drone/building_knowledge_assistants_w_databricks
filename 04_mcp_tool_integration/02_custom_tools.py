# Databricks notebook source
# MAGIC %md
# MAGIC # Custom Tools & Advanced Patterns
# MAGIC
# MAGIC ## Two Learning Paths
# MAGIC
# MAGIC **Path A — Quick Start (30 min):**
# MAGIC Steps 2, 3, and 6. Covers when to build custom tools, how to write one,
# MAGIC and how to deploy it as a Databricks App. Skip Steps 4-5.
# MAGIC
# MAGIC **Path B — Deep Dive (50 min):**
# MAGIC All steps. Adds MCP server architecture concepts (Step 4) and
# MAGIC local server testing (Step 5) before production deployment.
# MAGIC
# MAGIC ## What You'll Build
# MAGIC - Decide when a custom tool is needed vs managed MCP (Genie, SQL, UC Functions)
# MAGIC - Build a simple custom tool with the `@tool` decorator
# MAGIC - Package it as a FastMCP server and deploy to Databricks Apps
# MAGIC - Connect the deployed server back to a LangChain agent
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed [01_genie_integration.py](01_genie_integration.py)
# MAGIC - Understanding of LangChain tools and agents

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies
# MAGIC
# MAGIC In addition to the usual stack, this notebook adds three new packages:
# MAGIC - `mcp` — the MCP protocol client/server library
# MAGIC - `databricks-mcp` — Databricks-specific MCP helpers (OAuth, transport)
# MAGIC - `nest_asyncio` — allows async MCP calls inside notebook event loops

# COMMAND ----------

%pip install -q --upgrade \
  "databricks-sdk>=0.101" \
  "mlflow[databricks]>=3.10" \
  "databricks-langchain[memory]>=0.17" \
  "databricks-vectorsearch>=0.66" \
  "langgraph>=1.1" \
  "langchain-core>=1.2" \
  mcp \
  databricks-mcp \
  nest_asyncio

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: When to Build Custom Tools (5 min)
# MAGIC
# MAGIC ### Use Managed MCPs When:
# MAGIC - Querying Databricks tables (SQL, Genie)
# MAGIC - Calling Unity Catalog Functions
# MAGIC - Standard Databricks operations
# MAGIC - You want zero setup and production-ready tools
# MAGIC
# MAGIC ### Build Custom Tools When:
# MAGIC - Wrapping external APIs (weather, stock prices, news)
# MAGIC - Custom business logic not in UC Functions
# MAGIC - Third-party integrations (Slack, Salesforce, GitHub)
# MAGIC - Rate limiting or caching layer needed
# MAGIC - Legacy system integration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision Tree
# MAGIC
# MAGIC **Question: Where does the data/capability live?**
# MAGIC
# MAGIC ```
# MAGIC ├─ In Databricks
# MAGIC │   └─> Use Managed MCP (SQL, Genie, UC Functions)
# MAGIC │
# MAGIC ├─ External API/Service
# MAGIC │   └─> Build Custom Tool
# MAGIC │
# MAGIC └─ Complex multi-step logic
# MAGIC     └─> Consider UC Function OR Custom MCP Server
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Build a Simple Custom Tool (10 min)
# MAGIC
# MAGIC Let's build a simple tool that wraps an external weather API.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example: Weather API Tool
# MAGIC
# MAGIC This tool will:
# MAGIC - Call a free weather API
# MAGIC - Return current weather for a city
# MAGIC - Be usable by our agent

# COMMAND ----------

from langchain_core.tools import tool
import requests

@tool
def get_weather(city: str) -> str:
    """
    Get current weather conditions for a city.

    Use this when users ask about:
    - Current weather
    - Temperature
    - Weather conditions
    - Whether to plan outdoor activities

    Args:
        city: City name (e.g., "San Francisco", "New York")
    """
    try:
        # Using wttr.in free weather API
        response = requests.get(
            f"https://wttr.in/{city}?format=j1",
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()
            current = data['current_condition'][0]
            temp_c = current['temp_C']
            description = current['weatherDesc'][0]['value']
            humidity = current['humidity']

            return f"Weather in {city}: {description}, {temp_c}°C, {humidity}% humidity"
        else:
            return f"Could not fetch weather for {city}. Status: {response.status_code}"

    except requests.exceptions.Timeout:
        return f"Weather API timed out for {city}"
    except Exception as e:
        return f"Error getting weather: {str(e)}"

print("✅ Weather tool created!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test the Custom Tool
# MAGIC
# MAGIC Let's test the weather tool directly before adding it to an agent.

# COMMAND ----------

# Test the tool directly
test_result = get_weather.invoke({"city": "San Francisco"})
print("Test: get_weather('San Francisco')")
print(test_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add Custom Tool to Agent
# MAGIC
# MAGIC Now let's add this to an agent:

# COMMAND ----------

from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

# Configuration
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"

# Initialize LLM
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1)

# Collect tools
tools = [
    get_weather  # Our custom tool!
]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Build simple agent
def call_model(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

agent = workflow.compile()

print("✅ Agent created with custom weather tool!")

# COMMAND ----------

# Test agent with weather question
result = agent.invoke({
    "messages": [HumanMessage(content="What's the weather like in Seattle?")]
})

print("Question: What's the weather like in Seattle?")
print(f"Answer: {result['messages'][-1].content}")
print()
print("✅ Agent successfully used the custom tool!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Storing API Keys Securely
# MAGIC
# MAGIC **Best Practice**: Use Databricks Secrets for API keys

# COMMAND ----------

# MAGIC %md
# MAGIC **Best Practice Steps:**
# MAGIC
# MAGIC 1. **Create a secret scope** (run once):
# MAGIC    ```python
# MAGIC    dbutils.secrets.createScope("api_keys")
# MAGIC    ```
# MAGIC
# MAGIC 2. **Store your API key** (run once):
# MAGIC    ```python
# MAGIC    dbutils.secrets.put("api_keys", "openweather_api_key", "your-api-key-here")
# MAGIC    ```
# MAGIC
# MAGIC 3. **Access in your tool**:
# MAGIC    ```python
# MAGIC    api_key = dbutils.secrets.get("api_keys", "openweather_api_key")
# MAGIC    ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Quick Start Path: Jump to Step 6
# MAGIC
# MAGIC If you are following Path A, scroll down to **Step 6: Deploy MCP Server
# MAGIC to Production** and skip Steps 4-5.
# MAGIC
# MAGIC For Path B, continue below.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: MCP Server Concepts (OPTIONAL - 10 min)
# MAGIC
# MAGIC **⏭️ Skip this step if you want to go straight to production deployment (Step 6).**
# MAGIC
# MAGIC This step explains MCP server architecture for deeper understanding.

# COMMAND ----------

# MAGIC %md
# MAGIC ### When You Need a Full MCP Server:
# MAGIC
# MAGIC - Multiple related tools (e.g., math operations: add, subtract, multiply)
# MAGIC - Complex state management
# MAGIC - Need stdio or HTTP transport
# MAGIC - Want to reuse across multiple agents/projects
# MAGIC - Need fine-grained control over tool lifecycle
# MAGIC
# MAGIC ### MCP Server Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────┐
# MAGIC │   Agent/Client   │
# MAGIC └────────┬─────────┘
# MAGIC          │
# MAGIC      MCP Protocol
# MAGIC      (stdio/HTTP)
# MAGIC          │
# MAGIC ┌────────▼─────────┐
# MAGIC │   MCP Server     │
# MAGIC │  (Your Code)     │
# MAGIC ├──────────────────┤
# MAGIC │ @app.list_tools()│
# MAGIC │ @app.call_tool() │
# MAGIC └────────┬─────────┘
# MAGIC          │
# MAGIC    External APIs
# MAGIC    or Business Logic
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Local MCP Server Testing (OPTIONAL - 10 min)
# MAGIC
# MAGIC **⏭️ Skip this section for Quick Start path.**
# MAGIC
# MAGIC ### Example: Calculator MCP Server
# MAGIC
# MAGIC Let's build a simple calculator MCP server with multiple math operations:

# COMMAND ----------

# MAGIC %pip install -q mcp

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Create MCP server
app = Server("calculator-mcp")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """Define all tools this MCP server provides."""
    return [
        Tool(
            name="add",
            description="Add two numbers together",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="multiply",
            description="Multiply two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["a", "b"]
            }
        ),
        Tool(
            name="power",
            description="Raise a number to a power",
            inputSchema={
                "type": "object",
                "properties": {
                    "base": {
                        "type": "number",
                        "description": "Base number"
                    },
                    "exponent": {
                        "type": "number",
                        "description": "Exponent"
                    }
                },
                "required": ["base", "exponent"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool invocations."""
    if name == "add":
        result = arguments["a"] + arguments["b"]
        return [TextContent(
            type="text",
            text=f"{arguments['a']} + {arguments['b']} = {result}"
        )]

    elif name == "multiply":
        result = arguments["a"] * arguments["b"]
        return [TextContent(
            type="text",
            text=f"{arguments['a']} × {arguments['b']} = {result}"
        )]

    elif name == "power":
        result = arguments["base"] ** arguments["exponent"]
        return [TextContent(
            type="text",
            text=f"{arguments['base']}^{arguments['exponent']} = {result}"
        )]

    else:
        return [TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]

# Entry point for running as a standalone server
if __name__ == "__main__":
    stdio_server(app)

print("✅ Calculator MCP server defined!")
print()
print("Server provides 3 tools:")
print("  - add: Add two numbers")
print("  - multiply: Multiply two numbers")
print("  - power: Raise to a power")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## Step 6: Deploy MCP Server to Production (15 min)
# MAGIC
# MAGIC ### 🎯 Start Here if You Skipped Steps 4-5
# MAGIC
# MAGIC This step shows you how to deploy a production-ready MCP server as a Databricks App.
# MAGIC We'll take the weather tool from Step 3 and deploy it.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Architecture: Local Development vs Production
# MAGIC
# MAGIC **What you built in Step 3:**
# MAGIC ```python
# MAGIC @tool
# MAGIC def get_weather(city: str) -> str:
# MAGIC     # Your tool code
# MAGIC ```
# MAGIC
# MAGIC **What we're deploying now:**
# MAGIC
# MAGIC Following the [official Databricks MCP app template](https://github.com/databricks/app-templates/tree/main/mcp-server-hello-world):
# MAGIC ```
# MAGIC my-weather-mcp-server/
# MAGIC ├── requirements.txt          ← just "uv" (the package manager)
# MAGIC ├── pyproject.toml            ← project deps (fastmcp, fastapi, uvicorn)
# MAGIC ├── app.yaml                  ← command: uv run weather-mcp-server
# MAGIC └── server/
# MAGIC     ├── tools.py              ← ★ your tool logic (add tools here!)
# MAGIC     ├── app.py                ← FastMCP + FastAPI setup
# MAGIC     └── main.py               ← uvicorn entrypoint
# MAGIC ```
# MAGIC
# MAGIC | Aspect | Local (Sections 1-2) | Production (This Section) |
# MAGIC |--------|---------------------|---------------------------|
# MAGIC | Transport | N/A | HTTP (streamable) |
# MAGIC | Framework | `@tool` | FastMCP + FastAPI |
# MAGIC | Package Manager | pip | uv (fast, reliable) |
# MAGIC | Access | Single notebook | Multi-user, authenticated |

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5.1: Create Project Structure

# COMMAND ----------

import os
import shutil

# Choose a name for your MCP server
PROJECT_NAME = "my-weather-mcp-server"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
PROJECT_DIR = f"/Workspace/Users/{username}/{PROJECT_NAME}"

# Clean slate — remove the entire project directory and recreate it.
# This prevents stale artifacts (.venv, pyproject.toml, requirements.txt, etc.)
# from interfering with the Databricks Apps builder.
if os.path.exists(PROJECT_DIR):
    shutil.rmtree(PROJECT_DIR)
    print(f"🧹 Cleaned previous project directory")

os.makedirs(f"{PROJECT_DIR}/server", exist_ok=True)
print(f"✅ Created project at: {PROJECT_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5.2: Configuration Files
# MAGIC
# MAGIC These files tell Databricks Apps how to run your server.
# MAGIC
# MAGIC **How Databricks Apps works:**
# MAGIC - The runtime is Python 3.11 in a dedicated virtual environment
# MAGIC - If `requirements.txt` exists, `pip install -r requirements.txt` runs at build time
# MAGIC - Many packages are pre-installed (fastapi, uvicorn, requests, databricks-sdk)
# MAGIC - The `app.yaml` command runs directly in this environment
# MAGIC
# MAGIC See: [Databricks Apps system environment](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/system-env)
# MAGIC
# MAGIC **How this works (same as [official template](https://github.com/databricks/app-templates/tree/main/mcp-server-hello-world)):**
# MAGIC - `requirements.txt` contains only `uv` — the builder pip-installs it at build time
# MAGIC - `app.yaml` runs `uv run weather-mcp-server` — uv reads `pyproject.toml` and
# MAGIC   installs the real dependencies (fastmcp, fastapi, uvicorn) automatically
# MAGIC - To add tools, just edit `server/tools.py` — same `@mcp_server.tool` pattern

# COMMAND ----------

# requirements.txt — only contains "uv" (the package manager).
# The builder pip-installs uv, then app.yaml uses uv to install everything else.
with open(f"{PROJECT_DIR}/requirements.txt", "w") as f:
    f.write("uv\n")

# app.yaml — tells Databricks Apps how to start the server.
# "uv run" reads pyproject.toml, installs deps, and runs the entry point.
with open(f"{PROJECT_DIR}/app.yaml", "w") as f:
    f.write('command: ["uv", "run", "weather-mcp-server"]\n')

# pyproject.toml — project metadata and dependencies.
# uv reads this to know what to install and what command to run.
with open(f"{PROJECT_DIR}/pyproject.toml", "w") as f:
    f.write("""\
[project]
name = "weather-mcp-server"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastmcp>=2.12.5",
    "fastapi>=0.115.12",
    "uvicorn>=0.34.2",
    "requests",
]

[project.scripts]
weather-mcp-server = "server.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["server"]
""")

print("✅ Created requirements.txt, app.yaml, pyproject.toml")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5.3: Server Implementation
# MAGIC
# MAGIC Three files, same layout as the
# MAGIC [official Databricks MCP template](https://github.com/databricks/app-templates/tree/main/mcp-server-hello-world):
# MAGIC
# MAGIC | File | Role |
# MAGIC |------|------|
# MAGIC | `server/tools.py` | **Add your tools here** — use `@mcp_server.tool` decorator |
# MAGIC | `server/app.py` | FastMCP + FastAPI setup (boilerplate, rarely edit) |
# MAGIC | `server/main.py` | Uvicorn entrypoint (boilerplate, rarely edit) |

# COMMAND ----------

# ── server/__init__.py ──
with open(f"{PROJECT_DIR}/server/__init__.py", "w") as f:
    f.write("")

# ── server/tools.py ── (★ This is the file you edit to add tools)
with open(f"{PROJECT_DIR}/server/tools.py", "w") as f:
    f.write('''\
"""MCP tool definitions — add your tools here."""
import requests


def load_tools(mcp_server):
    """Register all tools with the MCP server."""

    @mcp_server.tool
    def get_weather(city: str) -> str:
        """
        Get current weather conditions for a city.

        Args:
            city: City name (e.g., "San Francisco", "New York")
        """
        try:
            resp = requests.get(f"https://wttr.in/{city}?format=j1", timeout=5)
            if resp.status_code == 200:
                cur = resp.json()["current_condition"][0]
                desc = cur["weatherDesc"][0]["value"]
                return f"Weather in {city}: {desc}, {cur['temp_C']}C, {cur['humidity']}% humidity"
            return f"Could not fetch weather for {city}"
        except Exception as e:
            return f"Error: {e}"
''')

# ── server/app.py ── (FastMCP + FastAPI setup — rarely needs editing)
with open(f"{PROJECT_DIR}/server/app.py", "w") as f:
    f.write('''\
"""FastMCP server setup — follows the official Databricks app template."""
from fastmcp import FastMCP
from fastapi import FastAPI
from .tools import load_tools

mcp_server = FastMCP(name="weather-mcp-server")
load_tools(mcp_server)
mcp_app = mcp_server.http_app()

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "healthy"}

combined_app = FastAPI(
    routes=[*mcp_app.routes, *app.routes],
    lifespan=mcp_app.lifespan,
)
''')

# ── server/main.py ── (Uvicorn entrypoint — rarely needs editing)
with open(f"{PROJECT_DIR}/server/main.py", "w") as f:
    f.write('''\
"""Entry point for the MCP server."""
import os
import uvicorn

def main():
    port = int(os.environ.get("DATABRICKS_APP_PORT", os.environ.get("PORT", 8000)))
    uvicorn.run("server.app:combined_app", host="0.0.0.0", port=port)
''')

print("✅ Created server files")
print(f"\n📂 {PROJECT_NAME}/")
print(f"   ├── requirements.txt   # just 'uv'")
print(f"   ├── pyproject.toml     # deps: fastmcp, fastapi, uvicorn")
print(f"   ├── app.yaml           # uv run weather-mcp-server")
print(f"   └── server/")
print(f"       ├── tools.py       ← ★ add your tools here")
print(f"       ├── app.py         ← FastMCP + FastAPI setup")
print(f"       └── main.py        ← uvicorn entrypoint")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5.4: Verify Files Before Deploy

# COMMAND ----------

import os

files_to_check = [
    f"{PROJECT_DIR}/requirements.txt",
    f"{PROJECT_DIR}/pyproject.toml",
    f"{PROJECT_DIR}/app.yaml",
    f"{PROJECT_DIR}/server/__init__.py",
    f"{PROJECT_DIR}/server/tools.py",
    f"{PROJECT_DIR}/server/app.py",
    f"{PROJECT_DIR}/server/main.py",
]

print("Project files:")
for fp in files_to_check:
    status = "✅" if os.path.exists(fp) else "❌ MISSING"
    print(f"  {status}: {fp.split(PROJECT_NAME + '/')[-1]}")

# Warn about stale artifacts that break the builder
for stale in [".venv", "__pycache__", "uv.lock"]:
    if os.path.exists(os.path.join(PROJECT_DIR, stale)):
        print(f"  ⚠️  STALE: {stale} — re-run Step 5.1 to clean up")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5.5: Deploy Using Databricks SDK
# MAGIC
# MAGIC Deploy the app using the Databricks Python SDK:

# COMMAND ----------

print(f"📁 Project Directory: {PROJECT_DIR}")
print(f"📝 App Name: {PROJECT_NAME}")

# Remove stale build artifacts that can interfere with fresh deployments.
import os, shutil
for stale in [".venv", "__pycache__", "uv.lock"]:
    p = os.path.join(PROJECT_DIR, stale)
    if os.path.isdir(p):  shutil.rmtree(p); print(f"🧹 Removed {stale}")
    elif os.path.isfile(p): os.remove(p); print(f"🧹 Removed {stale}")

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import App, AppDeployment

w = WorkspaceClient()

# Create app
try:
    print(f"\n🚀 Attempting to create app '{PROJECT_NAME}'...")
    app = w.apps.create_and_wait(
        app=App(
            name=PROJECT_NAME,
            description="Weather MCP Server from Custom Tools notebook"
        )
    )
    print(f"✅ App created")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"ℹ️  App already exists (continuing...)")
    else:
        print(f"❌ Error creating app: {e}")
        raise

# Deploy app
try:
    print(f"\n📦 Deploying from {PROJECT_DIR}...")
    deployment = w.apps.deploy(
        app_name=PROJECT_NAME,
        app_deployment=AppDeployment(
            source_code_path=PROJECT_DIR
        )
    )
    print(f"✅ Deployment started: {deployment.deployment_id}")
    print(f"\n⏳ Monitor progress:")
    print(f"   - Databricks UI > Apps > {PROJECT_NAME}")
    print(f"   - Or check status in the next cell in ~2-3 minutes")
except Exception as e:
    print(f"\n❌ Deployment error: {e}")
    raise
# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5.6: Check Deployment Status
# MAGIC
# MAGIC **⏳ Wait 2-3 minutes** for deployment to complete, then run this cell to check status:

# COMMAND ----------

from databricks.sdk import WorkspaceClient

PROJECT_NAME = "my-weather-mcp-server"  # Must match above

w = WorkspaceClient()
app_info = w.apps.get(name=PROJECT_NAME)

print(f"App Name: {app_info.name}")
print(f"App URL: {app_info.url if hasattr(app_info, 'url') else 'Not yet available'}")

# Check deployment status
deployment_ready = False
if hasattr(app_info, 'active_deployment') and app_info.active_deployment:
    deployment = app_info.active_deployment

    # Try different status paths (SDK version compatibility)
    if hasattr(deployment, 'status') and deployment.status:
        state = deployment.status.state if hasattr(deployment.status, 'state') else str(deployment.status)
        print(f"Deployment Status: {state}")
        deployment_ready = state in ["SUCCEEDED", "RUNNING"]
    elif hasattr(deployment, 'deployment_status'):
        state = deployment.deployment_status.state
        print(f"Deployment Status: {state}")
        deployment_ready = state in ["SUCCEEDED", "RUNNING"]
    else:
        print(f"Deployment found but status format unknown")
        # If we have a URL, assume it's ready
        deployment_ready = hasattr(app_info, 'url') and app_info.url is not None

# Check compute/app status
if hasattr(app_info, 'compute_status') and app_info.compute_status:
    compute_state = app_info.compute_status.state if hasattr(app_info.compute_status, 'state') else str(app_info.compute_status)
    print(f"Compute Status: {compute_state}")

if deployment_ready and app_info.url:
    print("\n✅ App is ready! Proceed to the next cell to connect.")
    print(f"   MCP Endpoint will be: {app_info.url}/mcp")
elif app_info.url:
    print("\n⚠️ App URL exists but deployment status unclear. Try connecting anyway in the next cell.")
else:
    print("\n⏳ App is still deploying. Wait 1-2 minutes and re-run this cell.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5.7: Connect to Your Deployed Server
# MAGIC
# MAGIC Once deployment shows **SUCCEEDED** above, connect to the MCP server.
# MAGIC
# MAGIC #### Why OAuth Is Required
# MAGIC
# MAGIC Databricks Apps enforce OAuth authentication. The notebook's default session
# MAGIC token is not an OAuth token, so `WorkspaceClient()` alone cannot reach the
# MAGIC app. You need a **service principal** with OAuth M2M credentials.
# MAGIC
# MAGIC | Context | Auth Method | Works with MCP Apps? |
# MAGIC |---|---|---|
# MAGIC | Notebook default (`WorkspaceClient()`) | Session token | No |
# MAGIC | Service Principal (`client_id` + `client_secret`) | OAuth M2M | Yes |
# MAGIC | Agent Serving (on-behalf-of-user) | OAuth via `ModelServingUserCredentials` | Yes |
# MAGIC
# MAGIC #### One-Time Setup: Create a Service Principal
# MAGIC
# MAGIC 1. **Settings → Identity and access → Service principals → Add new** (name it `mcp-bootcamp-sp`)
# MAGIC 2. **Secrets tab → Generate secret** — copy both the Client ID and Client Secret immediately
# MAGIC 3. **Entitlements tab** — enable Workspace access
# MAGIC 4. **Compute → Apps → your app → Permissions** — add the SP with Can Use
# MAGIC
# MAGIC #### How the Code Below Works
# MAGIC
# MAGIC Two `WorkspaceClient` instances are used:
# MAGIC 1. Default client — looks up the app URL via notebook token
# MAGIC 2. OAuth client — authenticates as the service principal
# MAGIC
# MAGIC We use the low-level `streamablehttp_client` + `DatabricksOAuthClientProvider`
# MAGIC instead of `DatabricksMCPClient` to avoid a known `httpx` issue in notebooks.
# MAGIC
# MAGIC For production, store SP credentials in **Databricks Secrets** instead of widgets.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Enter your Service Principal credentials
# MAGIC Run this cell to create the input widgets, then fill them in at the **top of the notebook**.

# COMMAND ----------

import nest_asyncio
nest_asyncio.apply()

dbutils.widgets.text("sp_client_id", "", "Service Principal Client ID")
dbutils.widgets.text("sp_client_secret", "", "Service Principal Client Secret")
print("⬆️  Fill in the widgets at the top of this notebook, then run the next cell.")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksOAuthClientProvider
from mcp.client.streamable_http import streamablehttp_client as connect
from mcp import ClientSession

PROJECT_NAME = "my-weather-mcp-server"  # Must match the app name from Step 5.5

# ── Step 1: Look up the app URL using the notebook's default auth ──
w = WorkspaceClient()
app_info = w.apps.get(name=PROJECT_NAME)
mcp_server_url = f"{app_info.url}/mcp"
print(f"🌐 MCP server URL: {mcp_server_url}")

# ── Step 2: Read the service principal credentials from the widgets ──
sp_client_id = dbutils.widgets.get("sp_client_id")
sp_client_secret = dbutils.widgets.get("sp_client_secret")

if not sp_client_id or not sp_client_secret:
    raise ValueError(
        "⚠️ Service principal credentials are required!\n"
        "Fill in the 'sp_client_id' and 'sp_client_secret' widgets at the top of the notebook.\n"
        "See the instructions in Step 5.7 above to create a service principal."
    )

# ── Step 3: Create an OAuth M2M WorkspaceClient ──
# NOTE: w.config.host inside a notebook returns the regional URL (e.g. eastus2.azuredatabricks.net)
# which doesn't support SP OAuth. Use the canonical workspace URL instead.
workspace_url = f"https://{spark.conf.get('spark.databricks.workspaceUrl')}"
oauth_client = WorkspaceClient(
    host=workspace_url,
    client_id=sp_client_id,
    client_secret=sp_client_secret,
)
print("✅ OAuth M2M WorkspaceClient created")

# ── Step 4: Connect to the MCP server ──
# Use the low-level streamable HTTP client with DatabricksOAuthClientProvider.
# DatabricksMCPClient.list_tools() has a known issue with httpx in notebooks,
# so we use the underlying MCP protocol directly.
async def test_mcp_connection():
    auth = DatabricksOAuthClientProvider(oauth_client)
    async with connect(mcp_server_url, auth=auth) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tools = tools_result.tools
            print(f"\n✅ Available tools: {[t.name for t in tools]}")

            result = await session.call_tool("get_weather", {"city": "Tokyo"})
            print(f"\n🌤️ Test Result:\n{result.content[0].text}")
            return tools

import asyncio
mcp_tools = asyncio.run(test_mcp_connection())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5.8: Use in Your Agent
# MAGIC
# MAGIC To use the deployed MCP server from a LangChain agent, wrap each MCP tool
# MAGIC as a `@tool` function that calls the server over the streamable HTTP transport.
# MAGIC The agent sees a normal LangChain tool; the MCP call happens inside.

# COMMAND ----------

from databricks_langchain import ChatDatabricks
from langchain_core.tools import tool as langchain_tool
from langchain_core.messages import HumanMessage

# Wrap each MCP tool as a LangChain tool.
# We call the MCP server over the streamable HTTP transport inside each tool.
async def call_mcp_tool(tool_name: str, args: dict) -> str:
    auth = DatabricksOAuthClientProvider(oauth_client)
    async with connect(mcp_server_url, auth=auth) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            return result.content[0].text

@langchain_tool
def get_weather(city: str) -> str:
    """Get current weather conditions for a city."""
    import asyncio
    return asyncio.get_event_loop().run_until_complete(
        call_mcp_tool("get_weather", {"city": city})
    )

langchain_tools = [get_weather]

llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-6")
llm_with_tools = llm.bind_tools(langchain_tools)

response = llm_with_tools.invoke([
    HumanMessage(content="What's the weather in Paris?")
])

print("Question: What's the weather in Paris?")
print(f"Answer: {response.content}")

if hasattr(response, 'tool_calls') and response.tool_calls:
    for tool_call in response.tool_calls:
        result = get_weather.invoke(tool_call['args'])
        print(f"\nTool Result: {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary & Next Steps
# MAGIC
# MAGIC ### What You Learned:
# MAGIC
# MAGIC **Core Concepts (Steps 2-3):**
# MAGIC - ✅ When to build custom tools vs use managed MCPs
# MAGIC - ✅ Build simple tools with `@tool` decorator
# MAGIC - ✅ Add tools to agents
# MAGIC
# MAGIC **Production Deployment (Step 6):**
# MAGIC - ✅ Create FastMCP server (official Databricks template pattern)
# MAGIC - ✅ Deploy as Databricks App with uv
# MAGIC - ✅ Connect using DatabricksMCPClient
# MAGIC - ✅ Use in production agents
# MAGIC
# MAGIC **Optional Deep Dive (Steps 4-5):**
# MAGIC - ⚙️ MCP server architecture concepts
# MAGIC - ⚙️ Local MCP server development and testing
# MAGIC
# MAGIC ### The Pattern:
# MAGIC
# MAGIC ```python
# MAGIC # Development: Simple tool
# MAGIC @tool
# MAGIC def my_tool(param: str) -> str:
# MAGIC     return do_something(param)
# MAGIC
# MAGIC # Production: Add to server/tools.py
# MAGIC @mcp_server.tool
# MAGIC def my_tool(param: str) -> str:
# MAGIC     return do_something(param)  # Same logic!
# MAGIC ```
# MAGIC
# MAGIC ### When to Use Each:
# MAGIC
# MAGIC | Scenario | Approach |
# MAGIC |----------|----------|
# MAGIC | Single external API | Simple `@tool` |
# MAGIC | Multiple related tools | MCP Server |
# MAGIC | Databricks data | Managed MCP (Genie, SQL) |
# MAGIC | Complex state/lifecycle | MCP Server |
# MAGIC | Need reusability | MCP Server |
# MAGIC | Production deployment | FastMCP as Databricks App |
# MAGIC
# MAGIC ### Complete Agent Architecture:
# MAGIC
# MAGIC ```
# MAGIC User Query
# MAGIC      ↓
# MAGIC   Agent
# MAGIC      ↓
# MAGIC ┌──────────┬──────────┬──────────┬──────────┐
# MAGIC │  Vector  │  Genie   │ Weather  │ Custom   │
# MAGIC │  Search  │   MCP    │   MCP    │   MCP    │
# MAGIC │(Managed) │(Managed) │(Deployed)│(Deployed)│
# MAGIC └──────────┴──────────┴──────────┴──────────┘
# MAGIC      ↓           ↓          ↓          ↓
# MAGIC     Docs       Data     External   Your
# MAGIC                         API       Logic
# MAGIC ```
# MAGIC
# MAGIC ### Next Steps:
# MAGIC - Add more tools to your MCP server
# MAGIC - Explore MCP server concepts (Sections 3-4) if you skipped them
# MAGIC - Continue to next module: Evaluation & Production Deployment
# MAGIC - Explore: [Databricks MCP Documentation](https://docs.databricks.com/aws/en/generative-ai/mcp/)