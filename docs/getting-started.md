# Getting Started

## Prerequisites

### Databricks Workspace

- **Runtime:** MLR 17.3 LTS or higher
- Unity Catalog enabled
- Lakebase access
- Genie access
- Foundation Model endpoints available

### Permissions

- Create catalogs and schemas
- Create Vector Search indexes
- Create Lakebase projects
- Deploy Databricks Apps

### Knowledge

- Python programming
- Basic LangChain / LangGraph concepts
- Familiarity with the ReAct agent pattern

## Setup Instructions

!!! warning "Follow these steps in order"

    Each module builds on the previous one. Start with Module 0 before continuing.

### 1. Clone the repository

```bash
git clone https://github.com/Data-drone/building_knowledge_assistants_w_databricks.git databricks_agent_bootcamp
cd databricks_agent_bootcamp
```

### 2. Update `config.py`

Open `config.py` and set values for your workspace:

```python
CATALOG = "agent_bootcamp"                     # Your catalog name
SCHEMA = "knowledge_assistant"                 # Your schema name
HOST = "https://your-workspace.cloud.databricks.com"
REGION = "us-west-2"                           # Your cloud region
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
LAKEBASE_PROJECT = "knowledge-assistant-state"
LAKEBASE_BRANCH = "development"
GENIE_SPACE_ID = ""                            # Fill after Module 04 setup
```

Later modules carry small local config blocks for teaching clarity, so you don't need to expand `config.py` for every notebook.

### 3. Run the setup notebook

Open [`00_foundations/00_setup.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/00_foundations/00_setup.py) in your Databricks workspace and run all cells. This creates:

- Unity Catalog catalog and schema
- Sample policy documents in a Volume
- Employee and leave balance tables
- A governed UC function (`get_employee`)
- Lakebase project with development and production branches

Wait for the notebook to complete before moving to Module 1.

### 4. Follow the learning path

| Module | What You'll Build | Duration |
|---|---|---|
| [0: Foundations](modules/00-foundations.md) | Shared assets for the bootcamp | 15 min |
| [1: RAG Pipeline](modules/01-rag-pipeline.md) | Agent that answers from documents | 40 min |
| [2: Memory](modules/02-memory.md) | Multi-turn and cross-session memory | 50 min |
| [3: Evaluation](modules/03-evaluation.md) | Tracing, scorers, and quality gates | 75 min |
| [4: Data Tools](modules/04-data-tools.md) | Genie, SQL, and custom MCP tools | 75 min |
| [5: Deployment](modules/05-deployment.md) | Production Databricks App | 60 min |

## Project Structure

```
databricks_agent_bootcamp/
├── config.py                       # Shared configuration
├── 00_foundations/                  # Platform basics
├── 01_rag_pipeline/                # Vector Search + documents
├── 02_memory/                      # Lakebase checkpointing
├── 03_evaluation/                  # Tracing, scorers, judges
├── 04_mcp_tool_integration/        # SQL, Genie, and custom tools
├── 05_deployment/                  # Production monitoring + eval
└── apps/
    └── knowledge_assistant_agent/  # Canonical Databricks Apps runtime
```

!!! tip "Deployment note"

    Module 5 uses direct Databricks Apps commands (`databricks sync` + `databricks apps deploy`) from `apps/knowledge_assistant_agent`. The repo's `databricks.yml` is only a minimal workspace/extension config, not the main deployment workflow.
