# Setup

Everything you need before running the first notebook.

---

## Databricks workspace

<div class="setup-grid" markdown>

<div class="setup-card" markdown>
### :material-server-network: Runtime
MLR 17.3 LTS or higher. Single-user cluster recommended for the bootcamp.
</div>

<div class="setup-card" markdown>
### :material-shield-check-outline: Permissions
- Create catalogs and schemas
- Create Vector Search indexes
- Create Lakebase projects
- Deploy Databricks Apps
- Access Foundation Model endpoints
</div>

<div class="setup-card" markdown>
### :material-console: CLI setup
Install the Databricks CLI and authenticate:
```bash
pip install databricks-cli
databricks configure --token
```
</div>

<div class="setup-card" markdown>
### :material-key-outline: Secrets & env vars
The app uses environment variables (set in `app.yaml`):
- `LLM_ENDPOINT`
- `CATALOG` / `SCHEMA`
- `LAKEBASE_AUTOSCALING_PROJECT`
- `LAKEBASE_AUTOSCALING_BRANCH`
</div>

</div>

---

## Required services

| Service | Used for | Setup |
|---|---|---|
| **Unity Catalog** | Governance, tables, volumes, functions | Enabled by admin |
| **Vector Search** | Document similarity search | Create index in notebook 01 |
| **Genie** | Natural language SQL | Create space in notebook 04 |
| **Lakebase** | Agent memory (checkpointer + store) | Create project in notebook 00 |
| **MLflow 3.0+** | Tracing, evaluation, monitoring | Pre-installed on MLR 17.3+ |
| **Foundation Models** | LLM reasoning + embeddings | Endpoint must be `ONLINE` |
| **Databricks Apps** | Production deployment | Requires Apps access |

---

## Configuration

All shared settings live in `config.py` at the repo root:

```python
# Update these for your workspace
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
HOST = "https://your-workspace.cloud.databricks.com"
REGION = "us-west-2"
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
LAKEBASE_PROJECT = "knowledge-assistant-state"
LAKEBASE_BRANCH = "development"
```

See the full config reference in [`config.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/config.py).

---

## Clone and start

```bash
git clone https://github.com/Data-drone/building_knowledge_assistants_w_databricks.git
cd building_knowledge_assistants_w_databricks
```

Edit `config.py` with your workspace values, then open `00_foundations/00_setup.py` in Databricks and run all cells.

---

## Project structure

```
building_knowledge_assistants_w_databricks/
├── config.py                       # Shared configuration
├── 00_foundations/                  # Platform setup
├── 01_rag_pipeline/                # Vector Search + RAG agent
├── 02_memory/                      # Lakebase memory
├── 03_evaluation/                  # MLflow tracing + eval
├── 04_mcp_tool_integration/        # Genie, SQL, custom tools
├── 05_deployment/                  # Apps deploy + monitoring
└── apps/
    └── knowledge_assistant_agent/  # Production app source
```
