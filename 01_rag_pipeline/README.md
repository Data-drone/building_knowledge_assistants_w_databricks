# RAG Pipeline - Quick Start

Build your first AI agents on Databricks in under 40 minutes!

## What You'll Learn

This module teaches you to build AI agents using Databricks primitives:
1. **Notebook 00** (15 min): Your first agent - LLM + conversation loop
2. **Notebook 01** (20-25 min): Add Vector Search tool - RAG agent

## Prerequisites

- Access to Databricks workspace
- LLM endpoint available (`databricks-claude-sonnet-4-6` or similar)
- For Notebook 01: Unity Catalog volume with sample documents

## Learning Path

### Start Here: 00_your_first_agent_on_databricks.py

**What you'll build**: A simple chatbot using LangGraph

**Key concepts**:
- Using `databricks_langchain` to call LLMs
- Building conversation loops with LangGraph
- MLflow auto-logging for observability
- Understanding what makes something an "agent"

**Time**: 15 minutes

---

### Next: 01_building_a_doc_store_on_vector_search.py

**What you'll build**: RAG agent that answers from your documents

**Key concepts**:
- Document chunking for Vector Search
- Creating Delta Sync indexes
- Wrapping Vector Search as a LangChain tool
- RAG pattern (Retrieval + Generation)

**Time**: 20-25 minutes

---

## What's Next?

After completing these notebooks, you'll understand:
- ✅ How agents work (LLM + loop + tools)
- ✅ How to use Databricks LLM endpoints
- ✅ How to build with LangGraph
- ✅ How Vector Search enables RAG

### Continue Learning

**For deeper understanding**:
- Chunking strategies → See other tutorial sections
- Vector Search optimization → See other tutorial sections
- Advanced agent patterns → See other tutorial sections

**For production features** (covered in later modules):
- **Module 02**: Conversation memory with Lakebase
- **Module 03**: Evaluation and metrics
- **Module 04**: Multi-tool agents (Genie, UC Functions)
- **Module 05**: Deployment to Databricks Apps

---

## Quick Start

1. Open `00_your_first_agent_on_databricks.py`
2. Run all cells
3. Chat with your first agent!
4. Continue to `01_building_a_doc_store_on_vector_search.py`
5. Add Vector Search and build a RAG agent

---

## Technical Stack

- **LLM**: Databricks LLM endpoints (Claude Sonnet 4.6)
- **Orchestration**: LangGraph
- **Vector Search**: Databricks Vector Search with Delta Sync
- **Observability**: MLflow auto-logging
- **Storage**: Unity Catalog (Delta tables, Volumes)

---

## Query Deployed App From A Notebook

If you have deployed the agent as a Databricks App, use OAuth auth (not
`dbutils.notebook...apiToken()`).

### 1) Store an OAuth token in Databricks Secrets (run in terminal or `%sh`)

```bash
TOKEN=$(databricks auth token --profile adb-984752964297111 -o json | jq -r '.access_token // .token_value // .token')
databricks secrets create-scope my-secrets --profile adb-984752964297111 || true
databricks secrets put-secret my-secrets apps_oauth_token \
  --string-value "$TOKEN" \
  --profile adb-984752964297111
```

### 2) Invoke `/invocations` from a notebook cell

```python
import requests

APP_URL = "https://knowledge-assistant-agent-app-984752964297111.11.azure.databricksapps.com"
OAUTH_TOKEN = dbutils.secrets.get("my-secrets", "apps_oauth_token")

resp = requests.post(
    f"{APP_URL}/invocations",
    headers={
        "Authorization": f"Bearer {OAUTH_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    },
    json={"input": [{"role": "user", "content": "What is Databricks MCP in one sentence?"}]},
    timeout=60,
    allow_redirects=False,
)

print("Status:", resp.status_code)
print("Content-Type:", resp.headers.get("content-type"))
print(resp.text)
```

If you get HTML with a sign-in page, your token is not a valid OAuth token for
Databricks Apps.

---

## Need Help?

- Check MLflow traces for debugging
- Review error messages (version pins prevent most issues)
- Ensure endpoints are ONLINE before querying

---

**Total Time**: 35-40 minutes from zero to RAG agent 🚀
