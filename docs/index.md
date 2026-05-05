---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# Databricks Agent Bootcamp

Build a production-ready internal Knowledge Assistant using Databricks Apps, Agent Framework, Vector Search, Genie, MLflow, and Lakebase memory.

<div class="hero-ctas" markdown>

[Start the bootcamp](bootcamp.md){ .md-button .md-button--primary }
[View notebooks](notebooks.md){ .md-button }
[Open GitHub :material-github:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks){ .md-button }

</div>

<div class="hero-badges" markdown>

<span class="badge">Intermediate</span>
<span class="badge">~2 hours</span>
<span class="badge">Notebook-first</span>
<span class="badge">Production patterns</span>

</div>

</div>

---

## What you'll build

<div class="build-grid" markdown>

<div class="build-card" markdown>
### :material-robot-outline: Internal Knowledge Assistant
An agent that answers employee questions from policy documents and structured data — deployed as a Databricks App with full observability.
</div>

<div class="build-card" markdown>
### :material-file-search-outline: Document Q&A via Vector Search
Answers from unstructured documents using Delta Sync embeddings and `VectorSearchRetrieverTool`.
</div>

<div class="build-card" markdown>
### :material-database-search-outline: Structured Data via Genie
Natural language queries over employee tables and leave balances — Genie converts questions to SQL.
</div>

<div class="build-card" markdown>
### :material-head-cog-outline: Lakebase Memory
Short-term context with CheckpointSaver. Long-term user preferences with DatabricksStore and LLM-managed memory tools.
</div>

<div class="build-card" markdown>
### :material-rocket-launch-outline: Databricks Apps Deployment
Production endpoint with `/invocations` API, built-in chat UI, environment-based config, and OAuth authentication.
</div>

<div class="build-card" markdown>
### :material-chart-line: MLflow Observability
End-to-end tracing, built-in LLM judges, custom scorers, and registered production monitors with configurable sampling.
</div>

</div>

---

## Bootcamp flow

<div class="flow-sequence" markdown>

<div class="flow-step" markdown>
<span class="flow-num">01</span>
**Setup workspace** — Unity Catalog, Lakebase, sample data
</div>

<div class="flow-step" markdown>
<span class="flow-num">02</span>
**Create knowledge sources** — Document chunking, Vector Search index
</div>

<div class="flow-step" markdown>
<span class="flow-num">03</span>
**Build the assistant** — LangGraph agent with RAG tools
</div>

<div class="flow-step" markdown>
<span class="flow-num">04</span>
**Add structured data** — Genie spaces, MCP tools, multi-tool routing
</div>

<div class="flow-step" markdown>
<span class="flow-num">05</span>
**Add memory** — Short-term checkpoints, long-term user preferences
</div>

<div class="flow-step" markdown>
<span class="flow-num">06</span>
**Evaluate with MLflow** — Tracing, scorers, judges, quality gates
</div>

<div class="flow-step" markdown>
<span class="flow-num">07</span>
**Deploy as a Databricks App** — Packaging, validation, monitoring
</div>

<div class="flow-step" markdown>
<span class="flow-num">08</span>
**Debug & operate** — Production monitoring, troubleshooting, iteration
</div>

</div>

[Start with Step 01 :material-arrow-right:](bootcamp.md){ .md-button .md-button--primary }

---

## Quick start

Add the repo directly to your Databricks workspace — no local clone needed:

**Workspace → Repos → Add Repo** → paste `https://github.com/Data-drone/building_knowledge_assistants_w_databricks.git`

All notebooks are runnable immediately. See [Setup](setup.md) for full instructions.

---

## Who is this for?

**Developers** familiar with Python and LangChain/LangGraph who want to build production agents on Databricks. You should understand the ReAct pattern and have access to a Databricks workspace with Unity Catalog, Lakebase, and Foundation Model endpoints.

---

<div class="footer-links" markdown>

[:material-github: GitHub](https://github.com/Data-drone/building_knowledge_assistants_w_databricks) · [:material-book-open-variant: Databricks Docs](https://docs.databricks.com/) · [:material-flask: MLflow Docs](https://mlflow.org/docs/latest/)

</div>
