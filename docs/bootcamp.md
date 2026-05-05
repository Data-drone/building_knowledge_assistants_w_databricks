# Bootcamp

A guided sequence from workspace setup to production deployment. Follow the steps in order — each builds on the previous.

---

<div class="bootcamp-steps" markdown>

<div class="step-card" markdown>
<span class="step-num">01</span>

### Setup workspace

Create Unity Catalog assets, Lakebase project, sample data, and a governed UC function.

**Notebook:** [`00_foundations/00_setup.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/00_foundations/00_setup.py)

**You'll have:** Catalog, schema, volumes, tables, Lakebase branches, UC function

<span class="badge">15 min</span>
</div>

<div class="step-card" markdown>
<span class="step-num">02</span>

### Create knowledge sources

Chunk documents, create embeddings with Delta Sync, and build a Vector Search index.

**Notebook:** [`01_rag_pipeline/01_building_a_doc_store_on_vector_search.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/01_rag_pipeline/01_building_a_doc_store_on_vector_search.py)

**You'll have:** Delta table of chunks, synced Vector Search index

<span class="badge">25 min</span>
</div>

<div class="step-card" markdown>
<span class="step-num">03</span>

### Build the assistant

Create a LangGraph agent with `VectorSearchRetrieverTool` for document Q&A.

**Notebooks:**

- [`01_rag_pipeline/00_your_first_agent_on_databricks.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/01_rag_pipeline/00_your_first_agent_on_databricks.py)
- [`01_rag_pipeline/01_building_a_doc_store_on_vector_search.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/01_rag_pipeline/01_building_a_doc_store_on_vector_search.py)

**You'll have:** Working RAG agent answering from documents

<span class="badge">40 min</span>
</div>

<div class="step-card" markdown>
<span class="step-num">04</span>

### Add structured data & Genie

Connect Genie for natural-language SQL, add custom MCP tools, and enable multi-tool routing.

**Notebooks:**

- [`04_mcp_tool_integration/00_setup.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/04_mcp_tool_integration/00_setup.py)
- [`04_mcp_tool_integration/01_genie_integration.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/04_mcp_tool_integration/01_genie_integration.py)
- [`04_mcp_tool_integration/02_custom_tools.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/04_mcp_tool_integration/02_custom_tools.py)

**You'll have:** Agent with document search + SQL + custom tools

<span class="badge">75 min</span>
</div>

<div class="step-card" markdown>
<span class="step-num">05</span>

### Add memory

Add short-term conversation memory (CheckpointSaver) and long-term user preferences (DatabricksStore with LLM-managed tools).

**Notebooks:**

- [`02_memory/01_short_term_memory.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/02_memory/01_short_term_memory.py)
- [`02_memory/02_long_term_memory.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/02_memory/02_long_term_memory.py)

**You'll have:** Agent with multi-turn context and persistent user memory

<span class="badge">50 min</span>
</div>

<div class="step-card" markdown>
<span class="step-num">06</span>

### Evaluate with MLflow

Trace agent runs, score quality with built-in and custom judges, set up quality gates.

**Notebooks:**

- [`03_evaluation/00_first_eval_loop.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/03_evaluation/00_first_eval_loop.py)
- [`03_evaluation/01_tracing.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/03_evaluation/01_tracing.py)
- [`03_evaluation/02_evaluation.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/03_evaluation/02_evaluation.py)

**You'll have:** Evaluation pipeline with scorers, judges, and pass/fail thresholds

<span class="badge">75 min</span>
</div>

<div class="step-card" markdown>
<span class="step-num">07</span>

### Deploy as a Databricks App

Package the agent, deploy with `databricks apps deploy`, validate the endpoint, and register production monitors.

**Notebooks:**

- [`05_deployment/01_packaging_for_apps.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/05_deployment/01_packaging_for_apps.py)
- [`05_deployment/02_deploy_and_validate.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/05_deployment/02_deploy_and_validate.py)
- [`05_deployment/03_production_monitoring.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/05_deployment/03_production_monitoring.py)

**You'll have:** Live Databricks App with API endpoint and chat UI

<span class="badge">45 min</span>
</div>

<div class="step-card" markdown>
<span class="step-num">08</span>

### Debug & operate

Run end-to-end validation, test memory behavior, verify monitoring, and iterate.

**Notebook:** [`05_deployment/04_end_to_end_deployment.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/05_deployment/04_end_to_end_deployment.py)

**You'll have:** Validated production deployment with quality assurance

<span class="badge">15 min</span>
</div>

</div>
