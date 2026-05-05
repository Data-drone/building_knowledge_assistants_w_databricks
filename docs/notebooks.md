# Notebooks

All notebooks in bootcamp order. Each runs in Databricks on MLR 17.3 LTS or higher.

---

<div class="notebook-grid" markdown>

<div class="notebook-card" markdown>
<div class="notebook-header">
<span class="notebook-tag">Foundations</span>
<span class="badge">15 min</span>
</div>

### 00_setup.py

**Outcome:** Workspace ready with all shared assets — catalog, schema, tables, Lakebase, UC function.

**Prerequisites:** Databricks workspace, Unity Catalog permissions, Lakebase access

<div class="notebook-actions" markdown>
[Open notebook :material-open-in-new:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/00_foundations/00_setup.py){ .nb-link }
</div>
</div>

<div class="notebook-card" markdown>
<div class="notebook-header">
<span class="notebook-tag">RAG</span>
<span class="badge">15 min</span>
</div>

### 00_your_first_agent_on_databricks.py

**Outcome:** Working LangGraph agent connected to a Databricks-hosted LLM with MLflow auto-tracing.

**Prerequisites:** LLM endpoint available

<div class="notebook-actions" markdown>
[Open notebook :material-open-in-new:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/01_rag_pipeline/00_your_first_agent_on_databricks.py){ .nb-link }
</div>
</div>

<div class="notebook-card" markdown>
<div class="notebook-header">
<span class="notebook-tag">RAG</span>
<span class="badge">25 min</span>
</div>

### 01_building_a_doc_store_on_vector_search.py

**Outcome:** Document chunks in Delta, synced Vector Search index, RAG agent answering from documents.

**Prerequisites:** Setup notebook complete, documents in volume

<div class="notebook-actions" markdown>
[Open notebook :material-open-in-new:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/01_rag_pipeline/01_building_a_doc_store_on_vector_search.py){ .nb-link }
</div>
</div>

<div class="notebook-card" markdown>
<div class="notebook-header">
<span class="notebook-tag">Memory</span>
<span class="badge">20 min</span>
</div>

### 01_short_term_memory.py

**Outcome:** Agent with multi-turn conversation memory via Lakebase CheckpointSaver.

**Prerequisites:** Vector Search index, Lakebase project

<div class="notebook-actions" markdown>
[Open notebook :material-open-in-new:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/02_memory/01_short_term_memory.py){ .nb-link }
</div>
</div>

<div class="notebook-card" markdown>
<div class="notebook-header">
<span class="notebook-tag">Memory</span>
<span class="badge">30 min</span>
</div>

### 02_long_term_memory.py

**Outcome:** Agent with LLM-managed memory tools (get/save/delete) and per-user preference storage.

**Prerequisites:** Short-term memory notebook, Lakebase

<div class="notebook-actions" markdown>
[Open notebook :material-open-in-new:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/02_memory/02_long_term_memory.py){ .nb-link }
</div>
</div>

<div class="notebook-card" markdown>
<div class="notebook-header">
<span class="notebook-tag">Evaluation</span>
<span class="badge">15 min</span>
</div>

### 00_first_eval_loop.py

**Outcome:** Simple evaluation loop with MLflow eval rows and basic built-in scorers.

**Prerequisites:** Working agent from RAG module

<div class="notebook-actions" markdown>
[Open notebook :material-open-in-new:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/03_evaluation/00_first_eval_loop.py){ .nb-link }
</div>
</div>

<div class="notebook-card" markdown>
<div class="notebook-header">
<span class="notebook-tag">Evaluation</span>
<span class="badge">15 min</span>
</div>

### 01_tracing.py

**Outcome:** Deep inspection of one agent run — spans, latency, token counts, tool calls.

**Prerequisites:** First eval loop notebook

<div class="notebook-actions" markdown>
[Open notebook :material-open-in-new:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/03_evaluation/01_tracing.py){ .nb-link }
</div>
</div>

<div class="notebook-card" markdown>
<div class="notebook-header">
<span class="notebook-tag">Evaluation</span>
<span class="badge">45 min</span>
</div>

### 02_evaluation.py

**Outcome:** Full evaluation pipeline with built-in scorers, custom judges, agent behavior checks, and quality gates.

**Prerequisites:** Tracing notebook complete

<div class="notebook-actions" markdown>
[Open notebook :material-open-in-new:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/03_evaluation/02_evaluation.py){ .nb-link }
</div>
</div>

<div class="notebook-card" markdown>
<div class="notebook-header">
<span class="notebook-tag">Data Tools</span>
<span class="badge">10 min</span>
</div>

### 00_setup.py (MCP)

**Outcome:** Genie space created, employee data ready for natural language queries.

**Prerequisites:** Setup notebook, employee tables

<div class="notebook-actions" markdown>
[Open notebook :material-open-in-new:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/04_mcp_tool_integration/00_setup.py){ .nb-link }
</div>
</div>

<div class="notebook-card" markdown>
<div class="notebook-header">
<span class="notebook-tag">Data Tools</span>
<span class="badge">35 min</span>
</div>

### 01_genie_integration.py

**Outcome:** Agent with Genie for natural language SQL, multi-tool routing between documents and data.

**Prerequisites:** Genie space created, Vector Search index

<div class="notebook-actions" markdown>
[Open notebook :material-open-in-new:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/04_mcp_tool_integration/01_genie_integration.py){ .nb-link }
</div>
</div>

<div class="notebook-card" markdown>
<div class="notebook-header">
<span class="notebook-tag">Data Tools</span>
<span class="badge">30 min</span>
</div>

### 02_custom_tools.py

**Outcome:** Custom MCP tools via UC functions, advanced tool patterns.

**Prerequisites:** Genie integration notebook

<div class="notebook-actions" markdown>
[Open notebook :material-open-in-new:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/04_mcp_tool_integration/02_custom_tools.py){ .nb-link }
</div>
</div>

<div class="notebook-card" markdown>
<div class="notebook-header">
<span class="notebook-tag">Deployment</span>
<span class="badge">10 min</span>
</div>

### 01_packaging_for_apps.py

**Outcome:** Understanding of the Apps development loop — sync, iterate, test locally.

**Prerequisites:** Completed agent from earlier modules

<div class="notebook-actions" markdown>
[Open notebook :material-open-in-new:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/05_deployment/01_packaging_for_apps.py){ .nb-link }
</div>
</div>

<div class="notebook-card" markdown>
<div class="notebook-header">
<span class="notebook-tag">Deployment</span>
<span class="badge">15 min</span>
</div>

### 02_deploy_and_validate.py

**Outcome:** Live Databricks App with validated `/invocations` endpoint.

**Prerequisites:** Packaging notebook, Databricks CLI

<div class="notebook-actions" markdown>
[Open notebook :material-open-in-new:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/05_deployment/02_deploy_and_validate.py){ .nb-link }
</div>
</div>

<div class="notebook-card" markdown>
<div class="notebook-header">
<span class="notebook-tag">Deployment</span>
<span class="badge">20 min</span>
</div>

### 03_production_monitoring.py

**Outcome:** Registered scorers running continuously on production traces with configurable sampling.

**Prerequisites:** Deployed app, MLflow experiment

<div class="notebook-actions" markdown>
[Open notebook :material-open-in-new:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/05_deployment/03_production_monitoring.py){ .nb-link }
</div>
</div>

<div class="notebook-card" markdown>
<div class="notebook-header">
<span class="notebook-tag">Deployment</span>
<span class="badge">15 min</span>
</div>

### 04_end_to_end_deployment.py

**Outcome:** Full release validation — memory continuity, isolation, monitoring, quality gates.

**Prerequisites:** Production monitoring set up

<div class="notebook-actions" markdown>
[Open notebook :material-open-in-new:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/05_deployment/04_end_to_end_deployment.py){ .nb-link }
</div>
</div>

</div>
