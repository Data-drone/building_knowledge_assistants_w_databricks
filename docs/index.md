# Databricks Agent Bootcamp

<div class="db-hero" markdown>

## Build Production-Ready AI Agents on Databricks

A hands-on tutorial series that takes you from zero to a deployed **Internal Knowledge Assistant** — answering questions from documents and structured data, with conversation memory and production monitoring.

**2 hours hands-on** · **6 modules** · **Intermediate level**

[Get Started :material-arrow-right:](getting-started.md){ .md-button .md-button--primary }
[View on GitHub :material-github:](https://github.com/Data-drone/building_knowledge_assistants_w_databricks){ .md-button }

</div>

## What You'll Build

An **Internal Knowledge Assistant** that:

- :material-file-search: Answers policy questions from documents (Vector Search)
- :material-database-search: Queries employee data with natural language (Genie)
- :material-chat-processing: Maintains conversation context across turns (Lakebase Checkpointer)
- :material-account-heart: Remembers user preferences across sessions (Lakebase Store)
- :material-chart-line: Traces all execution for observability (MLflow)
- :material-check-decagram: Evaluates quality with built-in and custom judges
- :material-rocket-launch: Deploys to production as a Databricks App

## Learning Path

<div class="module-grid" markdown>

<div class="module-card" markdown>
<span class="module-num">MODULE 0</span>
### Foundations <span class="badge-duration">15 min</span>
Set up Unity Catalog, Lakebase, sample data, and a governed UC function.

[:material-arrow-right: Start here](modules/00-foundations.md)
</div>

<div class="module-card" markdown>
<span class="module-num">MODULE 1</span>
### RAG Pipeline <span class="badge-duration">40 min</span>
Build your first agent and add document-based Q&A with Vector Search.

[:material-arrow-right: Build RAG](modules/01-rag-pipeline.md)
</div>

<div class="module-card" markdown>
<span class="module-num">MODULE 2</span>
### Memory <span class="badge-duration">50 min</span>
Add short-term conversation memory and long-term user preferences.

[:material-arrow-right: Add memory](modules/02-memory.md)
</div>

<div class="module-card" markdown>
<span class="module-num">MODULE 3</span>
### Evaluation <span class="badge-duration">75 min</span>
Observe, trace, and evaluate agent quality with MLflow scorers and judges.

[:material-arrow-right: Evaluate](modules/03-evaluation.md)
</div>

<div class="module-card" markdown>
<span class="module-num">MODULE 4</span>
### Data Tools <span class="badge-duration">75 min</span>
Add Genie for SQL queries, custom MCP tools, and multi-tool routing.

[:material-arrow-right: Extend](modules/04-data-tools.md)
</div>

<div class="module-card" markdown>
<span class="module-num">MODULE 5</span>
### Deployment <span class="badge-duration">60 min</span>
Deploy to Databricks Apps with monitoring and quality gates.

[:material-arrow-right: Deploy](modules/05-deployment.md)
</div>

</div>

## Who Is This For?

**Target audience:** Developers familiar with LangChain/agents but new to Databricks.

**Prerequisites:**

- Python programming
- Basic LangChain/LangGraph concepts
- Familiarity with agents (ReAct pattern)

**Databricks requirements:**

- MLR 17.3 LTS or higher
- Unity Catalog enabled
- Lakebase and Genie access
- Foundation Model endpoints available
