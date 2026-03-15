# Databricks Agent Bootcamp

A hands-on tutorial series for building production-ready AI agents on Databricks. Build an **Internal Knowledge Assistant** that answers employee questions using Vector Search (documents) and Genie (structured data), with conversation memory powered by Lakebase.

**Target Audience:** Developers familiar with LangChain/agents but new to Databricks
**Duration:** 2 hours hands-on
**Level:** Intermediate

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Observability Layer"
        MLflow[MLflow 3.0<br/>Tracing → Evaluation → Judges]
    end

    subgraph "Agent Layer"
        Agent[ResponsesAgent<br/>LangGraph]
        LLM[Mosaic AI Gateway<br/>Claude Sonnet 4.6]
        Tools[MCP Tools]
        Memory[Lakebase<br/>Checkpointer + Store]
    end

    subgraph "Data Layer"
        VS[Vector Search<br/>Document Chunks]
        Genie[Genie<br/>Employee Data]
        UC[UC Functions<br/>Custom Tools]
    end

    User -->|Query| Agent
    Agent --> LLM
    Agent --> Tools
    Tools --> VS
    Tools --> Genie
    Tools --> UC
    Agent --> Memory
    Agent --> MLflow

    style Agent fill:#4a9eff
    style MLflow fill:#ff6b6b
    style Memory fill:#51cf66
```

---

## 📚 Learning Path

### Module 0: Foundations (45 min)
Build a solid understanding of the Databricks platform.

| Notebook | Topics | Duration |
|----------|--------|----------|
| [00_setup.py](00_foundations/00_setup.py) | Unity Catalog, Lakebase, Sample Data | 15 min |
| [01_mosaic_gateway.py](00_foundations/01_mosaic_gateway.py) | LLM Endpoints, Pay-per-token vs Provisioned | 10 min |
| [02_platform_orientation.py](00_foundations/02_platform_orientation.py) | MCP, Unity Catalog, ResponsesAgent | 15 min |

**You'll learn:** Platform concepts, governance model, why MCP matters

---

### Module 1: RAG Pipeline (45 min)
Build your first document-based Q&A agent.

| Notebook | Topics | Duration |
|----------|--------|----------|
| [00_your_first_agent_on_databricks.py](01_rag_pipeline/00_your_first_agent_on_databricks.py) | First agent loop with Databricks-hosted LLM | 15 min |
| [01_building_a_doc_store_on_vector_search.py](01_rag_pipeline/01_building_a_doc_store_on_vector_search.py) | Document chunking, Delta Sync Vector Search, RAG pattern | 25 min |

**You'll build:** Agent that answers policy questions from documents

---

### Module 2: Memory (60 min)
Add conversation memory for multi-turn interactions.

This module maps directly to Databricks' latest stateful agent guidance:
[AI agent memory](https://docs.databricks.com/aws/en/generative-ai/agent-framework/stateful-agents)

| Notebook | Topics | Duration |
|----------|--------|----------|
| [01_short_term_memory.py](02_memory/01_short_term_memory.py) | Checkpoint saver, thread memory, multi-turn conversations | 20 min |
| [02_long_term_memory.py](02_memory/02_long_term_memory.py) | DatabricksStore, user preferences, cross-session memory | 25 min |

**You'll build:** Agent with multi-turn conversations and persistent state

---

### Module 3: Evaluation (60 min)
Implement observability and evaluation.

| Notebook | Topics | Duration |
|----------|--------|----------|
| [01_tracing.py](03_evaluation/01_tracing.py) | MLflow tracing, span analysis | 15 min |
| [02_evaluation.py](03_evaluation/02_evaluation.py) | Built-in scorers, custom judges, quality gates | 30 min |

**You'll build:** Comprehensive evaluation framework with scorers and judges

---

### Module 4: MCP Tool Integration (75 min)
Learn how to add data query capabilities and custom tools to your agents.

| Notebook | Topics | Duration |
|----------|--------|----------|
| [00_setup.py](04_mcp_tool_integration/00_setup.py) | Genie space creation, sample data for structured queries | 10 min |
| [01_genie_integration.py](04_mcp_tool_integration/01_genie_integration.py) | Genie spaces, natural language SQL, multi-tool agents | 35 min |
| [02_custom_tools.py](04_mcp_tool_integration/02_custom_tools.py) | Custom tools, MCP servers, advanced patterns | 30 min |

**You'll build:** Agent with multiple tools (Vector Search + Genie + custom tools) that intelligently routes between documentation, data queries, and external APIs

---

### Module 5: Deployment (45 min)
Deploy your agent to production on Databricks Apps.

| Notebook | Topics | Duration |
|----------|--------|----------|
| [01_apps_dev_loop.py](05_deployment/01_apps_dev_loop.py) | Day-to-day Apps development loop | 10 min |
| [02_apps_deployment.py](05_deployment/02_apps_deployment.py) | Deploy app + validate `/invocations` | 15 min |
| [03_production_monitoring.py](05_deployment/03_production_monitoring.py) | MLflow traces + online evaluation checks | 10 min |
| [04_end_to_end_deployment.py](05_deployment/04_end_to_end_deployment.py) | End-to-end Apps deployment pipeline | 10 min |

**You'll build:** Production-ready Databricks App with monitoring

---

## 🚀 Quick Start

### Prerequisites

**Databricks Workspace:**
- **Databricks Runtime:** MLR 17.3 LTS or higher
- Unity Catalog enabled
- Lakebase access
- Genie access
- Foundation Model endpoints available

**Permissions:**
- Create catalogs/schemas
- Create Vector Search indexes
- Create Lakebase projects
- Deploy Databricks Apps

**Knowledge:**
- Python programming
- Basic LangChain/LangGraph concepts
- Familiarity with agents (ReAct pattern)

### Setup Instructions

**⚠️ Follow these steps in order**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Data-drone/building_knowledge_assistants_w_databricks.git databricks_agent_bootcamp
   cd databricks_agent_bootcamp
   ```

2. **Update configuration in [config.py](config.py):**
   ```python
   CATALOG = "agent_bootcamp"           # Your catalog name
   SCHEMA = "knowledge_assistant"       # Your schema name
   HOST = "https://your-workspace.cloud.databricks.com"
   REGION = "us-west-2"                 # Your cloud region
   ```

3. **Run the setup notebook (15 minutes):**
   - Open [00_foundations/00_setup.py](00_foundations/00_setup.py)
   - Run all cells to install foundational dependencies and create Unity Catalog assets, sample data, and the Lakebase project
   - Wait for the notebook to complete before moving to the next module

4. **Follow the learning path:**
   - Start with Module 0 (Foundations)
   - Progress through modules sequentially
   - Each "driver" notebook demonstrates the complete agent

---

## 🧰 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Agent Framework** | MLflow ResponsesAgent + LangGraph | Agent orchestration and execution |
| **LLM** | Mosaic AI Gateway (Claude Sonnet 4.6) | Reasoning and generation |
| **Vector Search** | Databricks Vector Search (Delta Sync) | Document similarity search |
| **Structured Data** | Genie | Natural language to SQL |
| **Memory** | Lakebase (PostgreSQL) | Conversation state (checkpointer + store) |
| **Observability** | MLflow 3.0+ | Tracing, evaluation, judges |
| **Governance** | Unity Catalog | Permissions, audit logs |
| **Deployment** | Databricks Apps | Production app endpoint |

---

## 📖 Key Concepts

### ResponsesAgent
MLflow's recommended agent base class with:
- OpenAI Responses API compatibility
- Multi-agent orchestration support
- Advanced streaming with events
- Thread-based conversation management

### Model Context Protocol (MCP)
Databricks' standard for exposing services as agent tools:
- **Vector Search MCP**: Governed similarity search
- **Genie MCP**: Natural language to SQL
- **UC Functions MCP**: Execute registered functions

**Why MCP?** Unity Catalog permissions flow through automatically.

### Lakebase
Managed PostgreSQL with:
- **Autoscaling**: Scale-to-zero after 15 min idle
- **Branching**: Copy-on-write branches (like Git)
- **Checkpointer**: Short-term conversation memory
- **Store**: Long-term facts and preferences

For stateful Apps, pass a stable `thread_id` per conversation so short-term
memory stays scoped to one session.
The deployed app in this repo uses the autoscaling Lakebase target
`knowledge-assistant-state/production` rather than a legacy provisioned instance.

### MLflow 3.0 Evaluation
- **Scorers**: Built-in (Correctness, Groundedness, Guidelines)
- **make_judge()**: Create custom judges with natural language
- **Agent-as-a-Judge**: Analyze traces with `{{ trace }}` variable
- **Judge Alignment**: Align judges with human feedback (DSPy-SIMBA)

---

## 🎯 What You'll Build

By the end of this bootcamp, you'll have built a production-ready **Internal Knowledge Assistant** that:

✅ Answers policy questions from documents (Vector Search)
✅ Queries employee data (Genie)
✅ Maintains conversation context (Lakebase checkpointer)
✅ Remembers user preferences (Lakebase store)
✅ Traces all execution (MLflow)
✅ Evaluates quality (Scorers + Judges)
✅ Deploys to production (Databricks Apps)
✅ Monitors in real-time (Inference tables)

---

## 📦 Project Structure

```
databricks_agent_bootcamp/
├── config.py                       # Shared configuration
├── databricks.yml                  # Databricks asset bundle definition
├── 00_foundations/                  # Platform basics
├── 01_rag_pipeline/                # Vector Search + documents
├── 02_memory/                      # Lakebase checkpointing
├── 03_evaluation/                  # MLflow judges + scorers
├── 04_mcp_tool_integration/        # Genie + custom MCP tools
├── 05_deployment/                  # Production monitoring + eval
└── apps/
    └── knowledge_assistant_agent/  # Canonical Databricks Apps runtime
```

---

## 🌟 Best Practices

### Development
- Use **development branch** in Lakebase for testing
- Enable **MLflow tracing** for all agent calls
- Test with **small eval datasets** first

### Production
- The bootcamp's only deployment path is **Databricks Apps**
- Deploy the agent from `apps/knowledge_assistant_agent`
- Use the app-hosted `/invocations` endpoint for validation and testing
- Pass `custom_inputs.thread_id` on every request to preserve short-term memory
- Configure short-term memory with either `LAKEBASE_INSTANCE_NAME` or the autoscaling pair
  `LAKEBASE_AUTOSCALING_PROJECT` + `LAKEBASE_AUTOSCALING_BRANCH`
- Pass a stable user identity for long-term memory so preferences persist across sessions
- Enable **MLflow tracing** for monitoring
- Use **connection pooling** (min_size=2) to avoid cold starts
- Configure **online evaluation** for quality monitoring

### Security
- Always use **MCP tools** (not direct SDK) for governance
- Grant **minimal permissions** via Unity Catalog
- Store secrets in **Databricks Secrets**
- Enable **audit logging** for compliance

---

## 🆘 Troubleshooting

### Common Issues

**Vector Search index not syncing**
- Check that source Delta table has data
- Verify embedding endpoint is available
- Wait for index state: `ONLINE`

**Lakebase connection timeout**
- First connection after idle has 20-30s cold start
- Use connection pooling (min_size >= 2) to keep warm

**CheckpointSaver setup fails on Databricks Apps**
- For autoscaling Lakebase, configure `LAKEBASE_AUTOSCALING_PROJECT` and `LAKEBASE_AUTOSCALING_BRANCH`
- Ensure the app service principal has `USAGE, CREATE` on schema `public`
- Ensure checkpoint tables in `public` grant `SELECT, INSERT, UPDATE` to the app role

**DatabricksStore setup fails on Databricks Apps**
- Use the same Lakebase autoscaling project/branch settings as short-term memory
- Ensure the app role can create and update `public.store`, `public.store_vectors`,
  `public.store_migrations`, and `public.vector_migrations`
- Verify your embedding endpoint and dimensions match the store configuration

**MCP tools not found**
- Verify MCP URL format is correct
- Check Unity Catalog permissions on underlying assets
- Ensure authentication token has required scopes

**Agent not using tools**
- Check that tools are bound to LLM: `llm.bind_tools(tools)`
- Verify tool descriptions are clear and specific
- Review traces to see LLM reasoning

---

## 📚 Additional Resources

- [Databricks AI agent memory](https://docs.databricks.com/aws/en/generative-ai/agent-framework/stateful-agents)
- [Databricks Agent Framework Documentation](https://docs.databricks.com/en/generative-ai/agent-framework/author-agent)
- [MLflow 3.0 Evaluation Guide](https://mlflow.org/docs/latest/evaluation/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Unity Catalog Best Practices](https://docs.databricks.com/data-governance/unity-catalog/)

---

## 🤝 Contributing

Issues and pull requests welcome! Please follow these guidelines:
- Test all notebooks before submitting
- Update README for significant changes
- Follow Databricks notebook format conventions

---

## 🎓 Next Steps

After completing this bootcamp:

1. **Customize for your use case:**
   - Add domain-specific documents
   - Create custom UC Functions
   - Tune evaluation criteria

2. **Explore advanced patterns:**
   - Multi-agent orchestration
   - Human-in-the-loop workflows
   - Custom rerankers

3. **Scale to production:**
   - Load testing and optimization
   - Multi-region deployment
   - Compliance and security hardening

---

**Happy Building! 🚀**
