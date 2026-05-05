# Module 5: Deployment

<span class="badge-duration">60 minutes</span>

Deploy your agent to production on Databricks Apps with monitoring, quality gates, and a release pipeline.

## What You'll Build

A production-ready Databricks App with:

- `/invocations` API endpoint
- Interactive chat UI
- MLflow trace monitoring
- Continuous evaluation with registered scorers
- End-to-end release pipeline

## Notebooks

| Notebook | Topics | Duration |
|---|---|---|
| [`01_apps_dev_loop.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/05_deployment/01_apps_dev_loop.py) | Day-to-day Apps development loop | 10 min |
| [`02_apps_deployment.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/05_deployment/02_apps_deployment.py) | Deploy app + validate `/invocations` | 15 min |
| [`03_production_monitoring.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/05_deployment/03_production_monitoring.py) | MLflow traces + online evaluation | 20 min |
| [`04_end_to_end_deployment.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/05_deployment/04_end_to_end_deployment.py) | End-to-end release pipeline | 15 min |

## Prerequisites

- All prior modules completed (or at minimum Modules 0–2)
- App source code at `apps/knowledge_assistant_agent`
- Databricks CLI configured
- OAuth token for testing

## Key Concepts

### Deployment Flow

The deployment uses direct Databricks Apps commands:

```bash
# Sync source to workspace files
databricks sync "apps/knowledge_assistant_agent" \
  "/Users/$USERNAME/knowledge_assistant_agent_app"

# Deploy the app
databricks apps deploy knowledge-assistant-agent-app \
  --source-code-path "/Workspace/Users/$USERNAME/knowledge_assistant_agent_app"
```

No model registry or bundle deployment involved — just source code push.

### App Runtime

The deployed app uses `mlflow.genai.agent_server.AgentServer` with:

- `@invoke()` and `@stream()` decorators for request handlers
- `/invocations` endpoint for API calls
- Static file serving for the chat UI at `GET /`
- Environment variables for configuration (LLM endpoint, Lakebase settings, etc.)

### Memory in Production

Pass `thread_id` and `user_id` in every request:

```json
{
  "input": [{"role": "user", "content": "How much vacation time do I get?"}],
  "custom_inputs": {
    "thread_id": "conversation-123",
    "user_id": "alice@company.com"
  }
}
```

- `thread_id` — scopes short-term memory to one conversation
- `user_id` — scopes long-term memory to one user

### Production Monitoring

Register scorers for continuous evaluation of live traffic:

| Scorer | Type | Sample Rate |
|---|---|---|
| `Safety` | Built-in LLM judge | 100% |
| `Guidelines` (professional tone) | Built-in LLM judge | 50% |
| `response_clarity` | Custom LLM judge | 30% |
| `response_length_check` | Code-based | 100% |

Scorers run automatically on new traces — no manual intervention after registration.

### Quality Gates

The end-to-end pipeline validates:

1. All responses return HTTP 200 with JSON content (not an HTML login page)
2. Short-term memory continuity — same `thread_id` preserves context
3. Short-term memory isolation — different `thread_id` starts fresh
4. Long-term memory availability — `user_id` is passed through
5. MLflow traces appear in the app experiment

### App Source Structure

```
apps/knowledge_assistant_agent/
├── app.yaml                 # Runtime config and env vars
├── requirements.txt         # Python dependencies
├── static/index.html        # Chat UI
└── agent_server/
    ├── agent.py             # Entry point (@invoke, @stream)
    ├── langgraph_agent.py   # KnowledgeAssistant class
    ├── start_server.py      # AgentServer setup
    └── utils_memory.py      # Memory utility functions
```

## What You'll Understand

- How to deploy agents as Databricks Apps
- Day-to-day development loop for iterating
- Endpoint validation and health checks
- Production monitoring with registered scorers
- End-to-end release pipeline with quality gates

## What's Next?

You've completed the bootcamp. From here:

- **Customize** — add your own documents, tables, and evaluation criteria
- **Extend** — multi-agent orchestration, human-in-the-loop workflows
- **Scale** — load testing, multi-region deployment, compliance hardening
