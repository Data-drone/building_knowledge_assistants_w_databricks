# Module 3: Evaluation

<span class="badge-duration">75 minutes</span>

Build a complete evaluation workflow: from a toy loop to production-style scoring with MLflow tracing, built-in scorers, custom judges, and quality gates.

## What You'll Build

An evaluation pipeline that assesses your agent's quality using three types of scorers — progressing from simple heuristics to agent behavior checks and multi-turn memory testing.

## Notebooks

| Notebook | Topics | Duration |
|---|---|---|
| [`00_first_eval_loop.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/03_evaluation/00_first_eval_loop.py) | MLflow eval rows, basic built-in scorers | 15 min |
| [`01_tracing.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/03_evaluation/01_tracing.py) | MLflow tracing, one-run inspection | 15 min |
| [`02_evaluation.py`](https://github.com/Data-drone/building_knowledge_assistants_w_databricks/blob/main/03_evaluation/02_evaluation.py) | Built-in scorers, custom judges, quality gates | 45 min |

## Prerequisites

- Vector Search index from [Module 1](01-rag-pipeline.md)
- Lakebase target created (for CheckpointSaver)
- LLM endpoint available
- Recommended: completed [Module 2](02-memory.md)

## Key Concepts

### The Evaluation Progression

The module teaches evaluation in the order you typically need it:

```
Toy Loop → Trace One Run → Built-In Scorers → @scorer checks
    → Custom Judges → Agent Behavior → Multi-Turn → Quality Gates
```

### Three Types of Scorers

| Type | Example | When to Use |
|---|---|---|
| **Built-in LLM judge** | `Correctness()`, `Safety()`, `Guidelines()` | Quick start, broad quality checks |
| **Code-based `@scorer`** | `policy_detail_check` | Deterministic checks, no LLM cost |
| **Custom LLM `make_judge()`** | `hr_accuracy`, `policy_completeness` | Domain-specific nuance |

### Notebook 00: First Eval Loop

Start with the simplest possible evaluation:

1. Create a small eval dataset in MLflow's row structure
2. Generate answers in a loop
3. Run a couple of built-in scorers
4. Build intuition for the eval pattern before scaling up

### Notebook 01: Tracing

Before scoring many runs, inspect one deeply:

- Enable MLflow tracing for the LangGraph agent
- See spans for LLM calls, tool calls, and memory operations
- Check latency, token counts, and errors
- Add custom attributes with `@mlflow.trace`

### Notebook 02: Full Evaluation

The main evaluation notebook uses `predict_fn` so MLflow handles prediction, tracing, and parallelism:

```python
results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=[Correctness(), Safety(), hr_accuracy_judge],
)
```

Topics covered:

- **Built-in scorers** — `Correctness` for fact-checking, `Safety` for harmful content, `Guidelines` for custom rules
- **Code-based scorers** — `@scorer` decorator for fast deterministic checks
- **Custom LLM judges** — `make_judge()` for domain-specific evaluation
- **Agent behavior** — `ToolCallCorrectness` and `ToolCallEfficiency`
- **Multi-turn memory** — `ConversationCompleteness` and `KnowledgeRetention`
- **Quality gates** — pass/fail thresholds on scorer outputs

## What You'll Understand

- How to start evaluation with a simple Q&A loop
- How to inspect one run deeply with MLflow tracing
- The `predict_fn` pattern for integrated generation + scoring
- When custom judges are actually necessary (hint: try built-ins first)
- How to connect scorer outputs to human feedback and quality gates

## What's Next?

Continue to [Module 4: Data Tools](04-data-tools.md) to add structured data queries and custom MCP tools.
