# Evaluation - Observability and Quality Assurance

Build production-ready evaluation and observability for your AI agents in under 60 minutes!

## What You'll Learn

This module teaches you to add systematic evaluation and observability to AI agents:
1. **Notebook 01** (20 min): MLflow Tracing - Add comprehensive observability to your agent
2. **Notebook 02** (30 min): Agent Evaluation - Build custom judges and production quality gates

## Prerequisites

### Required
- **Vector Search index** available (from Module 01)
  - Expected index: `{CATALOG}.{SCHEMA}.policy_index`
- **Lakebase instance** created (for CheckpointSaver)
- LLM endpoint available (`databricks-claude-sonnet-4-5` or similar)

### Recommended
- Completed [Module 02 (Memory)](../02_memory/README.md) - This module builds directly on the memory-enabled agent
  - Module 03 is self-contained, but Module 02 provides the agent foundation

## Learning Path

### Start Here: 01_tracing.py

**What you'll build**: Add comprehensive observability to your agent with MLflow tracing

**Key concepts**:
- Enable MLflow tracing for full observability
- Instrument code with `@mlflow.trace` decorator
- Inspect traces (spans, latency, tool calls, errors)
- Add custom attributes for debugging
- Use traces to identify performance bottlenecks
- Apply production tracing features (async logging, sampling)

**Time**: 20 minutes

---

### Next: 02_evaluation.py

**What you'll build**: Systematic agent evaluation with custom judges and production quality gates

**Key concepts**:
- Create evaluation datasets with ground truth
- Build custom judges with `make_judge()` (MLflow 3.x)
- Run comprehensive evaluation with `mlflow.genai.evaluate()`
- Interpret scores and identify issues
- Implement human feedback loops
- Set up production quality gates

**Time**: 30 minutes

---

## What's Next?

After completing these notebooks, you'll understand:
- ✅ How to add observability to agents (MLflow tracing)
- ✅ Systematic quality evaluation with custom judges
- ✅ Creating domain-specific evaluation metrics
- ✅ Human feedback integration
- ✅ Production monitoring and quality gates

### Continue Learning

**For production deployment**:
- **Module 04**: Multi-tool agents (Genie, UC Functions)
- **Module 05**: Deployment to Databricks Apps

---

## Quick Start

1. Complete [Module 02 (Memory)](../02_memory/README.md) OR have a basic understanding of checkpointed RAG agents
2. Open `01_tracing.py`
3. Enable MLflow tracing
4. Inspect traces and add custom instrumentation
5. Apply production tracing features
6. Continue to `02_evaluation.py`
7. Create evaluation datasets and custom judges
8. Set up production quality gates

---

## Technical Stack

- **LLM**: Databricks LLM endpoints (Claude Sonnet 4.5)
- **Orchestration**: LangGraph
- **Vector Search**: Databricks Vector Search (for RAG)
- **Memory**: CheckpointSaver (from Module 02)
- **Tracing**: MLflow Tracing (OpenTelemetry-compatible)
- **Evaluation**: MLflow Evaluate with built-in + custom metrics

---

## Evaluation Architecture

### Custom Judges with make_judge() (MLflow 3.x)
- **Domain-specific**: HR policy accuracy, completeness, professional tone
- **Ratings**: Categorical (excellent/good/fair/poor/very_poor) or boolean feedback
- **LLM-as-Judge**: Uses LLMs to evaluate outputs against custom criteria
- **Usage**: Enforce business rules and quality standards
- **When**: You need domain-specific quality assessment

### Evaluation Data Format

MLflow GenAI requires data structured as:
```python
eval_data = [
    {
        "inputs": {"question": "..."},
        "outputs": "predicted response",
        "expectations": {"expected_answer": "..."}
    }
]
```

---

## Production Patterns

### 1. Observability Stack
```python
# Enable tracing
mlflow.tracing.enable()
mlflow.config.enable_async_logging()

# Add custom instrumentation
@mlflow.trace
def my_function():
    pass
```

### 2. Evaluation Pipeline
```python
from mlflow.genai.judges import make_judge
from typing import Literal

# Create custom judges
judge = make_judge(
    name="hr_accuracy",
    instructions=(
        "Evaluate if {{ outputs }} is factually accurate.\n"
        "Compare to the expected answer: {{ expectations }}\n"
        "Rate as: excellent, good, fair, poor, or very_poor"
    ),
    feedback_value_type=Literal["excellent", "good", "fair", "poor", "very_poor"],
    model=f"databricks:/{LLM_ENDPOINT}",
)

# Transform data to MLflow GenAI format
eval_data_formatted = []
for idx, row in eval_data.iterrows():
    pred = predict(row["question"])
    eval_data_formatted.append({
        "inputs": {"question": row["question"]},
        "outputs": pred,
        "expectations": {"expected_answer": row["expected_answer"]}
    })

# Run evaluation
results = mlflow.genai.evaluate(
    data=eval_data_formatted,
    scorers=[judge1, judge2, judge3],
)

# Check quality gates
print(results.metrics)
```

### 3. Continuous Monitoring
```
Deploy → Trace → Evaluate → Human Feedback → Refine Judges → Repeat
```

---

## Common Issues & Solutions

### Tracing
- **Issue**: Traces not appearing
- **Fix**: Ensure `mlflow.tracing.enable()` is called before agent execution

### Evaluation
- **Issue**: Need to pre-generate predictions before evaluation
- **Fix**: mlflow.evaluate() with custom metrics requires predictions column in dataframe

### Custom Judges
- **Issue**: Judge scores don't match human intuition
- **Fix**: Collect 10-20 human ratings and refine grading_prompt with better examples

### Performance
- **Issue**: Evaluation taking too long
- **Fix**: Each judge calls an LLM - run in parallel or reduce dataset size

---

## Need Help?

- Check MLflow traces in UI for debugging
- Review score justifications (not just numbers)
- Compare built-in vs custom judges to understand gaps
- See Module 02 for agent architecture questions

---

**Total Time**: 50 minutes from basic agent to production-ready evaluation 🚀
