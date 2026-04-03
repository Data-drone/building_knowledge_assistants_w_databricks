# Evaluation - Observability and Quality Assurance

Build a natural evaluation workflow for your Databricks agent in about 75 minutes.

## What You'll Learn

This module teaches the evaluation flow in the order learners usually need it:
1. **Notebook 00** (15 min): First evaluation loop
2. **Notebook 01** (15 min): Trace one run with MLflow
3. **Notebook 02** (45 min): Score many runs with built-ins first, then custom judges, agent behavior, and multi-turn memory

## Prerequisites

### Required
- **Vector Search index** available (from Module 01)
  - Expected index: `{CATALOG}.{SCHEMA}.policy_index`
- **Lakebase short-term memory target** created (for CheckpointSaver)
  - Provisioned: `LAKEBASE_INSTANCE_NAME`
  - Autoscaling: `LAKEBASE_AUTOSCALING_PROJECT` + `LAKEBASE_AUTOSCALING_BRANCH`
- LLM endpoint available (`databricks-claude-sonnet-4-6` or similar)

### Recommended
- Completed [Module 02 (Memory)](../02_memory/README.md)
  - Module 03 keeps the same memory-capable bot structure and simply adds observability and evaluation around it

## Learning Path

### Start Here: 00_first_eval_loop.py

**What you'll build**: A first MLflow-native evaluation loop for the same bot structure from Module 02

**Key concepts**:
- Create a small eval dataset in MLflow's row structure
- Generate answers in a simple loop
- Run a couple of built-in scorers with minimal MLflow setup
- Learn the core eval pattern before moving to richer scoring

**Time**: 15 minutes

---

### Next: 01_tracing.py

**What you'll build**: One-run observability with MLflow tracing

**Key concepts**:
- Enable MLflow tracing for the LangGraph agent
- Inspect spans, tool calls, latency, and errors
- Add custom attributes with `@mlflow.trace`
- Use tracing to debug one execution before scaling evaluation

**Time**: 15 minutes

---

### Next: 02_evaluation.py

**What you'll build**: Comprehensive evaluation using `predict_fn`, three types of scorers, agent behavior checks, and multi-turn memory testing

**Key concepts**:
- Use `predict_fn` to let MLflow handle prediction, tracing, and parallelism
- Start with built-in scorers including `Correctness` for fact-checking
- Write code-based scorers with the `@scorer` decorator
- Add `make_judge()` only for domain-specific HR accuracy and completeness
- Evaluate agent behavior with `ToolCallCorrectness` and `ToolCallEfficiency`
- Test multi-turn memory with conversation-level scorers
- Turn the final scorer set into human feedback loops and quality gates

**Time**: 45 minutes

---

## What's Next?

After completing these notebooks, you'll understand:
- ✅ How to start evaluation with a simple Q&A loop
- ✅ How to inspect one run deeply with MLflow tracing
- ✅ The `predict_fn` pattern for integrated generation + scoring
- ✅ Three types of scorers: built-in LLM, code-based `@scorer`, custom `make_judge()`
- ✅ How to evaluate agent behavior (tool calls) and conversation quality (memory)
- ✅ When custom judges are actually necessary
- ✅ How to connect scorer outputs to human feedback and quality gates

### Continue Learning

**For production deployment**:
- **Module 04**: Multi-tool agents (Genie, UC Functions)
- **Module 05**: Deployment to Databricks Apps

---

## Quick Start

1. Complete [Module 02 (Memory)](../02_memory/README.md) OR have a basic understanding of memory-capable LangGraph agents
2. Open `00_first_eval_loop.py`
3. Continue to `01_tracing.py`
4. Finish in `02_evaluation.py`

---

## Technical Stack

- **LLM**: Databricks LLM endpoints (Claude Sonnet 4.6)
- **Orchestration**: LangGraph
- **Vector Search**: Databricks Vector Search (for RAG)
- **Memory Shape**: Same prompted graph structure carried forward from Module 02
- **Configuration Style**: Each notebook carries a small local config block for clarity
- **Tracing**: MLflow Tracing
- **Evaluation**: MLflow GenAI evaluation with built-in scorers and custom judges

---

## Evaluation Architecture

### 1. First Evaluation Loop
- Create a few questions
- Generate answers
- Score with a simple heuristic
- Use this only to build intuition

### 2. Trace One Run
- Inspect the inside of a single execution
- See tool calls, retrieved context, span timings, and errors
- Use this to explain surprising evaluation outcomes

### 3. Built-In Scorers First
- `Correctness` for fact-checking against expected facts
- `Safety` for harmful output checks
- `Guidelines` for broad natural-language rules such as tone or groundedness
- Use these as the default path for many-run evaluation

### 4. Code-Based `@scorer` for Deterministic Checks
- No LLM call required — fast and free
- Use for exact-match checks, format validation, or heuristic signals

### 5. Custom LLM Judges Last
- Use `make_judge()` for domain-specific checks like HR policy accuracy
- Add them only when built-ins are too generic for the job

### 6. Agent Behavior and Conversation Quality
- `ToolCallCorrectness` and `ToolCallEfficiency` for tool-calling agents
- `ConversationCompleteness` and `KnowledgeRetention` for multi-turn memory

### Three Types of Scorers

| Type | Example | When to Use |
| --- | --- | --- |
| Built-in LLM | `Correctness()`, `Safety()`, `Guidelines()` | Quick start, broad quality |
| Code-based `@scorer` | `policy_detail_check` | Deterministic checks, no LLM cost |
| Custom LLM `make_judge()` | `hr_accuracy`, `policy_completeness` | Domain-specific nuance |

### Evaluation Data Format

```python
eval_data = [
    {
        "inputs": {"question": "...", "user_id": "..."},
        "expectations": {
            "expected_answer": "...",
            "expected_facts": ["fact 1", "fact 2"],
        }
    }
]

def predict_fn(question: str, user_id: str) -> str:
    return agent.invoke(...)

results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=[Correctness(), Safety(), hr_accuracy_judge],
)
```

---

## Production Patterns

### 1. Observability Stack
```python
mlflow.tracing.enable()
mlflow.config.enable_async_logging()

@mlflow.trace
def traced_helper():
    pass
```

### 2. Batch Evaluation Pipeline
```python
from mlflow.genai.scorers import Correctness, Safety, Guidelines, ToolCallCorrectness, scorer
from mlflow.genai.judges import make_judge
from mlflow.entities import Feedback

# Built-in scorers
built_in_scorers = [
    Correctness(),
    Safety(),
    Guidelines(name="professional_tone", guidelines=["Use a professional, helpful tone."]),
    ToolCallCorrectness(),
]

# Code-based scorer (no LLM needed)
@scorer
def policy_detail_check(outputs, expectations) -> Feedback:
    expected = expectations.get("expected_answer", "").lower()
    return Feedback(value="yes" if expected in str(outputs).lower() else "no")

# Custom LLM judge
hr_accuracy_judge = make_judge(
    name="hr_accuracy",
    instructions="Compare {{ outputs }} to {{ expectations }} for HR policy accuracy.",
    feedback_value_type=Literal["excellent", "good", "fair", "poor", "very_poor"],
    model=f"databricks:/{LLM_ENDPOINT}",
)

results = mlflow.genai.evaluate(
    data=eval_rows,
    predict_fn=predict_fn,
    scorers=built_in_scorers + [policy_detail_check, hr_accuracy_judge],
)
```

### 3. Continuous Improvement Loop
```text
Toy Loop -> Trace One Run -> Built-In Scorers -> @scorer checks -> Custom Judges -> Agent Behavior -> Multi-Turn -> Human Feedback -> Quality Gates
```

---

## Common Issues & Solutions

### Tracing
- **Issue**: Traces do not appear
- **Fix**: Call `mlflow.tracing.enable()` before generating answers

### Evaluation
- **Issue**: Built-in scorers look fine but answers are still wrong
- **Fix**: Add domain-specific judges for factual accuracy or completeness

### Custom Judges
- **Issue**: Judge scores do not match expert review
- **Fix**: Collect 10-20 human ratings and tighten the judge instructions around disagreement patterns

### Performance
- **Issue**: Batch evaluation takes too long
- **Fix**: Start with built-ins, keep the dataset small while iterating, and add custom judges only for the gaps that matter

---

## Need Help?

- Check MLflow traces in the UI when a score surprises you
- Start with built-ins before writing custom judges
- Review rationales, not just aggregate metrics
- See Module 02 for the shared bot structure

---

**Total Time**: 75 minutes from first eval loop to production-style scoring
