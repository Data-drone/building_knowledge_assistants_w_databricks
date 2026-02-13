# Databricks notebook source
# MAGIC %md
# MAGIC # Production Monitoring and Online Evaluation
# MAGIC
# MAGIC ## What You'll Do
# MAGIC Monitor your deployed agent using inference tables, run online evaluation with
# MAGIC the custom judges from Module 03, and set up quality alerts.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Query inference tables for production request/response data
# MAGIC - Analyze production traces with MLflow
# MAGIC - Review stakeholder feedback from the Review App
# MAGIC - Run online evaluation with custom judges on production data
# MAGIC - Define SQL monitoring queries for alerting
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed [02_agent_serving.py](02_agent_serving.py) — agent deployed to serving
# MAGIC - Understanding of custom judges from Module 03
# MAGIC
# MAGIC ## Estimated Time
# MAGIC 25 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies

# COMMAND ----------

%pip install -q --upgrade mlflow[databricks]>=3.1.0 databricks-sdk databricks-agents

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configuration

# COMMAND ----------

import mlflow
from mlflow.genai.judges import make_judge
from typing import Literal
from databricks.sdk import WorkspaceClient
from databricks import agents
import pandas as pd

# Configuration
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
LLM_ENDPOINT = "databricks-claude-sonnet-4-5"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.knowledge_assistant"

w = WorkspaceClient()
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# Determine the endpoint name from the model
# agents.deploy() typically creates an endpoint based on the model name
versions = list(w.model_versions.list(full_name=MODEL_NAME))
if not versions:
    raise Exception(f"No model versions found for {MODEL_NAME}. Deploy the agent first.")

# List serving endpoints to find ours
all_endpoints = list(w.serving_endpoints.list())
agent_endpoints = [ep for ep in all_endpoints if MODEL_NAME.replace(".", "_") in ep.name.replace("-", "_").replace(".", "_")]

if agent_endpoints:
    ENDPOINT_NAME = agent_endpoints[0].name
    print(f"✓ Found serving endpoint: {ENDPOINT_NAME}")
else:
    # Fallback: agents.deploy() prepends "agents_" to the model-derived name
    ENDPOINT_NAME = f"agents_{MODEL_NAME.replace('.', '-')}"
    print(f"⚠ Using constructed endpoint name: {ENDPOINT_NAME}")

# Set experiment for monitoring
experiment_name = f"/Users/{username}/agent_bootcamp_monitoring"
mlflow.set_experiment(experiment_name)

print(f"\n✓ Configuration loaded")
print(f"  Model: {MODEL_NAME}")
print(f"  Endpoint: {ENDPOINT_NAME}")
print(f"  Experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Understanding the Monitoring Stack
# MAGIC
# MAGIC When you deployed with `agents.deploy()`, three monitoring systems were
# MAGIC automatically enabled:
# MAGIC
# MAGIC | System | What It Captures | Use Case |
# MAGIC |--------|-----------------|----------|
# MAGIC | **Inference Tables** | Every request + response as Delta rows | Volume, errors, latency |
# MAGIC | **MLflow Tracing** | Internal execution spans (LLM calls, tool calls) | Debugging, performance |
# MAGIC | **Assessment Tables** | Human feedback from Review App | Quality signals |
# MAGIC
# MAGIC Together, these give you full observability into your production agent.
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────────────────────────┐
# MAGIC │                    Production Agent Endpoint                     │
# MAGIC └───────────┬──────────────────────────┬──────────────────────────┘
# MAGIC             │                          │
# MAGIC       Requests/Responses          MLflow Traces
# MAGIC             │                          │
# MAGIC             ▼                          ▼
# MAGIC  ┌─────────────────────┐   ┌─────────────────────────┐
# MAGIC  │  Inference Tables   │   │   MLflow Experiment      │
# MAGIC  │  (Delta Tables)     │   │   (Trace Storage)        │
# MAGIC  └──────┬──────────────┘   └───────────┬─────────────┘
# MAGIC         │                               │
# MAGIC    ┌────┴────────┐                      │
# MAGIC    │             │                      │
# MAGIC    ▼             ▼                      ▼
# MAGIC ┌─────────┐  ┌──────────────┐  ┌───────────────────┐
# MAGIC │  SQL    │  │  Online       │  │ Trace Analysis    │
# MAGIC │  Alerts │  │  Evaluation   │  │ (Latency, Errors) │
# MAGIC │         │  │  (Judges)     │  │                   │
# MAGIC └─────────┘  └──────────────┘  └───────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Query Inference Tables
# MAGIC
# MAGIC Inference tables store every request and response as Delta rows. The table name
# MAGIC is derived from the endpoint name.
# MAGIC
# MAGIC **Note:** It takes a few minutes after the first request for data to appear in
# MAGIC inference tables. If queries return empty results, send some test requests and wait.

# COMMAND ----------

# The inference table naming convention follows the endpoint name
# agents.deploy() stores logs in a system-managed catalog/schema
# Let's find the actual table
inference_table = f"{CATALOG}.{SCHEMA}.`{ENDPOINT_NAME}_payload`"

print(f"Looking for inference table: {inference_table}")
print("=" * 80)

try:
    # Check if the table exists
    tables = spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA} LIKE '*payload*'").collect()
    if tables:
        print("\nAvailable payload tables:")
        for t in tables:
            print(f"  • {t.tableName}")

    # Query recent requests
    df = spark.sql(f"""
        SELECT
            date,
            status_code,
            execution_time_ms,
            request,
            response
        FROM {inference_table}
        ORDER BY date DESC
        LIMIT 10
    """)

    row_count = df.count()
    print(f"\nRecent requests: {row_count}")
    if row_count > 0:
        display(df)
    else:
        print("No data yet. Send some requests to the endpoint and re-run this cell.")

except Exception as e:
    print(f"\n⚠ Could not query inference table: {e}")
    print("\nThis is expected if:")
    print("  • The endpoint was just deployed (tables take a few minutes to create)")
    print("  • No requests have been sent yet")
    print("  • The table naming convention differs")
    print(f"\nTry listing available tables:")
    print(f"  SHOW TABLES IN {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Production Metrics Dashboard
# MAGIC
# MAGIC Aggregate inference table data into daily metrics. In production, you'd schedule
# MAGIC this as a Databricks SQL dashboard with auto-refresh.

# COMMAND ----------

try:
    metrics_df = spark.sql(f"""
        SELECT
            DATE(date) as day,
            COUNT(*) as total_requests,
            ROUND(AVG(execution_time_ms), 0) as avg_latency_ms,
            ROUND(PERCENTILE(execution_time_ms, 0.50), 0) as p50_latency_ms,
            ROUND(PERCENTILE(execution_time_ms, 0.95), 0) as p95_latency_ms,
            ROUND(PERCENTILE(execution_time_ms, 0.99), 0) as p99_latency_ms,
            SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) as successful,
            SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) as errors,
            ROUND(
                SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1
            ) as success_rate_pct
        FROM {inference_table}
        GROUP BY DATE(date)
        ORDER BY day DESC
        LIMIT 14
    """)

    print("Daily Production Metrics (Last 14 Days):")
    print("=" * 80)
    display(metrics_df)

except Exception as e:
    print(f"⚠ Could not compute metrics: {e}")
    print("Metrics will be available once inference table data accumulates.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Analyze Production Traces
# MAGIC
# MAGIC MLflow traces capture **internal execution details** for each request:
# MAGIC - Which tools were called (Vector Search? Genie? Both?)
# MAGIC - How long each LLM call took
# MAGIC - What the LLM received and returned at each step
# MAGIC
# MAGIC This is the same tracing you used in Module 03, now running automatically in production.

# COMMAND ----------

print("Production Traces:")
print("=" * 80)

try:
    traces = mlflow.search_traces(
        experiment_names=[experiment_name],
        max_results=10,
        order_by=["timestamp DESC"],
    )

    if len(traces) > 0:
        print(f"\nFound {len(traces)} recent traces\n")

        for idx, trace in traces.head(5).iterrows():
            print(f"  Trace {idx + 1}:")
            print(f"    Request ID:     {trace.get('request_id', 'N/A')}")
            print(f"    Execution time: {trace.get('execution_time_ms', 'N/A')}ms")
            print(f"    Status:         {trace.get('status', 'N/A')}")
            print()
    else:
        print("No traces found yet.")
        print("Traces appear after requests are processed by the endpoint.")
        print("Send some test requests and re-run this cell.")

except Exception as e:
    print(f"⚠ Could not search traces: {e}")
    print("This may occur if no requests have been processed yet.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Review App Feedback Analysis
# MAGIC
# MAGIC The Review App (enabled in the previous notebook) collects human feedback:
# MAGIC - **Thumbs up/down** on each response
# MAGIC - **Written comments** explaining what was good or bad
# MAGIC - **Flags** for incorrect or inappropriate responses
# MAGIC
# MAGIC This data is stored in assessment tables and is invaluable for improving the agent.

# COMMAND ----------

assessment_table = f"{CATALOG}.{SCHEMA}.`{ENDPOINT_NAME}_assessment_logs`"

try:
    feedback_df = spark.sql(f"""
        SELECT
            text_assessment,
            retrieval_assessment,
            source,
            timestamp
        FROM {assessment_table}
        ORDER BY timestamp DESC
        LIMIT 20
    """)

    row_count = feedback_df.count()
    print(f"Review App Feedback: {row_count} assessments")
    print("=" * 80)

    if row_count > 0:
        display(feedback_df)

        # Summarize ratings
        summary = spark.sql(f"""
            SELECT
                COUNT(*) as total_assessments,
                SUM(CASE WHEN text_assessment.ratings['answer_correct'].value = 'positive'
                     THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN text_assessment.ratings['answer_correct'].value = 'negative'
                     THEN 1 ELSE 0 END) as negative
            FROM {assessment_table}
        """)
        print("\nRating Summary:")
        display(summary)
    else:
        print("No feedback yet. Share the Review App URL with stakeholders.")

except Exception as e:
    print(f"⚠ Could not query assessment table: {e}")
    print("Assessment data appears after stakeholders use the Review App.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Set Up Online Evaluation
# MAGIC
# MAGIC Online evaluation runs your **custom judges from Module 03** on production data.
# MAGIC This gives you continuous, automated quality monitoring without manual review.
# MAGIC
# MAGIC **Recall from Module 03:**
# MAGIC - `hr_accuracy` judge → factual correctness against HR policies
# MAGIC - `completeness` judge → did the response include all necessary details?
# MAGIC
# MAGIC We'll re-create these judges and run them on recent production responses.

# COMMAND ----------

# Re-create the judges from Module 03 (same definitions)
hr_accuracy_judge = make_judge(
    name="hr_accuracy",
    instructions=(
        "Evaluate if the response in {{ outputs }} is factually accurate "
        "according to HR policies.\n\n"
        "Check for:\n"
        "- Correct numbers and dates\n"
        "- Accurate policy details\n"
        "- Appropriate caveats and eligibility requirements\n\n"
        "Rate as:\n"
        "- 'excellent' = Perfectly accurate with all details correct\n"
        "- 'good' = Accurate with minor details missing\n"
        "- 'fair' = Mostly accurate but missing important details\n"
        "- 'poor' = Partially accurate with significant errors\n"
        "- 'very_poor' = Incorrect or misleading"
    ),
    feedback_value_type=Literal["excellent", "good", "fair", "poor", "very_poor"],
    model=f"databricks:/{LLM_ENDPOINT}",
)

completeness_judge = make_judge(
    name="completeness",
    instructions=(
        "Evaluate if the response in {{ outputs }} is complete and provides "
        "all necessary information.\n\n"
        "Check if important details are included:\n"
        "- Eligibility requirements\n"
        "- Specific rates or amounts\n"
        "- Important caveats or conditions\n"
        "- Next steps or how to take action\n\n"
        "Rate as:\n"
        "- 'excellent' = Complete with all necessary details\n"
        "- 'good' = Mostly complete with one minor detail missing\n"
        "- 'fair' = Adequate but missing some important information\n"
        "- 'poor' = Incomplete, missing several key details\n"
        "- 'very_poor' = Very incomplete or too vague"
    ),
    feedback_value_type=Literal["excellent", "good", "fair", "poor", "very_poor"],
    model=f"databricks:/{LLM_ENDPOINT}",
)

print("✓ Re-created judges from Module 03:")
print("  • hr_accuracy — factual correctness")
print("  • completeness — information completeness")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Run Online Evaluation on Production Data
# MAGIC
# MAGIC Pull recent successful production responses and evaluate them with the judges.
# MAGIC
# MAGIC In production, you'd schedule this to run hourly or daily as a Databricks Job,
# MAGIC tracking quality trends over time in your MLflow experiment.

# COMMAND ----------

try:
    # Pull recent successful responses from inference tables
    recent_data = spark.sql(f"""
        SELECT
            request:messages[0].content as question,
            response:output[0].content as answer
        FROM {inference_table}
        WHERE status_code = 200
        ORDER BY date DESC
        LIMIT 20
    """).toPandas()

    if len(recent_data) > 0:
        # Format for MLflow evaluation
        eval_data = []
        for _, row in recent_data.iterrows():
            eval_data.append({
                "inputs": {"question": row["question"]},
                "outputs": row["answer"],
            })

        print(f"Evaluating {len(eval_data)} recent production responses...")
        print("(Running hr_accuracy and completeness judges)")
        print()

        results = mlflow.genai.evaluate(
            data=eval_data,
            scorers=[hr_accuracy_judge, completeness_judge],
        )

        print("✓ Online evaluation complete!")
        print("\nAggregate Metrics:")
        print("=" * 80)
        if hasattr(results, 'metrics'):
            for metric_name, value in results.metrics.items():
                print(f"  {metric_name}: {value}")

        # Show detailed results
        if results.tables:
            table_key = list(results.tables.keys())[0]
            results_df = results.tables[table_key]
            print(f"\nDetailed Results ({len(results_df)} responses evaluated):")
            display(results_df)

            # Identify issues
            for col in results_df.columns:
                if 'value' in col.lower():
                    poor_count = results_df[results_df[col].isin(['poor', 'very_poor'])].shape[0]
                    if poor_count > 0:
                        judge_name = col.split('/')[0]
                        print(f"\n⚠ {judge_name}: {poor_count} responses rated poor/very_poor")
    else:
        print("No production data available yet for evaluation.")
        print("Send requests to the endpoint and re-run this cell.")

except Exception as e:
    print(f"⚠ Could not run online evaluation: {e}")
    print("This requires production request data in inference tables.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Scheduling Online Evaluation
# MAGIC
# MAGIC To run evaluation continuously, schedule this notebook (or a production version)
# MAGIC as a **Databricks Job**:
# MAGIC
# MAGIC ```
# MAGIC Schedule: Every hour (or daily, depending on traffic)
# MAGIC
# MAGIC What it does:
# MAGIC   1. Pull last N responses from inference table
# MAGIC   2. Run judges on each response
# MAGIC   3. Log aggregate metrics to MLflow
# MAGIC   4. Flag any "poor" or "very_poor" responses
# MAGIC ```
# MAGIC
# MAGIC This creates a time series of quality metrics you can track in MLflow:
# MAGIC
# MAGIC | Day | Requests | Avg Accuracy | Avg Completeness | Poor Responses |
# MAGIC |-----|----------|-------------|------------------|----------------|
# MAGIC | Mon | 342 | excellent | good | 2 |
# MAGIC | Tue | 289 | good | good | 5 |
# MAGIC | Wed | 401 | excellent | excellent | 1 |
# MAGIC
# MAGIC A spike in "poor" responses tells you something changed — maybe a data source
# MAGIC was updated, an MCP tool went down, or the LLM behavior shifted.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Quality Alert Queries
# MAGIC
# MAGIC Define SQL queries that can be scheduled as **Databricks SQL Alerts**.
# MAGIC These fire automatically when quality degrades.

# COMMAND ----------

print("Production Monitoring Queries:")
print("=" * 80)

# Query 1: Error rate alert
print("\n1. Error Rate Alert (trigger if > 5% errors in last 24h):")
print("-" * 60)
print(f"""
SELECT
    COUNT(*) as total_requests,
    SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) as errors,
    ROUND(
        SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1
    ) as error_rate_pct
FROM {inference_table}
WHERE date >= CURRENT_TIMESTAMP - INTERVAL 24 HOURS
HAVING error_rate_pct > 5.0
""")

# Query 2: Latency alert
print("\n2. Latency Alert (trigger if p95 > 10 seconds):")
print("-" * 60)
print(f"""
SELECT
    ROUND(PERCENTILE(execution_time_ms, 0.95), 0) as p95_latency_ms
FROM {inference_table}
WHERE date >= CURRENT_TIMESTAMP - INTERVAL 24 HOURS
HAVING p95_latency_ms > 10000
""")

# Query 3: Negative feedback alert
print("\n3. Negative Feedback Alert (trigger if > 20% negative in last 24h):")
print("-" * 60)
print(f"""
SELECT
    COUNT(*) as total_assessments,
    SUM(CASE WHEN text_assessment.ratings['answer_correct'].value = 'negative'
        THEN 1 ELSE 0 END) as negative_count,
    ROUND(
        SUM(CASE WHEN text_assessment.ratings['answer_correct'].value = 'negative'
            THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1
    ) as negative_rate_pct
FROM {assessment_table}
WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL 24 HOURS
HAVING negative_rate_pct > 20.0
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 12: Setting Up Alerts in Databricks SQL
# MAGIC
# MAGIC To make these queries into automated alerts:
# MAGIC
# MAGIC 1. Navigate to **Databricks SQL → SQL Editor**
# MAGIC 2. Create each query from Step 11
# MAGIC 3. Click **Create Alert** on the query
# MAGIC 4. Set the trigger condition (e.g., "rows returned > 0")
# MAGIC 5. Configure notification destination (email, Slack, PagerDuty)
# MAGIC 6. Set schedule (e.g., every hour)
# MAGIC
# MAGIC **Recommended alert configuration:**
# MAGIC
# MAGIC | Alert | Threshold | Schedule | Notification |
# MAGIC |-------|-----------|----------|-------------|
# MAGIC | Error rate | > 5% | Every hour | Slack + Email |
# MAGIC | P95 latency | > 10s | Every hour | Slack |
# MAGIC | Negative feedback | > 20% | Every 4 hours | Email |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### What We Did
# MAGIC - ✅ Queried inference tables for production request/response data
# MAGIC - ✅ Built a daily metrics dashboard (volume, latency, success rate)
# MAGIC - ✅ Analyzed production traces with MLflow
# MAGIC - ✅ Examined Review App feedback from stakeholders
# MAGIC - ✅ Re-used custom judges from Module 03 for online evaluation
# MAGIC - ✅ Defined SQL monitoring queries for automated alerts
# MAGIC
# MAGIC ### Key Concepts
# MAGIC
# MAGIC **Three monitoring pillars:**
# MAGIC
# MAGIC | Pillar | Data Source | What You Learn |
# MAGIC |--------|-----------|---------------|
# MAGIC | **Inference Tables** | Request/response Delta logs | Volume, errors, latency trends |
# MAGIC | **MLflow Traces** | Internal execution spans | Which tools are used, where time is spent |
# MAGIC | **Online Evaluation** | Judge scores on production data | Quality trends, degradation detection |
# MAGIC
# MAGIC **Three feedback signals:**
# MAGIC
# MAGIC | Signal | Source | Speed | Automation |
# MAGIC |--------|-------|-------|------------|
# MAGIC | SQL Alerts | Inference tables | Real-time | Fully automated |
# MAGIC | Online Evaluation | Judges on production data | Scheduled (hourly/daily) | Fully automated |
# MAGIC | Review App | Human stakeholders | Continuous | Human-in-the-loop |
# MAGIC
# MAGIC **Production checklist:**
# MAGIC - ✅ Inference tables enabled (automatic with `agents.deploy()`)
# MAGIC - ✅ MLflow tracing active (automatic with `agents.deploy()`)
# MAGIC - ✅ Review App enabled for stakeholder feedback
# MAGIC - ✅ Error rate alert defined
# MAGIC - ✅ Latency alert defined
# MAGIC - ✅ Negative feedback alert defined
# MAGIC - ✅ Online evaluation judges configured
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC Continue to [04_end_to_end_deployment.py](04_end_to_end_deployment.py) for the
# MAGIC capstone: a complete deployment pipeline that ties all modules together.

# COMMAND ----------

print("✓ Production monitoring setup complete!")
print("\nMonitoring stack:")
print("  ✓ Inference tables — request/response logging")
print("  ✓ MLflow traces — internal execution visibility")
print("  ✓ Online evaluation — automated judge-based quality")
print("  ✓ SQL alerts — error rate, latency, feedback")
print("\nNext: End-to-end deployment capstone")
