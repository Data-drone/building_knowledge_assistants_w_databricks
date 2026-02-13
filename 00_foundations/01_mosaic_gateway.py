# Databricks notebook source
# MAGIC %md
# MAGIC # Mosaic AI Gateway: LLM Endpoints
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - Understand Mosaic AI Gateway architecture
# MAGIC - List available Foundation Model endpoints
# MAGIC - Test chat completions with different models
# MAGIC - Compare pay-per-token vs provisioned throughput
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Completed [00_setup.py](00_setup.py)
# MAGIC - Access to Foundation Model endpoints
# MAGIC
# MAGIC ## Estimated Time
# MAGIC 10 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Required Packages

# COMMAND ----------

# MAGIC %pip install --upgrade langchain-databricks

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Import Dependencies

# COMMAND ----------

import sys
sys.path.append("..")

from config import LLM_ENDPOINT, get_workspace_client
from langchain_databricks import ChatDatabricks
from databricks.sdk import WorkspaceClient
import time

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. What is Mosaic AI Gateway?
# MAGIC
# MAGIC Mosaic AI Gateway is Databricks' unified interface for accessing LLMs:
# MAGIC
# MAGIC ### Key Features
# MAGIC - **Multi-provider support**: OpenAI, Anthropic, Cohere, Meta, Mistral, etc.
# MAGIC - **Pay-per-token**: No infrastructure management required
# MAGIC - **Provisioned throughput**: Reserved capacity for production workloads
# MAGIC - **Unified API**: Single endpoint pattern across all providers
# MAGIC - **Built-in governance**: Unity Catalog permissions and audit logs
# MAGIC - **Automatic scaling**: Handle traffic spikes without config changes
# MAGIC
# MAGIC ### Endpoint Types
# MAGIC
# MAGIC | Type | Use Case | Billing | Latency |
# MAGIC |------|----------|---------|---------|
# MAGIC | Pay-per-token | Development, variable workloads | Per token | Higher variance |
# MAGIC | Provisioned throughput | Production, predictable load | Reserved capacity | Lower, consistent |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. List Available Endpoints
# MAGIC
# MAGIC Foundation Model endpoints are pre-configured by Databricks.

# COMMAND ----------

w = get_workspace_client()

print("Available Foundation Model Endpoints:")
print("=" * 80)

# List serving endpoints
endpoints = w.serving_endpoints.list()

# Filter for Foundation Models (typically start with "databricks-")
foundation_models = [
    ep for ep in endpoints
    if ep.name.startswith("databricks-")
]

for ep in foundation_models:
    print(f"\nName: {ep.name}")
    print(f"  State: {ep.state}")
    print(f"  Task: {ep.task}")

print(f"\n✓ Found {len(foundation_models)} Foundation Model endpoints")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test Chat Completion
# MAGIC
# MAGIC LangChain's `ChatDatabricks` provides a convenient interface to Mosaic AI Gateway.

# COMMAND ----------

# Initialize LLM
llm = ChatDatabricks(
    endpoint=LLM_ENDPOINT,
    temperature=0.7,
    max_tokens=500
)

print(f"Using endpoint: {LLM_ENDPOINT}\n")

# Test simple completion
response = llm.invoke("Explain what Unity Catalog is in one sentence.")

print(f"Response: {response.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Compare Model Latency
# MAGIC
# MAGIC Test different models to understand latency characteristics.

# COMMAND ----------

# Test models
test_models = [
    "databricks-claude-sonnet-4-5",  # High capability
    "databricks-claude-haiku-4",     # Fast, efficient
    "databricks-llama-3-3-70b",      # Open source
]

test_prompt = "What are the key benefits of using LangGraph for building agents?"

print("Model Latency Comparison")
print("=" * 80)

for model_name in test_models:
    try:
        llm = ChatDatabricks(
            endpoint=model_name,
            temperature=0.1,
            max_tokens=200
        )

        start = time.time()
        response = llm.invoke(test_prompt)
        latency = time.time() - start

        print(f"\n{model_name}")
        print(f"  Latency: {latency:.2f}s")
        print(f"  Tokens: ~{len(response.content.split())} words")
        print(f"  Preview: {response.content[:100]}...")

    except Exception as e:
        print(f"\n{model_name}")
        print(f"  ✗ Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Understanding Pay-Per-Token vs Provisioned
# MAGIC
# MAGIC ### Pay-Per-Token (Default)
# MAGIC ```python
# MAGIC llm = ChatDatabricks(endpoint="databricks-claude-sonnet-4-5")
# MAGIC # Automatically uses pay-per-token pricing
# MAGIC # Billed based on input/output tokens consumed
# MAGIC ```
# MAGIC
# MAGIC **Best for:**
# MAGIC - Development and testing
# MAGIC - Variable/unpredictable workloads
# MAGIC - Low-volume production use
# MAGIC
# MAGIC **Considerations:**
# MAGIC - Shared capacity (higher latency variance)
# MAGIC - Rate limits apply
# MAGIC - Cost scales linearly with usage
# MAGIC
# MAGIC ### Provisioned Throughput
# MAGIC ```python
# MAGIC # Create provisioned endpoint via Databricks UI or API
# MAGIC # Reserve specific throughput (e.g., "small", "medium", "large")
# MAGIC
# MAGIC llm = ChatDatabricks(endpoint="my-provisioned-claude")
# MAGIC # Guaranteed capacity, lower latency
# MAGIC ```
# MAGIC
# MAGIC **Best for:**
# MAGIC - Production applications
# MAGIC - Latency-sensitive workloads
# MAGIC - Predictable high-volume usage
# MAGIC
# MAGIC **Considerations:**
# MAGIC - Fixed hourly cost (regardless of usage)
# MAGIC - Lower per-token cost at scale
# MAGIC - Consistent, low latency

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Streaming Responses
# MAGIC
# MAGIC Stream tokens for better user experience in chat applications.

# COMMAND ----------

llm = ChatDatabricks(
    endpoint=LLM_ENDPOINT,
    temperature=0.5
)

print("Streaming response:")
print("-" * 80)

# Stream response chunks
for chunk in llm.stream("Explain the concept of vector embeddings in 3 sentences."):
    print(chunk.content, end="", flush=True)

print("\n" + "-" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Token Usage Tracking
# MAGIC
# MAGIC Monitor token consumption for cost management.

# COMMAND ----------

from langchain_core.messages import HumanMessage

llm = ChatDatabricks(
    endpoint=LLM_ENDPOINT,
    temperature=0.1
)

# Invoke with response metadata
response = llm.invoke([
    HumanMessage(content="What is the purpose of checkpointing in LangGraph?")
])

# Access token usage (if available in response metadata)
print("Response Metadata:")
print(f"  Content length: {len(response.content)} characters")
print(f"  Estimated tokens: ~{len(response.content) // 4} (approx 4 chars/token)")

# Note: Actual token counts available via MLflow tracing (covered in module 03)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Try It Yourself
# MAGIC
# MAGIC Experiment with different parameters:

# COMMAND ----------

# Exercise: Test temperature effects
temperatures = [0.0, 0.5, 1.0]
prompt = "Generate a creative name for an AI-powered HR assistant."

print("Temperature Effect on Creativity")
print("=" * 80)

for temp in temperatures:
    llm = ChatDatabricks(
        endpoint=LLM_ENDPOINT,
        temperature=temp
    )

    response = llm.invoke(prompt)
    print(f"\nTemperature {temp}: {response.content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC You've learned how to:
# MAGIC
# MAGIC - ✅ List and access Foundation Model endpoints
# MAGIC - ✅ Make chat completion requests via `ChatDatabricks`
# MAGIC - ✅ Compare model latency characteristics
# MAGIC - ✅ Understand pay-per-token vs provisioned throughput tradeoffs
# MAGIC - ✅ Stream responses for better UX
# MAGIC
# MAGIC ## Key Takeaways
# MAGIC
# MAGIC 1. **Use pay-per-token for development** - simpler, no capacity planning
# MAGIC 2. **Choose provisioned for production** - better latency, cost predictability
# MAGIC 3. **Select model based on use case**:
# MAGIC    - Sonnet 4.5: Complex reasoning, multi-step tasks
# MAGIC    - Haiku 4: Fast responses, simple queries
# MAGIC    - Open source (Llama): Cost optimization, data sovereignty
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC Continue to [02_platform_orientation.py](02_platform_orientation.py) to understand Databricks platform concepts.
