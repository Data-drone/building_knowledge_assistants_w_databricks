# Databricks notebook source
# MAGIC %md
# MAGIC # Building a Doc Store on Vector Search
# MAGIC
# MAGIC ## What You'll Build
# MAGIC Upgrade your agent from Notebook 00 by adding a **tool**:
# MAGIC - Create a Vector Search index over your documents
# MAGIC - Give your agent the ability to search these documents
# MAGIC - Build a RAG (Retrieval Augmented Generation) agent
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC - ✅ Prepare documents for Vector Search (chunking)
# MAGIC - ✅ Create a Delta Sync Vector Search index
# MAGIC - ✅ Wrap Vector Search as a LangChain tool
# MAGIC - ✅ Add tool to LangGraph agent
# MAGIC - ✅ Test RAG pattern
# MAGIC
# MAGIC ## What's RAG?
# MAGIC **RAG = Retrieval + Generation**
# MAGIC 1. **Retrieve** relevant documents from your data
# MAGIC 2. **Generate** answers using LLM + retrieved context
# MAGIC
# MAGIC Let's build it!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Install required packages
%pip install -q --upgrade \
  mlflow[databricks]>=3.1.0 \
  databricks-langchain>=0.8.0 \
  databricks-vectorsearch>=0.30 \
  langgraph>=0.2.50 \
  langchain-core>=0.3.0

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports & Configuration

# COMMAND ----------

import mlflow
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from databricks.vector_search.client import VectorSearchClient
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

# Configuration
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"
DOCS_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/docs"

# Derived names
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.policy_chunks"
VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.policy_index"
#ENDPOINT_NAME = "agent_bootcamp_endpoint"
ENDPOINT_NAME = "one-env-shared-endpoint-15"

# Enable MLflow auto-logging
mlflow.langchain.autolog()
username = spark.sql("SELECT current_user()").collect()[0][0]
mlflow.set_experiment(f"/Users/{username}/01_doc_store")

print("✓ Configuration loaded")
print(f"  Chunks Table: {CHUNKS_TABLE}")
print(f"  Vector Index: {VECTOR_INDEX}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Prepare Documents for Vector Search
# MAGIC
# MAGIC Before we can search documents, we need to:
# MAGIC 1. Load documents from storage
# MAGIC 2. **Chunk** them into smaller pieces
# MAGIC 3. Store chunks in a Delta table
# MAGIC
# MAGIC ### Why Chunk?
# MAGIC - Embedding models have token limits (512-8192 tokens)
# MAGIC - Smaller chunks = more precise retrieval
# MAGIC - Can fit multiple relevant chunks in LLM context

# COMMAND ----------

# Load markdown files from volume
files = dbutils.fs.ls(DOCS_VOLUME)

documents = []
for file in files:
    if file.name.endswith('.md'):
        content = dbutils.fs.head(file.path)
        documents.append({
            "filename": file.name,
            "content": content
        })

print(f"✓ Loaded {len(documents)} documents")
for doc in documents:
    print(f"  - {doc['filename']}")

# COMMAND ----------

# Simple chunking function (avoids numpy compatibility issues)
def chunk_text(text, chunk_size=800, chunk_overlap=100):
    """Split text into overlapping chunks."""
    chunks = []
    # Try splitting by markdown headers first
    sections = text.split("\n## ")

    for section in sections:
        # If section is small enough, keep it
        if len(section) <= chunk_size:
            if section.strip():
                chunks.append(section.strip())
        else:
            # Split large sections by paragraphs
            paragraphs = section.split("\n\n")
            current_chunk = ""

            for para in paragraphs:
                if len(current_chunk) + len(para) <= chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

    return chunks

# Chunk all documents
import hashlib

chunks_data = []
for doc in documents:
    chunks = chunk_text(doc["content"])

    for idx, chunk in enumerate(chunks):
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
        chunks_data.append({
            "chunk_id": f"{doc['filename']}_{idx}_{chunk_hash}",
            "source_file": doc['filename'],
            "chunk_text": chunk
        })

print(f"✓ Created {len(chunks_data)} chunks")
print(f"\nSample chunk:\n{chunks_data[0]['chunk_text'][:200]}...")

# COMMAND ----------

# Save to Delta table with Change Data Feed enabled (required for Delta Sync)
chunks_df = spark.createDataFrame(chunks_data)

chunks_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .option("delta.enableChangeDataFeed", "true") \
    .saveAsTable(CHUNKS_TABLE)

# Ensure Change Data Feed is enabled (in case table already existed)
spark.sql(f"ALTER TABLE {CHUNKS_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"✓ Saved to {CHUNKS_TABLE}")
print(f"✓ Change Data Feed enabled (required for Delta Sync)")
display(spark.sql(f"SELECT * FROM {CHUNKS_TABLE} LIMIT 3"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create Vector Search Index
# MAGIC
# MAGIC Vector Search enables semantic similarity search:
# MAGIC 1. Converts text to embeddings (dense vectors)
# MAGIC 2. Finds similar chunks using vector math
# MAGIC 3. Returns most relevant documents
# MAGIC
# MAGIC We'll use **Delta Sync** which auto-syncs with the Delta table.

# COMMAND ----------

vsc = VectorSearchClient()

# Create or get endpoint
try:
    vsc.create_endpoint(name=ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"✓ Created endpoint: {ENDPOINT_NAME}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"✓ Using existing endpoint: {ENDPOINT_NAME}")
    else:
        raise

# Check endpoint status
import time
try:
    endpoint = vsc.get_endpoint(ENDPOINT_NAME)
    # Try different ways to access the state
    if hasattr(endpoint, 'endpoint_status'):
        state = endpoint.endpoint_status.get('state', 'UNKNOWN')
    elif hasattr(endpoint, 'state'):
        state = endpoint.state
    elif isinstance(endpoint, dict):
        state = endpoint.get('endpoint_status', {}).get('state', endpoint.get('state', 'UNKNOWN'))
    else:
        state = 'UNKNOWN'

    print(f"✓ Endpoint exists: {ENDPOINT_NAME}")
    print(f"  Current state: {state}")

    # Only wait if explicitly not ONLINE
    if state and state != "ONLINE":
        print("  Waiting for endpoint to come online...")
        max_wait = 300  # 5 minutes
        waited = 0
        while waited < max_wait and state != "ONLINE":
            time.sleep(30)
            waited += 30
            endpoint = vsc.get_endpoint(ENDPOINT_NAME)
            if hasattr(endpoint, 'endpoint_status'):
                state = endpoint.endpoint_status.get('state', 'UNKNOWN')
            elif hasattr(endpoint, 'state'):
                state = endpoint.state
            print(f"  State: {state}")
            if state == "ONLINE":
                break

    if state == "ONLINE":
        print("✓ Endpoint is ONLINE")
    else:
        print(f"⚠ Continuing anyway (state: {state})")

except Exception as e:
    print(f"⚠ Could not verify endpoint status: {e}")
    print("  Continuing anyway...")

# COMMAND ----------

# Create Delta Sync index
try:
    vsc.create_delta_sync_index(
        endpoint_name=ENDPOINT_NAME,
        index_name=VECTOR_INDEX,
        source_table_name=CHUNKS_TABLE,
        pipeline_type="TRIGGERED",
        primary_key="chunk_id",
        embedding_source_column="chunk_text",
        embedding_model_endpoint_name="databricks-gte-large-en"
    )
    print(f"✓ Created index: {VECTOR_INDEX}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"✓ Using existing index: {VECTOR_INDEX}")
    else:
        raise

# Wait for index to be ready (critical - don't skip this!)
print("\nWaiting for index to sync (this can take 3-5 minutes)...")
max_wait_index = 600  # 10 minutes
waited_index = 0
index_ready = False

while waited_index < max_wait_index:
    try:
        index_info = vsc.get_index(ENDPOINT_NAME, VECTOR_INDEX)

        # The describe() method returns a dict with status info
        describe = index_info.describe()
        state = describe.get("status", {}).get("detailed_state", "UNKNOWN")

        print(f"  Index state: {state} (waited {waited_index}s)")

        if state in ["ONLINE", "ONLINE_NO_PENDING_UPDATE"]:
            print(f"✓ Index is ready: {state}")
            index_ready = True
            break

        if state == "PROVISIONING" or state == "SYNCING" or "SYNC" in state:
            print(f"  Still syncing... (this is normal)")

    except Exception as e:
        print(f"  Checking status... ({e})")

    time.sleep(30)
    waited_index += 30

if not index_ready:
    print(f"⚠ Warning: Index may not be fully ready yet")
    print(f"  If search fails below, wait a few minutes and re-run this notebook")

# COMMAND ----------

# Test similarity search
index = vsc.get_index(ENDPOINT_NAME, VECTOR_INDEX)

results = index.similarity_search(
    query_text="vacation policy",
    columns=["source_file", "chunk_text"],
    num_results=2
)

print("Test Query: 'vacation policy'\n")
for i, row in enumerate(results.get("result", {}).get("data_array", []), 1):
    print(f"{i}. {row[0]}")
    print(f"   {row[1][:150]}...\n")

print("✓ Vector Search is working!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Wrap Vector Search as a Tool
# MAGIC
# MAGIC To give our agent access to Vector Search, we use the built-in Databricks LangChain retriever tool instead of writing a custom `@tool` function.
# MAGIC
# MAGIC This keeps the notebook simpler while still showing the same RAG pattern you'll use in LangGraph.

# COMMAND ----------

search_policy_documents = VectorSearchRetrieverTool(
    index_name=VECTOR_INDEX,
    tool_name="search_policy_documents",
    tool_description=(
        "Search company policy documents for information about vacation and leave "
        "policies, remote work, professional development, benefits, and equipment."
    ),
    columns=["source_file", "chunk_text"],
    text_column="chunk_text",
    num_results=3,
)

# Test the tool
print("Testing tool:")
result = search_policy_documents.invoke("vacation")
print(str(result)[:300] + "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Build Agent with Tool
# MAGIC
# MAGIC Now we add the Vector Search tool to our agent from Notebook 00 and give it a prompt template so the bot responds consistently.

# COMMAND ----------

# Initialize LLM
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1)

rag_bot_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a beginner-friendly Databricks coach helping users answer questions "
        "about company policy documents. Use retrieved context when it is available, "
        "cite source file names when the tool provides them, and do not make up policy "
        "details that are not in the documents. Answer in 2 short bullet points and end "
        "with one practical next step."
    ),
    MessagesPlaceholder("messages"),
])

# Bind tools
tools = [search_policy_documents]
llm_with_tools = llm.bind_tools(tools)

# Agent logic
def call_agent(state: MessagesState):
    prompt_messages = rag_bot_prompt.invoke({"messages": state["messages"]}).messages
    response = llm_with_tools.invoke(prompt_messages)
    return {"messages": [response]}

# Routing logic
def should_continue(state: MessagesState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

# Build graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_agent)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

agent = workflow.compile()

print("✓ RAG Agent built!")
print("✓ Prompt template added")
print("\nAgent flow:")
print("  User → Agent (decides to search?) → Vector Search → Agent (generates answer) → User")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Test Your RAG Agent
# MAGIC
# MAGIC Ask questions about your documents!

# COMMAND ----------

def ask(question: str):
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}\n")

    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    answer = result["messages"][-1].content
    print(f"A: {answer}\n")

    return result

# COMMAND ----------

# Test query 1
ask("How much vacation time do full-time employees get?")

# COMMAND ----------

# Test query 2
ask("What are the core in-office days for hybrid workers?")

# COMMAND ----------

# Test query 3: Edge case - information NOT in documents
ask("What is the cryptocurrency investment policy?")
# Agent should acknowledge lack of information

# COMMAND ----------

# MAGIC %md
# MAGIC ## What You Built 🎉
# MAGIC
# MAGIC You upgraded from a simple chatbot to a RAG agent!
# MAGIC
# MAGIC ### Key Concepts
# MAGIC - ✅ **Document Chunking**: Breaking text into searchable pieces
# MAGIC - ✅ **Vector Search**: Semantic similarity search with embeddings
# MAGIC - ✅ **Tools**: Extending agent capabilities beyond just LLM
# MAGIC - ✅ **RAG Pattern**: Retrieval + Generation for factual answers
# MAGIC
# MAGIC ### From Notebook 00 to Now
# MAGIC - **Notebook 00**: LLM + conversation loop (simple prompt bot)
# MAGIC - **Notebook 01**: + Vector Search tool (RAG agent)
# MAGIC
# MAGIC ### What's Still Missing?
# MAGIC - Memory (multi-turn with context from earlier in conversation)
# MAGIC - Multiple tools (Genie for data, UC Functions)
# MAGIC - Evaluation framework
# MAGIC - Deployment
# MAGIC
# MAGIC → These are covered in later modules!
# MAGIC
# MAGIC ## Deep Dives
# MAGIC Want to learn more?
# MAGIC - **Chunking strategies**: See other sections of this tutorial
# MAGIC - **Vector Search optimization**: See other sections of this tutorial
# MAGIC - **Advanced agent patterns**: See other sections of this tutorial

# COMMAND ----------

# MAGIC %md
# MAGIC ## Try It Yourself

# COMMAND ----------

# Exercise: Ask your own questions
your_question = "What equipment does the company provide for remote workers?"
ask(your_question)
