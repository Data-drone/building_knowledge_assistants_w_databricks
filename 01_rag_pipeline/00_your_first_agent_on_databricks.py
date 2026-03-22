# Databricks notebook source
# MAGIC %md
# MAGIC # Your First Agent on Databricks
# MAGIC
# MAGIC Build a simple AI agent in 15 minutes using Claude Sonnet 4.5, LangGraph, and MLflow!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies

# COMMAND ----------

%pip install -q --upgrade mlflow[databricks]>=3.1.0 databricks-langchain>=0.8.0 langgraph>=0.2.50 langchain-core>=0.3.0
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Setup and Configuration

# COMMAND ----------

import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, MessagesState, END

# Configuration
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"

# Setup MLflow
mlflow.langchain.autolog()
username = spark.sql("SELECT current_user()").collect()[0][0]
experiment_path = f"/Users/{username}/00_first_agent"
mlflow.set_experiment(experiment_path)

print("✓ Configuration complete")
print(f"  LLM Endpoint: {LLM_ENDPOINT}")
print(f"  Experiment: {experiment_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Test Direct LLM Call

# COMMAND ----------

# Initialize LLM
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.7)

# Make a test call
messages = [HumanMessage(content="Hello! What can you help me with?")]
response = llm.invoke(messages)

print("User:", messages[0].content)
print("AI:", response.content[:200])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Build LangGraph Agent

# COMMAND ----------

# Define agent node
def call_llm(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build workflow
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_llm)
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)
agent = workflow.compile()

print("✓ Agent built successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Test the Agent

# COMMAND ----------

# Test conversation
result = agent.invoke({
    "messages": [HumanMessage(content="Explain what an AI agent is in one sentence.")]
})

print("User: Explain what an AI agent is in one sentence.")
print(f"Agent: {result['messages'][-1].content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Add a Prompt Template

# COMMAND ----------

basic_bot_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a beginner-friendly Databricks coach with a playful, Yoda-inspired voice. "
        "Use simple words, slightly inverted sentence structure, and a wise, encouraging tone. "
        "Answer in 2 short bullet points and end with one practical next step."
    ),
    ("human", "{user_input}"),
])


def call_prompted_llm(state: MessagesState):
    latest_user_message = next(
        message.content
        for message in reversed(state["messages"])
        if isinstance(message, HumanMessage)
    )
    prompt_messages = basic_bot_prompt.invoke({"user_input": latest_user_message}).messages
    response = llm.invoke(prompt_messages)
    return {"messages": [response]}


prompted_workflow = StateGraph(MessagesState)
prompted_workflow.add_node("prompted_agent", call_prompted_llm)
prompted_workflow.set_entry_point("prompted_agent")
prompted_workflow.add_edge("prompted_agent", END)
prompted_agent = prompted_workflow.compile()

print("✓ Prompt-templated agent built successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Compare Both Bots
# MAGIC
# MAGIC Ask both bots the same question so you can see how the prompt changes the response style.

# COMMAND ----------

comparison_question = "How do I start building an AI agent on Databricks?"

basic_result = agent.invoke({
    "messages": [HumanMessage(content=comparison_question)]
})
prompted_result = prompted_agent.invoke({
    "messages": [HumanMessage(content=comparison_question)]
})

print(f"User: {comparison_question}")
print()
print("Minimal Bot:")
print(basic_result["messages"][-1].content)
print()
print("Prompt-Templated Bot:")
print(f"Prompted Agent: {prompted_result['messages'][-1].content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC You've built two versions of your first agent! Key concepts:
# MAGIC - **LLM**: ChatDatabricks wraps the endpoint
# MAGIC - **State**: MessagesState tracks conversation
# MAGIC - **Graph**: LangGraph creates the agent loop
# MAGIC - **Minimal bot**: Sends the user's message directly to the model
# MAGIC - **Prompted bot**: Uses a prompt template to shape tone, structure, and personality
# MAGIC - **MLflow**: Auto-logging captures traces
# MAGIC
# MAGIC Next: **Notebook 01** - Add Vector Search for RAG!

# COMMAND ----------

print("✓ Tutorial complete!")
print(f"View traces in MLflow: {experiment_path}")
