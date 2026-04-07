"""
Shared configuration for Databricks Agent Bootcamp.

This module contains all the configuration variables used across the tutorial notebooks.
Update these values to match your Databricks workspace environment.
"""

import os
from databricks.sdk import WorkspaceClient

# ==============================================================================
# UNITY CATALOG CONFIGURATION
# ==============================================================================

# Unity Catalog namespace for all assets
CATALOG = "agent_bootcamp"
SCHEMA = "knowledge_assistant"

# Full qualified namespace
UC_NAMESPACE = f"{CATALOG}.{SCHEMA}"

# ==============================================================================
# WORKSPACE CONFIGURATION
# ==============================================================================

# Databricks workspace host (e.g., "https://dbc-a1b2c3d4-e5f6.cloud.databricks.com")
HOST = os.getenv("DATABRICKS_HOST", "https://adb-984752964297111.11.azuredatabricks.net")

# Cloud region for Lakebase (e.g., "us-west-2", "eastus", "eu-west-1")
REGION = "eastus"

# ==============================================================================
# MODEL ENDPOINTS
# ==============================================================================

# Default LLM endpoint for agent reasoning
LLM_ENDPOINT = "databricks-claude-sonnet-4-6"

# Embedding endpoint for vector search
EMBEDDING_ENDPOINT = "databricks-gte-large-en"

# ==============================================================================
# LAKEBASE CONFIGURATION
# ==============================================================================

# Lakebase project name
LAKEBASE_PROJECT = "knowledge-assistant-state"

# Default branch for development
LAKEBASE_BRANCH = "development"

# Production branch name
LAKEBASE_PRODUCTION_BRANCH = "production"

# ==============================================================================
# GENIE CONFIGURATION
# ==============================================================================

# Genie space ID (update after creating space in notebook 04_genie/01_genie_space.py)
GENIE_SPACE_ID = "01234567-89ab-cdef-0123-456789abcdef"  # Placeholder

# ==============================================================================
# DATA PATHS
# ==============================================================================

# Volume for storing source documents
DOCS_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/source_docs"

# Table for chunked documents
CHUNKS_TABLE = f"{UC_NAMESPACE}.document_chunks"

# Vector search index name
VECTOR_INDEX = f"{UC_NAMESPACE}.document_chunks_index"

# Sample data tables for Genie
EMPLOYEE_TABLE = f"{UC_NAMESPACE}.employee_data"
LEAVE_BALANCES_TABLE = f"{UC_NAMESPACE}.leave_balances"

# ==============================================================================
# APPS DEPLOYMENT CONFIGURATION
# ==============================================================================

APP_NAME = "knowledge-assistant-agent-app"
APP_EXPERIMENT = "/Shared/knowledge_assistant_agent_app"


def get_app_url(app_name: str) -> str:
    """Return the base URL for a deployed Databricks App."""
    w = get_workspace_client()
    app = w.apps.get(app_name)
    return f"https://{app.url}"


# ==============================================================================
# MLFLOW CONFIGURATION
# ==============================================================================

# MLflow experiment path
MLFLOW_EXPERIMENT = f"/Users/{os.getenv('USER', 'default')}/agent_bootcamp"

# Model registry name for deployment
MODEL_NAME = f"{UC_NAMESPACE}.knowledge_assistant"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_workspace_client() -> WorkspaceClient:
    """
    Get authenticated Databricks workspace client.

    Returns:
        WorkspaceClient: Authenticated client for Databricks SDK operations
    """
    return WorkspaceClient()


def get_lakebase_connection_string(
    project: str = LAKEBASE_PROJECT,
    branch: str = LAKEBASE_BRANCH,
    region: str = REGION
) -> str:
    """
    Build Lakebase PostgreSQL connection string.

    Args:
        project: Lakebase project name
        branch: Branch name within the project
        region: Cloud region

    Returns:
        PostgreSQL connection string for Lakebase
    """
    token = get_workspace_client().dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    return f"postgresql://token:{token}@{project}-{branch}.{region}.lakebase.databricks.com:5432/databricks_postgres"


def sanitize_namespace_id(user_id: str) -> str:
    """
    Sanitize user ID for use in DatabricksStore namespaces.

    DatabricksStore doesn't allow periods in namespace labels.
    This function replaces common special characters with safe alternatives.

    Args:
        user_id: User identifier that may contain special characters

    Returns:
        Sanitized user ID safe for use in namespaces

    Examples:
        >>> sanitize_namespace_id("alice@company.com")
        'alice_at_company_com'
        >>> sanitize_namespace_id("john.doe@example.org")
        'john_doe_at_example_org'
    """
    return user_id.replace(".", "_").replace("@", "_at_")


def get_mcp_endpoint_url(mcp_type: str, **kwargs) -> str:
    """
    Build MCP endpoint URL for different Databricks services.

    Args:
        mcp_type: Type of MCP service ("vector_search", "genie", "uc_functions")
        **kwargs: Additional parameters (catalog, schema, space_id, etc.)

    Returns:
        Full MCP endpoint URL

    Examples:
        >>> get_mcp_endpoint_url("vector_search", catalog=CATALOG, schema=SCHEMA)
        >>> get_mcp_endpoint_url("genie", space_id=GENIE_SPACE_ID)
        >>> get_mcp_endpoint_url("uc_functions", catalog=CATALOG, schema=SCHEMA)
    """
    base_url = HOST.rstrip("/")

    if mcp_type == "vector_search":
        catalog = kwargs.get("catalog", CATALOG)
        schema = kwargs.get("schema", SCHEMA)
        return f"{base_url}/api/2.0/mcp/vector-search/{catalog}/{schema}"

    elif mcp_type == "genie":
        space_id = kwargs.get("space_id", GENIE_SPACE_ID)
        return f"{base_url}/api/2.0/mcp/genie/{space_id}"

    elif mcp_type == "uc_functions":
        catalog = kwargs.get("catalog", CATALOG)
        schema = kwargs.get("schema", SCHEMA)
        return f"{base_url}/api/2.0/mcp/functions/{catalog}/{schema}"

    else:
        raise ValueError(f"Unknown MCP type: {mcp_type}")


def print_config():
    """Print current configuration for debugging."""
    print("=" * 80)
    print("DATABRICKS AGENT BOOTCAMP CONFIGURATION")
    print("=" * 80)
    print(f"Unity Catalog: {CATALOG}.{SCHEMA}")
    print(f"Workspace: {HOST}")
    print(f"Region: {REGION}")
    print(f"LLM Endpoint: {LLM_ENDPOINT}")
    print(f"Embedding Endpoint: {EMBEDDING_ENDPOINT}")
    print(f"Lakebase Project: {LAKEBASE_PROJECT}")
    print(f"Lakebase Branch: {LAKEBASE_BRANCH}")
    print(f"Genie Space ID: {GENIE_SPACE_ID}")
    print(f"Vector Index: {VECTOR_INDEX}")
    print(f"Model Name: {MODEL_NAME}")
    print("=" * 80)


if __name__ == "__main__":
    # Print configuration when run directly
    print_config()
