# Databricks notebook source
# MAGIC %md
# MAGIC # Module 00: Bootcamp Setup
# MAGIC
# MAGIC ## What This Notebook Creates
# MAGIC
# MAGIC This notebook sets up the foundational infrastructure for the entire Agent Bootcamp:
# MAGIC
# MAGIC **Unity Catalog Assets:**
# MAGIC - ✅ Catalog: `agent_bootcamp`
# MAGIC - ✅ Schema: `knowledge_assistant`
# MAGIC - ✅ Volume: `/Volumes/agent_bootcamp/knowledge_assistant/source_docs`
# MAGIC
# MAGIC **Sample Data Tables:**
# MAGIC - ✅ `employee_data` - Sample employee records
# MAGIC - ✅ `leave_balances` - Vacation and sick leave balances
# MAGIC - ✅ `get_employee` UC function - Governed employee lookup tool
# MAGIC
# MAGIC **Source Documents:**
# MAGIC - ✅ Policy markdown files (vacation, remote work, benefits, etc.)
# MAGIC
# MAGIC **Lakebase:**
# MAGIC - ✅ Project: `knowledge-assistant-state`
# MAGIC - ✅ Branch: `development`
# MAGIC
# MAGIC ## How These Assets Show Up Later
# MAGIC - **Module 01 (RAG)** uses the seeded policy documents to build a Vector Search index
# MAGIC - **Module 04 (MCP tools)** uses the employee tables and `get_employee` UC function as governed tools
# MAGIC - **Module 02 (Memory)** uses the Lakebase project for short-term and long-term memory
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Databricks Runtime MLR 17.3 LTS or higher
# MAGIC - Unity Catalog enabled
# MAGIC - Permissions to create catalogs/schemas
# MAGIC - Lakebase access
# MAGIC
# MAGIC ## Estimated Time
# MAGIC 15 minutes

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Dependencies

# COMMAND ----------

%pip install -q --upgrade \
  "databricks-sdk>=0.101,<0.103" \
  "mlflow[databricks]>=3.10,<3.11" \
  "databricks-langchain>=0.17,<0.18"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Configuration
# MAGIC
# MAGIC Load configuration from `config.py` and display the setup parameters.

# COMMAND ----------

import sys
sys.path.append("/Workspace" + "/".join(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[:-2]))

from config import *

# Get current user
username = spark.sql("SELECT current_user()").collect()[0][0]

print("=" * 80)
print("BOOTCAMP SETUP CONFIGURATION")
print("=" * 80)
print(f"Catalog: {CATALOG}")
print(f"Schema: {SCHEMA}")
print(f"Full Namespace: {UC_NAMESPACE}")
print(f"Docs Volume: {DOCS_VOLUME}")
print(f"Lakebase Project: {LAKEBASE_PROJECT}")
print(f"Current User: {username}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Unity Catalog Assets
# MAGIC
# MAGIC Create the catalog, schema, and volume for storing source documents.

# COMMAND ----------

# Create catalog
try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
    print(f"✅ Catalog '{CATALOG}' ready")
except Exception as e:
    print(f"⚠️  Catalog creation: {e}")

# Create schema
try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {UC_NAMESPACE}")
    print(f"✅ Schema '{UC_NAMESPACE}' ready")
except Exception as e:
    print(f"⚠️  Schema creation: {e}")

# Create volume for source documents
try:
    spark.sql(f"""
        CREATE VOLUME IF NOT EXISTS {UC_NAMESPACE}.source_docs
    """)
    print(f"✅ Volume '{DOCS_VOLUME}' ready")
except Exception as e:
    print(f"⚠️  Volume creation: {e}")

print("\n✅ All Unity Catalog assets created successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create Sample Employee Data
# MAGIC
# MAGIC Create sample employee and leave balance tables for Genie demonstrations.

# COMMAND ----------

from pyspark.sql import Row
from datetime import date

# Sample employee data
employees = [
    (1, "Alice Johnson", "Engineering", "Senior Engineer", date(2019, 3, 15)),
    (2, "Bob Smith", "Engineering", "Staff Engineer", date(2018, 7, 22)),
    (3, "Carol Williams", "Engineering", "Engineer", date(2021, 1, 10)),
    (4, "David Brown", "Engineering", "Principal Engineer", date(2017, 5, 8)),
    (5, "Eve Davis", "Engineering", "Engineer", date(2022, 9, 12)),
    (6, "Frank Miller", "Sales", "Account Executive", date(2020, 2, 18)),
    (7, "Grace Wilson", "Sales", "Sales Manager", date(2018, 11, 5)),
    (8, "Henry Moore", "Sales", "Account Executive", date(2021, 6, 30)),
    (9, "Iris Taylor", "Sales", "Sales Director", date(2016, 4, 25)),
    (10, "Jack Anderson", "HR", "HR Manager", date(2019, 8, 14)),
    (11, "Karen Thomas", "HR", "HR Specialist", date(2020, 10, 3)),
    (12, "Leo Jackson", "Finance", "Financial Analyst", date(2021, 2, 27)),
    (13, "Mary White", "Finance", "Finance Manager", date(2018, 9, 19)),
    (14, "Nathan Harris", "Finance", "Accountant", date(2022, 1, 8)),
    (15, "Olivia Martin", "Marketing", "Marketing Manager", date(2019, 12, 11)),
    (16, "Paul Thompson", "Marketing", "Content Specialist", date(2021, 5, 22)),
    (17, "Quinn Garcia", "Marketing", "SEO Specialist", date(2020, 7, 16)),
    (18, "Rachel Martinez", "Engineering", "Engineering Manager", date(2017, 3, 4)),
    (19, "Sam Robinson", "Engineering", "DevOps Engineer", date(2020, 11, 29)),
    (20, "Tina Clark", "Sales", "Account Executive", date(2021, 8, 7)),
]

# Create employee DataFrame
employee_df = spark.createDataFrame(
    employees,
    ["employee_id", "name", "department", "role", "hire_date"]
)

# Save to Delta table
employee_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(EMPLOYEE_TABLE)

print(f"✅ Created {EMPLOYEE_TABLE} with {employee_df.count()} employees")
display(employee_df.limit(5))

# COMMAND ----------

# Leave balances data (corresponding to employees)
leave_balances = [
    (1, 20, 5, 10, 2),
    (2, 20, 8, 10, 3),
    (3, 15, 3, 10, 1),
    (4, 25, 12, 10, 4),
    (5, 15, 2, 10, 0),
    (6, 20, 10, 10, 5),
    (7, 25, 15, 10, 2),
    (8, 15, 4, 10, 1),
    (9, 25, 18, 10, 6),
    (10, 20, 7, 10, 3),
    (11, 15, 5, 10, 2),
    (12, 15, 3, 10, 1),
    (13, 20, 9, 10, 4),
    (14, 15, 1, 10, 0),
    (15, 20, 11, 10, 3),
    (16, 15, 6, 10, 2),
    (17, 15, 4, 10, 1),
    (18, 25, 13, 10, 5),
    (19, 20, 8, 10, 2),
    (20, 15, 7, 10, 3),
]

# Create leave balances DataFrame
leave_df = spark.createDataFrame(
    leave_balances,
    ["employee_id", "vacation_days_total", "vacation_days_used", "sick_days_total", "sick_days_used"]
)

# Save to Delta table
leave_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(LEAVE_BALANCES_TABLE)

print(f"✅ Created {LEAVE_BALANCES_TABLE} with {leave_df.count()} records")
display(leave_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Create Governed UC Function
# MAGIC
# MAGIC Create a Unity Catalog function that later modules can expose through MCP.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.get_employee(emp_id INT)
RETURNS STRING
LANGUAGE SQL
COMMENT 'Get employee info by ID (governed)'
RETURN (
  SELECT CONCAT(name, ' - ', role, ' (', department, ')')
  FROM {EMPLOYEE_TABLE}
  WHERE employee_id = emp_id
  LIMIT 1
)
""")

print(f"✅ Created UC function: {UC_NAMESPACE}.get_employee")
display(spark.sql(f"SELECT {UC_NAMESPACE}.get_employee(1) AS sample_employee"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Seed Source Documents
# MAGIC
# MAGIC Create sample policy documents that will be used for Vector Search demonstrations.

# COMMAND ----------

# Policy documents
policies = {
    "vacation_policy.md": """# Vacation and Time Off Policy

## Annual Vacation Allowance

All full-time employees receive paid vacation time based on tenure:

- **0-2 years**: 15 days per year
- **2-5 years**: 20 days per year
- **5+ years**: 25 days per year

## Requesting Time Off

1. Submit vacation requests at least 2 weeks in advance
2. Requests are subject to manager approval
3. Peak business periods may have blackout dates
4. Unused vacation days carry over (max 5 days)

## Sick Leave

All employees receive 10 sick days per year. Sick leave does not carry over.

## Holidays

The company observes 10 federal holidays per year, including:
- New Year's Day
- Memorial Day
- Independence Day
- Labor Day
- Thanksgiving (2 days)
- Christmas Day

## Parental Leave

- **Primary caregiver**: 16 weeks paid leave
- **Secondary caregiver**: 6 weeks paid leave
- Must be taken within 12 months of birth/adoption
""",

    "remote_work_policy.md": """# Remote Work Policy

## Eligibility

Remote work is available to employees who:
- Have been with the company for at least 6 months
- Have manager approval
- Can maintain productivity and communication standards
- Have a suitable home office setup

## Work Arrangements

### Fully Remote
- Work from home 5 days per week
- Must attend quarterly in-person meetings
- Equipment provided by company

### Hybrid
- Minimum 2 days per week in office
- Flexible scheduling with team coordination
- Office space reserved for hybrid workers

## Equipment and Expenses

The company provides:
- Laptop and monitor
- $500 annual home office stipend
- Internet reimbursement (up to $75/month)
- Ergonomic equipment upon request

## Expectations

- Core hours: 9 AM - 3 PM in your time zone
- Respond to messages within 4 hours during business hours
- Video on for team meetings
- Maintain secure home network

## Performance

Remote work privileges may be revoked if:
- Performance standards are not met
- Communication expectations are not maintained
- Team collaboration suffers
""",

    "professional_development.md": """# Professional Development Policy

## Learning Budget

Each employee receives an annual professional development budget:

- **Individual Contributors**: $2,000/year
- **Managers**: $3,000/year
- **Directors+**: $5,000/year

## Eligible Expenses

- Conference attendance
- Online courses (Coursera, Udemy, etc.)
- Professional certifications
- Technical books
- Workshop participation

## Conference Attendance

Employees may attend up to 2 conferences per year:
- Submit request 60 days in advance
- Company covers: registration, travel, accommodation
- Present learnings to team upon return

## Certification Support

The company supports industry certifications:
- Exam fees covered
- Study materials reimbursed
- Preparation time during work hours (up to 20 hours)
- Bonus for passing certification: $500

## Internal Learning

- Monthly tech talks (last Friday of month)
- Lunch and learn sessions
- Mentorship program
- Internal training library

## Time Off for Learning

Employees may use up to 5 days per year for dedicated learning:
- Does not count against vacation
- Requires manager approval
- Must relate to current role or career path
""",

    "benefits_overview.md": """# Employee Benefits Overview

## Health Insurance

### Medical Coverage
- Company pays 90% of premiums
- Multiple plan options (PPO, HMO, HDHP)
- Coverage begins on day 1
- Covers employee and dependents

### Dental and Vision
- Company pays 100% of premiums
- Annual checkups fully covered
- Orthodontics covered at 50%

## Retirement

### 401(k) Plan
- Company matches 100% up to 6% of salary
- Immediate vesting
- Wide range of investment options
- Financial advisor access

## Wellness Programs

- $50/month fitness reimbursement
- On-site gym (HQ)
- Mental health support (8 free counseling sessions/year)
- Annual health screening

## Life Insurance

- Company-paid life insurance (2x salary)
- Optional supplemental coverage available
- Accidental death & dismemberment

## Disability Insurance

- Short-term disability: 100% of salary for 90 days
- Long-term disability: 60% of salary after 90 days

## Additional Perks

- Free snacks and beverages
- Commuter benefits (pre-tax)
- Employee assistance program
- Pet insurance (optional)
- Legal services plan (optional)

## Leave Benefits

See separate policies for:
- Vacation and sick leave
- Parental leave
- Bereavement leave
- Jury duty leave
"""
}

# Write policies to volume
# Use dbutils.fs.put so this works consistently in jobs and interactive runs.
for filename, content in policies.items():
    file_path = f"{DOCS_VOLUME}/{filename}"
    dbutils.fs.put(file_path, content, overwrite=True)
    print(f"✅ Created {filename}")

# List files
files = dbutils.fs.ls(DOCS_VOLUME)
print(f"\n✅ Created {len(files)} policy documents in {DOCS_VOLUME}")
for file in files:
    print(f"  - {file.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Create Lakebase Project
# MAGIC
# MAGIC Create a Lakebase project for conversation memory (checkpointer) and long-term storage.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

try:
    # Create Lakebase project
    project = w.lakebase_projects.create(
        name=LAKEBASE_PROJECT,
        region=REGION
    )
    print(f"✅ Created Lakebase project: {LAKEBASE_PROJECT}")
    print(f"   Region: {REGION}")
    print(f"   Project ID: {project.project_id}")

    # The development branch is created automatically
    print(f"✅ Default branch 'main' created automatically")

except Exception as e:
    if "already exists" in str(e).lower():
        print(f"ℹ️  Lakebase project '{LAKEBASE_PROJECT}' already exists")
    else:
        print(f"⚠️  Lakebase project creation: {e}")
        print("\nNote: Lakebase may not be available in your workspace.")
        print("You can continue with the bootcamp - we'll handle this in Module 02.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Verification
# MAGIC
# MAGIC Verify all setup steps completed successfully.

# COMMAND ----------

print("=" * 80)
print("SETUP VERIFICATION")
print("=" * 80)

# Check catalog
try:
    spark.sql(f"USE CATALOG {CATALOG}")
    print(f"✅ Catalog '{CATALOG}' exists")
except Exception as e:
    print(f"❌ Catalog '{CATALOG}' NOT FOUND")

# Check schema
try:
    spark.sql(f"USE SCHEMA {UC_NAMESPACE}")
    print(f"✅ Schema '{UC_NAMESPACE}' exists")
except Exception as e:
    print(f"❌ Schema '{UC_NAMESPACE}' NOT FOUND")

# Check volume
try:
    files = dbutils.fs.ls(DOCS_VOLUME)
    print(f"✅ Volume '{DOCS_VOLUME}' exists ({len(files)} files)")
except Exception as e:
    print(f"❌ Volume NOT FOUND")

# Check employee table
try:
    count = spark.table(EMPLOYEE_TABLE).count()
    print(f"✅ Table '{EMPLOYEE_TABLE}' exists ({count} rows)")
except Exception as e:
    print(f"❌ Employee table NOT FOUND")

# Check leave balances table
try:
    count = spark.table(LEAVE_BALANCES_TABLE).count()
    print(f"✅ Table '{LEAVE_BALANCES_TABLE}' exists ({count} rows)")
except Exception as e:
    print(f"❌ Leave balances table NOT FOUND")

# Check Lakebase project
try:
    project = w.lakebase_projects.get(name=LAKEBASE_PROJECT)
    print(f"✅ Lakebase project '{LAKEBASE_PROJECT}' exists")
except Exception as e:
    print(f"⚠️  Lakebase project status unknown (this is OK for now)")

# Check UC function
try:
    sample_employee = spark.sql(
        f"SELECT {UC_NAMESPACE}.get_employee(1) AS employee_summary"
    ).collect()[0]["employee_summary"]
    print(f"✅ UC function '{UC_NAMESPACE}.get_employee' exists")
    print(f"   Sample output: {sample_employee}")
except Exception as e:
    print(f"❌ UC function '{UC_NAMESPACE}.get_employee' NOT FOUND")

print("=" * 80)
print("\n🎉 SETUP COMPLETE!")
print("\nNext Steps:")
print("1. Continue to Module 01 > 00_your_first_agent_on_databricks.py")
print("2. Then move to Module 01 > 01_building_a_doc_store_on_vector_search.py")
print("3. Return to the memory and MCP modules after RAG is working")
print("\nAll foundational assets are ready for the bootcamp! 🚀")
print("=" * 80)
