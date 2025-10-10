# data_generator.py
import os
import random
import pandas as pd
import numpy as np
from faker import Faker
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from delta.tables import DeltaTable

# File and table locations (folder paths used as Delta table locations)
EMPLOYEES_CSV = "employees.csv"
PROJECTS_CSV = "projects_history.csv"
EMPLOYEES_DELTA = "employees.delta"
PROJECTS_DELTA = "projects.delta"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ROLES = [
    "Data Scientist", "ML Engineer", "Data Engineer", "Backend Engineer",
    "Frontend Engineer", "DevOps Engineer", "Product Manager", "QA Engineer"
]

DOMAINS = ["Finance", "Healthcare", "Retail", "EdTech", "Media", "GovTech"]

SKILLS = [
    "Python", "SQL", "Spark", "Scala", "Java", "Go", "React", "Vue",
    "Docker", "Kubernetes", "TensorFlow", "PyTorch", "NLP", "LLMs",
    "Delta Lake", "Airflow", "DBT", "AWS", "GCP", "Azure"
]

def _random_skills():
    k = random.randint(3, 7)
    return ", ".join(random.sample(SKILLS, k))

def generate_new_data(num_employees: int = 400, num_projects: int = 900):
    """
    Create fresh mock CSVs for employees and project histories with consistent columns.
    """
    fake = Faker()
    employees = []
    for i in range(1000, 1000 + num_employees):
        role = random.choice(ROLES)
        domain = random.choice(DOMAINS)
        employees.append({
            "EmployeeID": i,
            "Name": fake.name(),
            "Age": random.randint(22, 58),
            "Role": role,
            "Domain_Expertise": domain,
            "Availability": random.choice(["Available", "Busy"]),
            "Skills": _random_skills(),
            "Cost_per_Hour": round(random.uniform(20, 140), 2),
        })

    projects = []
    pid = 5000
    for _ in range(num_projects):
        pid += 1
        domain = random.choice(DOMAINS)
        techs = ", ".join(random.sample(SKILLS, random.randint(3, 6)))
        desc = f"{domain} project using {techs} to deliver analytics and applications."
        team_size = random.randint(3, 9)
        member_ids = random.sample(range(1000, 1000 + num_employees), team_size)
        success = round(random.uniform(0.55, 0.98), 2)
        for eid in member_ids:
            indiv = float(min(1.0, max(0.0, np.random.normal(loc=success, scale=0.08))))
            projects.append({
                "ProjectID": pid,
                "EmployeeID": eid,
                "Project_Name": f"Project-{pid}",
                "Project_Description": desc,
                "Tech_Stack_Used": techs,
                "Domain": domain,
                "Project_Role": random.choice(["Lead", "Developer", "Analyst", "Engineer", "QA"]),
                "Project_Success_Score": success,
                "Individual_Performance_Score": round(indiv, 2),
            })

    pd.DataFrame(employees).to_csv(EMPLOYEES_CSV, index=False)
    pd.DataFrame(projects).to_csv(PROJECTS_CSV, index=False)

def _ensure_delta_tables(spark: SparkSession, employees_partition_col: str = "Domain_Expertise"):
    """
    Create Delta tables from CSVs if the folders don't exist yet.
    """
    if (not os.path.exists(EMPLOYEES_DELTA)) and os.path.exists(EMPLOYEES_CSV):
        df = spark.read.csv(EMPLOYEES_CSV, header=True, inferSchema=True)
        df.write.format("delta").partitionBy(employees_partition_col).mode("overwrite").save(EMPLOYEES_DELTA)
    if (not os.path.exists(PROJECTS_DELTA)) and os.path.exists(PROJECTS_CSV):
        df = spark.read.csv(PROJECTS_CSV, header=True, inferSchema=True)
        df.write.format("delta").partitionBy("Domain").mode("overwrite").save(PROJECTS_DELTA)

def pre_process_and_enrich_data(spark: SparkSession) -> tuple[bool, str]:
    """
    Option A: Evolve schema via DataFrame write with overwriteSchema to add Avg_Performance,
    then MERGE computed averages into employees.delta (ACID, auditable in history/time-travel).
    Steps:
      1) Compute Avg_Performance from projects.delta (mean of Individual_Performance_Score by EmployeeID).
      2) If employees.delta lacks Avg_Performance, add it with an overwriteSchema write.
      3) MERGE averages into employees.delta on EmployeeID to set values transactionally.
    """
    try:
        _ensure_delta_tables(spark)
        if not (os.path.exists(EMPLOYEES_DELTA) and os.path.exists(PROJECTS_DELTA)):
            return False, "Delta tables not initialized. Initialize from CSV first."

        # 1) Compute per-employee averages from projects.delta
        proj = spark.read.format("delta").load(PROJECTS_DELTA)
        if "Individual_Performance_Score" not in proj.columns:
            return False, "projects.delta missing Individual_Performance_Score."
        avg_df = (
            proj.groupBy("EmployeeID")
                .agg(F.avg("Individual_Performance_Score").alias("Avg_Performance"))
                .withColumn("Avg_Performance", F.col("Avg_Performance").cast(DoubleType()))
        )

        # 2) Ensure employees.delta has Avg_Performance via overwriteSchema
        emp = spark.read.format("delta").load(EMPLOYEES_DELTA)
        if "Avg_Performance" not in emp.columns:
            emp_add = emp.withColumn("Avg_Performance", F.lit(0.0).cast(DoubleType()))
            # overwrite data+schema in a single commit so schema is time-travel visible
            emp_add.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(EMPLOYEES_DELTA)

        # 3) MERGE averages (UPDATE when matched) into employees.delta
        emp_dt = DeltaTable.forPath(spark, EMPLOYEES_DELTA)
        (
            emp_dt.alias("emp")
            .merge(avg_df.alias("avg"), "emp.EmployeeID = avg.EmployeeID")
            .whenMatchedUpdate(set={"Avg_Performance": F.col("avg.Avg_Performance")})
            .execute()
        )

        return True, "Avg_Performance added and merged into employees.delta."
    except Exception as e:
        return False, f"Enrichment failed: {e}"
def compact_and_sort_delta(
    spark: SparkSession,
    table_path: str,
    partition_col: str | None,
    zorder_like_cols: list[str] | None,
    target_files_per_partition: int = 4,
) -> tuple[bool, str]:
    """
    OSS-friendly 'OPTIMIZE-like' routine:
      - Optionally sorts by provided columns to improve locality (emulates Z-Order behavior).
      - Coalesces/repartitions to reduce small files and overwrites the table.
      - Preserves partitioning by re-writing with partitionBy when partition_col is provided.
    Note: This is for education/demo; actual OPTIMIZE/ZORDER are platform features.
    """
    try:
        df = spark.read.format("delta").load(table_path)
        # Sort to emulate locality benefits of Z-Order for common filter columns
        if zorder_like_cols:
            sort_exprs = [F.col(c) for c in zorder_like_cols if c in df.columns]
            if sort_exprs:
                df = df.sort(*sort_exprs)

        if partition_col and partition_col in df.columns:
            # approximate per-partition file reduction via repartition(partition_col)
            df = df.repartition(target_files_per_partition, partition_col)
            df.write.format("delta").partitionBy(partition_col).mode("overwrite").option("overwriteSchema", "true").save(table_path)
        else:
            df = df.coalesce(target_files_per_partition)
            df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(table_path)

        return True, "Compaction and sort completed."
    except Exception as e:
        return False, f"Compaction failed: {e}"
