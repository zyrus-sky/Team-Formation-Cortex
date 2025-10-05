from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def main():
    """
    Main function to initialize Spark, ingest data, and perform basic exploration.
    """
    # --- Phase 1: Environment Setup & Data Ingestion ---
    
    # 1. Initialize a Local Spark Session
    # .master("local[*]") tells Spark to use all available cores on your local machine.
    # This simulates a distributed environment for development purposes.
    spark = SparkSession.builder \
        .appName("TeamFormationCortex-Ingestion") \
        .master("local[*]") \
        .getOrCreate()

    print("Spark session initialized successfully.")

    # File paths for the generated CSVs
    employees_csv_path = "employees.csv"
    projects_csv_path = "projects_history.csv"

    # 2. Load Data into Spark DataFrames
    # We use option("header", "true") to tell Spark that the first row is the column names.
    # inferSchema=True automatically detects the data types of columns, which is convenient for development.
    print(f"Loading data from {employees_csv_path}...")
    employees_df = spark.read.csv(employees_csv_path, header=True, inferSchema=True)
    
    print(f"Loading data from {projects_csv_path}...")
    projects_df = spark.read.csv(projects_csv_path, header=True, inferSchema=True)

    # 3. Confirm Data Loading
    print("\n--- Employee Data Schema ---")
    employees_df.printSchema()
    print("\n--- Sample Employee Data ---")
    employees_df.show(5)

    print("\n--- Project History Data Schema ---")
    projects_df.printSchema()
    print("\n--- Sample Project History Data ---")
    projects_df.show(5, truncate=False) # truncate=False prevents cutting off long descriptions

    # --- Phase 2: Data Exploration & Feature Engineering ---

    # 1. Create Temporary Views for Spark SQL
    employees_df.createOrReplaceTempView("employees")
    projects_df.createOrReplaceTempView("projects")
    print("\nTemporary SQL views 'employees' and 'projects' created.")

    # 2. Perform an Example Exploratory Query
    print("\n--- Running Example Spark SQL Query: Top 5 Performing Employees in Fintech ---")
    
    top_fintech_performers_df = spark.sql("""
        SELECT 
            e.Name,
            e.Role,
            p.Project_Name,
            p.Individual_Performance_Score
        FROM 
            employees e 
        JOIN 
            projects p ON e.EmployeeID = p.EmployeeID
        WHERE 
            p.Domain = 'Fintech'
        ORDER BY 
            p.Individual_Performance_Score DESC
        LIMIT 5
    """)

    top_fintech_performers_df.show()

    # Stop the Spark session to release resources
    spark.stop()
    print("\nSpark session stopped.")

if __name__ == "__main__":
    main()
