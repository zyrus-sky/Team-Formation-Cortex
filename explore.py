from pyspark.sql import SparkSession
from pyspark.sql.functions import col
# This file was responsible for exploring the data using Spark SQL. it was made for testing/learning purposes and is no longer used in the app
# However, it can be useful for future data exploration tasks.
def main():
    """
    Initializes Spark, loads data, and runs several analytical queries using Spark SQL.
    """
    # 1. Initialize a Local Spark Session
    spark = SparkSession.builder \
        .appName("TeamFormationCortex-Exploration") \
        .master("local[*]") \
        .getOrCreate()

    print("Spark session initialized successfully for exploration.")
    # Set log level to ERROR to reduce console noise from INFO and WARN messages
    spark.sparkContext.setLogLevel("ERROR")

    # Load the datasets
    employees_df = spark.read.csv("employees.csv", header=True, inferSchema=True)
    projects_df = spark.read.csv("projects_history.csv", header=True, inferSchema=True)

    # Create Temporary Views to enable Spark SQL
    employees_df.createOrReplaceTempView("employees")
    projects_df.createOrReplaceTempView("projects")
    print("Temporary SQL views 'employees' and 'projects' created.")

    # --- Running Analytical Queries ---

    # Query 1: Create a comprehensive, joined view
    print("\n--- Query 1: Comprehensive view of employees and their projects ---")
    comprehensive_df = spark.sql("""
        SELECT 
            e.EmployeeID,
            e.Name,
            e.Role AS Current_Role,
            p.Project_Name,
            p.Project_Role,
            p.Domain,
            p.Individual_Performance_Score,
            p.Tech_Stack_Used
        FROM 
            employees e
        JOIN 
            projects p ON e.EmployeeID = p.EmployeeID
    """)
    comprehensive_df.show(10, truncate=False)

    # Query 2: Find the Top 10 Performers Overall
    # This demonstrates aggregation (AVG), grouping (GROUP BY), and ordering (ORDER BY).
    print("\n--- Query 2: Top 10 Employees by Average Performance ---")
    top_performers_df = spark.sql("""
        SELECT 
            e.Name,
            e.Role,
            COUNT(p.ProjectID) AS Projects_Completed,
            ROUND(AVG(p.Individual_Performance_Score), 2) AS Avg_Performance
        FROM 
            employees e
        JOIN 
            projects p ON e.EmployeeID = p.EmployeeID
        GROUP BY
            e.Name, e.Role
        ORDER BY 
            Avg_Performance DESC
        LIMIT 10
    """)
    top_performers_df.show()
    
    # Query 3: Find Most Experienced 'Python' Developers for 'Fintech' Projects
    # This is a realistic query a manager would ask. It uses filtering (WHERE) and array functions (array_contains).
    print("\n--- Query 3: Most Experienced 'Python' Developers for 'Fintech' ---")
    fintech_python_experts_df = spark.sql("""
        SELECT
            e.Name,
            e.Role,
            COUNT(p.ProjectID) AS Fintech_Python_Projects
        FROM
            employees e
        JOIN
            projects p ON e.EmployeeID = p.EmployeeID
        WHERE
            p.Domain = 'Fintech' AND array_contains(split(p.Tech_Stack_Used, ','), 'Python')
        GROUP BY
            e.Name, e.Role
        ORDER BY
            Fintech_Python_Projects DESC
        LIMIT 10
    """)
    fintech_python_experts_df.show()

    # Query 4: Analyze Average Cost per Role
    # A valuable query for financial planning and budgeting.
    print("\n--- Query 4: Average Cost per Role ---")
    avg_cost_per_role_df = spark.sql("""
        SELECT
            Role,
            COUNT(EmployeeID) AS Number_of_Employees,
            CAST(AVG(Cost_per_Hour) AS DECIMAL(10, 2)) AS Average_Hourly_Cost
        FROM
            employees
        GROUP BY
            Role
        ORDER BY
            Average_Hourly_Cost DESC
    """)
    avg_cost_per_role_df.show()

    # Stop the Spark session
    spark.stop()
    print("\nSpark session stopped.")

if __name__ == "__main__":
    main()
