import csv
import random
from faker import Faker
import numpy as np
import os
from pyspark.sql.functions import avg, col, coalesce

def generate_new_data():
    """Generates new mock CSV files for employees and project history."""
    
    # --- Configuration ---
    NUM_EMPLOYEES = 1200
    NUM_PROJECTS = 3000
    EMPLOYEES_CSV = "employees.csv"
    PROJECTS_CSV = "projects_history.csv"

    # --- Data Pools ---
    fake = Faker()
    ROLES = {
        "Lead Architect": (110, 150), "Senior Developer": (85, 120), "Project Manager": (80, 110),
        "Data Scientist": (90, 130), "DevOps Engineer": (80, 115), "Junior Developer": (55, 75),
        "UI/UX Designer": (65, 95), "QA Engineer": (60, 85), "Tech Lead": (100, 140)
    }
    AVAILABILITY = ["Available", "On Project", "Available"] 
    DOMAINS = ["Fintech", "Healthcare", "E-commerce", "SaaS", "Logistics", "AI/ML", "Retail", "Cybersecurity"]
    
    PROJECT_ROLES = {
        "Lead Architect": ["Solution Architect", "System Designer"],
        "Senior Developer": ["Backend Lead", "Frontend Lead", "Module Owner", "Code Reviewer"],
        "Project Manager": ["Scrum Master", "Product Owner", "Program Manager"],
        "Data Scientist": ["ML Engineer", "Data Analyst", "Research Scientist"],
        "DevOps Engineer": ["CI/CD Specialist", "Cloud Engineer", "SRE"],
        "Junior Developer": ["Frontend Developer", "Backend Developer", "Bug Fixer", "Junior Coder"],
        "UI/UX Designer": ["Lead Designer", "UX Researcher", "Interaction Designer"],
        "QA Engineer": ["Automation Tester", "Manual Tester", "Performance Tester"],
        "Tech Lead": ["Tech Lead", "Lead Developer"]
    }
    
    TECH_STACKS = { "Fintech": ["Python", "Java", "PostgreSQL", "Kafka", "AWS", "Kubernetes", "React"], "Healthcare": ["Python", "Java", "SQL", "Azure", "Docker", "FHIR", "TensorFlow"], "E-commerce": ["Java", "React", "MongoDB", "GCP", "Docker", "Vue.js", "SQL"], "SaaS": ["Python", "Angular", "PostgreSQL", "AWS", "Kubernetes", "Go"], "Logistics": ["Java", "Kafka", "PostgreSQL", "GCP", "Docker", "Python", "React"], "AI/ML": ["Python", "PyTorch", "TensorFlow", "AWS", "Spark", "Kubernetes", "SQL"], "Retail": ["Java", "SQL", "Azure", "React", "MongoDB", "Kafka"], "Cybersecurity": ["Python", "Go", "AWS", "Kubernetes", "Elasticsearch", "Bash"] }
    
    PROJECT_TEMPLATES = { 
        "Fintech": [ ("Payment Gateway Integration", "Developed a microservices-based payment gateway for processing credit card and ACH transactions, ensuring PCI compliance and high availability on AWS."), ("Real-time Fraud Detection Engine", "Built an AI/ML model using Python and TensorFlow to analyze transaction patterns in real-time, reducing fraudulent activities by 15%."), ("Portfolio Management Dashboard", "Created a React-based dashboard for financial advisors to manage client portfolios, with data streamed via Kafka from a PostgreSQL database.") ], 
        "Healthcare": [ ("Patient Data Analytics Platform", "Engineered a HIPAA-compliant data platform on Azure to process electronic health records (EHR) using Spark, enabling predictive diagnostics."), ("Telemedicine Video Conferencing App", "Launched a cross-platform mobile app for secure doctor-patient video consultations, integrating with the FHIR standard for data exchange."), ("AI-Powered Medical Imaging Analysis", "Designed a deep learning model with PyTorch to detect anomalies in MRI scans, achieving 95% accuracy in clinical trials.") ], 
        "E-commerce": [ ("Product Recommendation Engine", "Implemented a collaborative filtering recommendation system using Python and Spark, increasing average order value by 12%."), ("Scalable Shopping Cart API", "Built a highly available shopping cart and checkout service using Java and MongoDB, deployed on GCP with Kubernetes for auto-scaling during peak traffic."), ("Inventory Management System", "Developed a centralized system to track inventory across multiple warehouses, providing real-time updates to the storefront.") ], 
        "SaaS": [ ("Multi-tenant User Authentication Service", "Created a secure and scalable authentication and authorization service for a B2B SaaS platform, supporting OAuth 2.0 and SAML."), ("Subscription Billing Platform", "Engineered a flexible billing system to handle various subscription tiers, usage-based pricing, and invoicing for a growing SaaS product."), ("Customer Data Platform (CDP)", "Built a platform to aggregate and unify customer data from multiple sources, providing a 360-degree view for marketing and support teams.") ], 
        "Cybersecurity": [ ("SIEM Integration Pipeline", "Developed a data pipeline to ingest security logs from various sources into an Elasticsearch cluster for real-time threat analysis."), ("Automated Penetration Testing Framework", "Built a framework using Python and Go to automate routine security vulnerability scans across the company's web assets.") ] 
    }
    
    # --- Employee Generation ---
    employees = []
    for i in range(1, NUM_EMPLOYEES + 1):
        role = random.choice(list(ROLES.keys()))
        cost_range = ROLES[role]
        employee_domain = random.choice(DOMAINS)
        skills_for_domain = TECH_STACKS[employee_domain]
        assigned_skills = random.sample(skills_for_domain, k=random.randint(2, 4))
        employees.append({ 
            "EmployeeID": 1000 + i, "Name": fake.name(), "Age": random.randint(22, 60), 
            "Role": role, "Cost_per_Hour": random.randint(cost_range[0], cost_range[1]), 
            "Availability": random.choice(AVAILABILITY), "Domain_Expertise": employee_domain, 
            "Skills": ",".join(assigned_skills) 
        })

    # --- Project History Generation ---
    projects = []
    for i in range(1, NUM_PROJECTS + 1):
        domain = random.choice(list(PROJECT_TEMPLATES.keys()))
        template = random.choice(PROJECT_TEMPLATES[domain])
        team_size = random.randint(2, 6)
        team_members_info = random.sample(employees, k=team_size)
        project_tech = list(set(random.sample(TECH_STACKS[domain], k=random.randint(3, 5))))
        overall_project_success = round(random.uniform(0.65, 0.99), 2)
        for member_info in team_members_info:
            main_role = member_info["Role"]
            possible_project_roles = PROJECT_ROLES.get(main_role, [main_role])
            project_role = random.choice(possible_project_roles)
            performance_noise = np.random.normal(0, 0.05)
            individual_performance = min(1.0, max(0.0, overall_project_success + performance_noise))
            projects.append({ 
                "ProjectID": 5000 + i, "EmployeeID": member_info["EmployeeID"], 
                "Project_Name": template[0], 
                "Project_Description": template[1] + f" Tech stack: {', '.join(project_tech)}.", 
                "Tech_Stack_Used": ",".join(project_tech), "Domain": domain, 
                "Project_Role": project_role, "Project_Success_Score": overall_project_success, 
                "Individual_Performance_Score": round(individual_performance, 2) 
            })

    # --- Write to CSV ---
    with open(EMPLOYEES_CSV, "w", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=employees[0].keys())
        writer.writeheader()
        writer.writerows(employees)
    with open(PROJECTS_CSV, "w", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=projects[0].keys())
        writer.writeheader()
        writer.writerows(projects)
    
    print(f"Generated {len(employees)} employees and {len(projects)} project history entries.")


# --- NEW: Pre-processing Function ---
def pre_process_and_enrich_data(spark):
    """
    Performs a one-time batch job to calculate average employee performance
    and enriches the main employees Parquet file with this data.
    """
    EMPLOYEES_PARQUET = "employees.parquet"
    PROJECTS_PARQUET = "projects.parquet"

    print("Starting data pre-processing and enrichment job...")
    
    try:
        employees_df = spark.read.parquet(EMPLOYEES_PARQUET)
        projects_df = spark.read.parquet(PROJECTS_PARQUET)

        # Calculate average performance for each employee
        print("Calculating average performance scores...")
        avg_performance = projects_df.groupBy("EmployeeID") \
            .agg(avg("Individual_Performance_Score").alias("Avg_Performance"))

        # Join the average performance back to the main employees DataFrame
        print("Joining performance scores with employee data...")
        # Use a left join to keep all employees, even if they have no project history
        enriched_employees_df = employees_df.join(avg_performance, "EmployeeID", "left") \
            .fillna(0, subset=['Avg_Performance']) # Fill nulls for employees with no projects

        # Overwrite the existing employees Parquet file with the new, enriched data
        print(f"Overwriting '{EMPLOYEES_PARQUET}' with enriched data...")
        enriched_employees_df.write.mode("overwrite").parquet(EMPLOYEES_PARQUET)
        
        print("Data enrichment job completed successfully.")
        return True, "Data successfully pre-processed and enriched."
    except Exception as e:
        print(f"An error occurred during data pre-processing: {e}")
        return False, f"Error: {e}"


if __name__ == "__main__":
    # This allows the script to be run standalone from the terminal
    # to generate the initial CSV files.
    print("Running data generator as a standalone script...")
    generate_new_data()

