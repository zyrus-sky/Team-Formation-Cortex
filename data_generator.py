import csv
import random
from faker import Faker
import numpy as np

# --- Configuration ---
NUM_EMPLOYEES = 1200
NUM_PROJECTS = 3000

# --- Data Pools ---
fake = Faker()
ROLES = {
    "Lead Architect": (110, 150), "Senior Developer": (85, 120), "Project Manager": (80, 110),
    "Data Scientist": (90, 130), "DevOps Engineer": (80, 115), "Junior Developer": (55, 75),
    "UI/UX Designer": (65, 95), "QA Engineer": (60, 85), "Tech Lead": (100, 140)
}
AVAILABILITY = ["Available", "On Project", "Available"] # Skewed towards available
DOMAINS = ["Fintech", "Healthcare", "E-commerce", "SaaS", "Logistics", "AI/ML", "Retail", "Cybersecurity"]

# --- New: Specific Roles employees can take on a project ---
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

TECH_STACKS = {
    "Fintech": ["Python", "Java", "PostgreSQL", "Kafka", "AWS", "Kubernetes", "React"],
    "Healthcare": ["Python", "Java", "SQL", "Azure", "Docker", "FHIR", "TensorFlow"],
    "E-commerce": ["Java", "React", "MongoDB", "GCP", "Docker", "Vue.js", "SQL"],
    "SaaS": ["Python", "Angular", "PostgreSQL", "AWS", "Kubernetes", "Go"],
    "Logistics": ["Java", "Kafka", "PostgreSQL", "GCP", "Docker", "Python", "React"],
    "AI/ML": ["Python", "PyTorch", "TensorFlow", "AWS", "Spark", "Kubernetes", "SQL"],
    "Retail": ["Java", "SQL", "Azure", "React", "MongoDB", "Kafka"],
    "Cybersecurity": ["Python", "Go", "AWS", "Kubernetes", "Elasticsearch", "Bash"]
}

PROJECT_TEMPLATES = {
    "Fintech": [
        ("Payment Gateway Integration", "Developed a microservices-based payment gateway for processing credit card and ACH transactions, ensuring PCI compliance and high availability on AWS."),
        ("Real-time Fraud Detection Engine", "Built an AI/ML model using Python and TensorFlow to analyze transaction patterns in real-time, reducing fraudulent activities by 15%."),
        ("Portfolio Management Dashboard", "Created a React-based dashboard for financial advisors to manage client portfolios, with data streamed via Kafka from a PostgreSQL database.")
    ],
    "Healthcare": [
        ("Patient Data Analytics Platform", "Engineered a HIPAA-compliant data platform on Azure to process electronic health records (EHR) using Spark, enabling predictive diagnostics."),
        ("Telemedicine Video Conferencing App", "Launched a cross-platform mobile app for secure doctor-patient video consultations, integrating with the FHIR standard for data exchange."),
        ("AI-Powered Medical Imaging Analysis", "Designed a deep learning model with PyTorch to detect anomalies in MRI scans, achieving 95% accuracy in clinical trials.")
    ],
    "E-commerce": [
        ("Product Recommendation Engine", "Implemented a collaborative filtering recommendation system using Python and Spark, increasing average order value by 12%."),
        ("Scalable Shopping Cart API", "Built a highly available shopping cart and checkout service using Java and MongoDB, deployed on GCP with Kubernetes for auto-scaling during peak traffic."),
        ("Inventory Management System", "Developed a centralized system to track inventory across multiple warehouses, providing real-time updates to the storefront.")
    ],
    "SaaS": [
        ("Multi-tenant User Authentication Service", "Created a secure and scalable authentication and authorization service for a B2B SaaS platform, supporting OAuth 2.0 and SAML."),
        ("Subscription Billing Platform", "Engineered a flexible billing system to handle various subscription tiers, usage-based pricing, and invoicing for a growing SaaS product."),
        ("Customer Data Platform (CDP)", "Built a platform to aggregate and unify customer data from multiple sources, providing a 360-degree view for marketing and support teams.")
    ],
    "Cybersecurity": [
        ("SIEM Integration Pipeline", "Developed a data pipeline to ingest security logs from various sources into an Elasticsearch cluster for real-time threat analysis."),
        ("Automated Penetration Testing Framework", "Built a framework using Python and Go to automate routine security vulnerability scans across the company's web assets.")
    ]
}

# --- Employee Generation ---
employees = []
for i in range(1, NUM_EMPLOYEES + 1):
    role = random.choice(list(ROLES.keys()))
    cost_range = ROLES[role]
    employees.append({
        "EmployeeID": 1000 + i,
        "Name": fake.name(),
        "Role": role,
        "Cost_per_Hour": random.randint(cost_range[0], cost_range[1]),
        "Availability": random.choice(AVAILABILITY),
        "Domain_Expertise": random.choice(DOMAINS)
    })
employee_map = {emp['EmployeeID']: emp for emp in employees}

# --- Project History Generation ---
projects = []
for i in range(1, NUM_PROJECTS + 1):
    domain = random.choice(list(PROJECT_TEMPLATES.keys()))
    template = random.choice(PROJECT_TEMPLATES[domain])
    
    # Assign a team of 2 to 6 members
    team_size = random.randint(2, 6)
    team_members_info = random.sample(employees, k=team_size)
    
    project_tech = list(set(random.sample(TECH_STACKS[domain], k=random.randint(3, 5))))
    
    # New: Overall project success score is the same for the whole project
    overall_project_success = round(random.uniform(0.65, 0.99), 2)

    for member_info in team_members_info:
        # New: Assign a specific project role based on the employee's main role
        main_role = member_info["Role"]
        possible_project_roles = PROJECT_ROLES.get(main_role, [main_role]) # Fallback to main role if not defined
        project_role = random.choice(possible_project_roles)

        # New: Generate individual performance score based on overall project success
        performance_noise = np.random.normal(0, 0.05) # Add some random variance
        individual_performance = min(1.0, max(0.0, overall_project_success + performance_noise))

        projects.append({
            "ProjectID": 5000 + i,
            "EmployeeID": member_info["EmployeeID"],
            "Project_Name": template[0],
            "Project_Description": template[1] + f" The project utilized a tech stack including {', '.join(project_tech)}.",
            "Tech_Stack_Used": ",".join(project_tech),
            "Domain": domain,
            "Project_Role": project_role,
            "Project_Success_Score": overall_project_success,
            "Individual_Performance_Score": round(individual_performance, 2)
        })

# --- Write to CSV ---
with open("employees.csv", "w", newline="", encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=employees[0].keys())
    writer.writeheader()
    writer.writerows(employees)

with open("projects_history.csv", "w", newline="", encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=projects[0].keys())
    writer.writeheader()
    writer.writerows(projects)

print(f"Generated {len(employees)} employees and {len(projects)} project history entries.")

