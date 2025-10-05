import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from optimizer import find_optimal_team
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col

@st.cache_resource
def initialize_spark_session():
    """Initializes and returns a Spark session."""
    print("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("TeamFormationCortex-WebApp") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark session initialized.")
    return spark

@st.cache_data
def load_and_prepare_data(_spark):
    """Loads and prepares employee and project data using Spark."""
    print("Loading and preparing data...")
    employees_df_spark = _spark.read.csv("employees.csv", header=True, inferSchema=True)
    projects_df_spark = _spark.read.csv("projects_history.csv", header=True, inferSchema=True)


    avg_performance = projects_df_spark.groupBy("EmployeeID") \
        .agg(avg("Individual_Performance_Score").alias("Avg_Performance"))


    employees_df_spark = employees_df_spark.join(avg_performance, "EmployeeID", "left")
    
    employees_pd = employees_df_spark.toPandas()
    projects_pd = projects_df_spark.toPandas()
    print("Data preparation complete.")
    return employees_pd, projects_pd

@st.cache_resource
def initialize_shared_models_and_db():
    """Initializes models and DB that don't depend on user input."""
    print("Initializing shared models and DB...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    db_client = chromadb.PersistentClient(path="./chroma_db")
    collection = db_client.get_collection(name="projects")
    print("Shared models and DB initialized.")
    return embedding_model, collection


spark = initialize_spark_session()
employees_df, projects_df = load_and_prepare_data(spark)
embedding_model, collection = initialize_shared_models_and_db()

def parse_request_with_ai(user_query, llm_model):
    """Uses LangChain and an LLM to parse the user's request into structured JSON."""
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "A user needs to form a project team. Extract the key details from their request. "
        "The request is: '{query}'. \n"
        "Please provide the 'required_skills' as a list of strings, "
        "the 'max_budget_per_hour' as a number, "
        "and a concise 'project_description' as a string for similarity analysis. "
        "Format your response as a JSON object with these three keys."
    )
    chain = prompt | llm_model | parser
    return chain.invoke({"query": user_query})

def generate_justification_with_ai(team_df, analysis, query, llm_model):
    """Generates a human-readable justification for the recommended team."""
    prompt = ChatPromptTemplate.from_template(
        "You are an expert HR strategist. An optimization engine has recommended a team for a project based on this request: '{query}'.\n\n"
        "Recommended Team Details:\n{team_details}\n\n"
        "Analysis:\n{analysis}\n\n"
        "Please write a concise, professional justification for why this is the ideal team. "
        "Highlight how their combined skills and experience make them a perfect fit for the project described in the user's query. "
        "Be persuasive and data-driven in your summary."
    )
    
    chain = prompt | llm_model
    response = chain.invoke({
        "query": query,
        "team_details": team_df.to_string(index=False),
        "analysis": str(analysis)
    })
    return response.content if hasattr(response, 'content') else response

st.set_page_config(layout="wide")
st.title("🧠 Team-Formation-Cortex")
st.markdown("Your AI-powered assistant for building optimal, data-driven project teams.")

with st.sidebar:
    st.header("🤖 AI Configuration")
    ai_provider = st.selectbox("Choose AI Provider", ["Google Gemini", "Ollama (Local)"])

    if ai_provider == "Google Gemini":
        google_api_key = st.text_input("Enter Google API Key", type="password")
        gemini_model = st.selectbox("Choose Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    else: 
        ollama_model = st.text_input("Enter Ollama Model Name", value="llama3:8b")
        st.info("Ensure your Ollama server is running locally to use this option.")

with st.form("project_form"):
    user_query = st.text_area(
        "**Describe your new project:** (Include necessary skills, budget, and a brief description)",
        height=150,
        placeholder="e.g., We're building a new fintech payment gateway using Python and AWS. We need at least one senior developer. The total hourly budget for the team is $350."
    )
    submitted = st.form_submit_button("Assemble Optimal Team")

if submitted and user_query:
    
    
    llm = None
    try:
        if ai_provider == "Google Gemini":
            if not google_api_key:
                st.error("Please enter your Google API Key in the sidebar.")
            else:
                llm = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=google_api_key, temperature=0)
        elif ai_provider == "Ollama (Local)":
            llm = ChatOllama(model=ollama_model, temperature=0)
    except Exception as e:
        st.error(f"Failed to initialize the AI model: {e}")

    if llm:
        with st.spinner("Cortex is analyzing... Please wait."):
            
            st.info("Step 1: Parsing your request with AI...")
            parsed_request = parse_request_with_ai(user_query, llm)
            required_skills = parsed_request.get('required_skills', [])
            max_budget = parsed_request.get('max_budget_per_hour', 0)
            description = parsed_request.get('project_description', '')

            st.info("Step 2: Searching for relevant experience in the vector database...")
            query_embedding = embedding_model.encode(description).tolist()
            similar_projects = collection.query(query_embeddings=[query_embedding], n_results=50)
            
            project_fit_scores = {}
            if similar_projects['ids']:
                similar_project_ids = [int(pid) for pid in similar_projects['ids'][0]]
                relevant_employees = projects_df[projects_df['ProjectID'].isin(similar_project_ids)]

                fit_counts = relevant_employees['EmployeeID'].value_counts()
                project_fit_scores = (fit_counts / fit_counts.max()).to_dict()

            st.info("Step 3: Running the optimization engine to find the perfect team...")
            recommended_ids, analysis = find_optimal_team(
                employees_df=employees_df,
                project_fit_scores=project_fit_scores,
                required_skills=required_skills,
                max_budget_per_hour=max_budget
            )

            if recommended_ids:
                st.info("Step 4: Generating AI-powered justification...")
                recommended_team_df = employees_df[employees_df['EmployeeID'].isin(recommended_ids)]
                justification = generate_justification_with_ai(recommended_team_df, analysis, user_query, llm)
                
                st.success("Optimal Team Assembled!")
                st.markdown("### AI-Generated Justification")
                st.write(justification)
                st.markdown("### Team Details")
                st.dataframe(recommended_team_df[['Name', 'Role', 'Skills', 'Cost_per_Hour', 'Avg_Performance']])
            else:
                st.error(f"**Analysis Failed:** {analysis.get('status', 'Could not find a team.')}. Please try adjusting your budget or skill requirements.")

elif submitted:
    st.warning("Please enter a project description.")

