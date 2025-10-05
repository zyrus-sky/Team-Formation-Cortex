import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, coalesce
import shutil
import sys
import time

# --- Import from our modular files ---
from data_generator import generate_new_data, pre_process_and_enrich_data
from optimizer import find_optimal_team

# --- NEW: GPU library imports with error handling ---
try:
    import cudf
    from cuml.cluster import KMeans as cuKMeans
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    from cuml.decomposition import PCA as cuPCA
    GPU_LIBRARIES_AVAILABLE = True
except ImportError:
    GPU_LIBRARIES_AVAILABLE = False
    # Import CPU libraries as fallbacks
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA


# --- 1. VISUAL ENHANCEMENTS & ANIMATIONS ---
st.set_page_config(layout="wide", page_title="Team-Formation-Cortex", page_icon="🧠")

st.markdown("""
<style>
    /* Main title animation */
    .main-title {
        font-size: 3rem !important;
        font-weight: bold;
        background: -webkit-linear-gradient(45deg, #090088, #00ff95 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 1s ease-in-out;
    }
    h1 { animation: fadeIn 0.8s ease-in-out; }
    .stButton>button {
        border-radius: 20px; border: 1px solid #00ff95;
        background-image: linear-gradient(45deg, #090088, #00ff95);
        color: white; transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0, 255, 149, 0.3);
    }
    .stButton>button:active { transform: scale(0.95); }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .st-emotion-cache-1r4qj8v {
        border-radius: 0.5rem; padding: 1rem;
        background-color: #0E1117; animation: fadeIn 1.2s;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. BIG DATA SETUP ---
EMPLOYEES_PARQUET = "employees.parquet"
PROJECTS_PARQUET = "projects.parquet"
EMPLOYEES_CSV = "employees.csv"
PROJECTS_CSV = "projects_history.csv"

@st.cache_resource
def initialize_spark_session(gpu_enabled=False):
    """Initializes and returns a Spark session with optional GPU configuration."""
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    hadoop_home_path = "C:\\hadoop"
    if 'HADOOP_HOME' not in os.environ:
        os.environ['HADOOP_HOME'] = hadoop_home_path

    builder = SparkSession.builder \
        .appName("TeamFormationCortex-WebApp") \
        .master("local[*]") \
        .config("spark.driver.memory", "3g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")

    if gpu_enabled and GPU_LIBRARIES_AVAILABLE:
        st.info("🚀 Initializing Spark with GPU acceleration (NVIDIA RAPIDS)...")
        builder = builder.config("spark.plugins", "com.nvidia.spark.SQLPlugin") \
                         .config("spark.rapids.sql.enabled", "true")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def initialize_data_sources(spark):
    """Checks for Parquet files and creates them from CSVs if they don't exist."""
    if not os.path.exists(EMPLOYEES_PARQUET) and os.path.exists(EMPLOYEES_CSV):
        st.info(f"`{EMPLOYEES_PARQUET}` not found. Creating from `{EMPLOYEES_CSV}`...")
        try:
            df = spark.read.csv(EMPLOYEES_CSV, header=True, inferSchema=True)
            df.write.parquet(EMPLOYEES_PARQUET)
        except Exception as e:
            st.error(f"Could not create `{EMPLOYEES_PARQUET}`. Error: {e}")
            if os.path.exists(EMPLOYEES_PARQUET): shutil.rmtree(EMPLOYEES_PARQUET)
            st.stop()

    if not os.path.exists(PROJECTS_PARQUET) and os.path.exists(PROJECTS_CSV):
        st.info(f"`{PROJECTS_PARQUET}` not found. Creating from `{PROJECTS_CSV}`...")
        try:
            df = spark.read.csv(PROJECTS_CSV, header=True, inferSchema=True)
            df.write.parquet(PROJECTS_PARQUET)
        except Exception as e:
            st.error(f"Could not create `{PROJECTS_PARQUET}`. Error: {e}")
            if os.path.exists(PROJECTS_PARQUET): shutil.rmtree(PROJECTS_PARQUET)
            st.stop()

def get_spark_dataframe(spark, file_path):
    """Reads a Parquet file into a Spark DataFrame."""
    try:
        return spark.read.parquet(file_path)
    except Exception:
        return None

def rebuild_vector_database(spark, embedding_model):
    """Rebuilds the ChromaDB vector index from the projects Parquet file."""
    with st.spinner("🧠 Rebuilding Vector Database..."):
        project_spark_df = get_spark_dataframe(spark, PROJECTS_PARQUET)
        if project_spark_df is None:
            st.error(f"Cannot rebuild: `{PROJECTS_PARQUET}` not found.")
            return

        distinct_projects_df = project_spark_df.select("ProjectID", "Project_Description").distinct()
        projects_to_embed = distinct_projects_df.collect()

        db_client = chromadb.PersistentClient(path="./chroma_db")
        if "projects" in [c.name for c in db_client.list_collections()]:
            db_client.delete_collection(name="projects")
        collection = db_client.create_collection(name="projects")
        
        if projects_to_embed:
            project_ids = [str(row['ProjectID']) for row in projects_to_embed]
            documents = [row['Project_Description'] for row in projects_to_embed]
            embeddings = embedding_model.encode(documents, show_progress_bar=False, device='cuda' if use_gpu else 'cpu')
            collection.add(ids=project_ids, embeddings=embeddings.tolist(), documents=documents)
            st.success(f"✅ Vector Database rebuilt with {collection.count()} embeddings.")
        else:
            st.warning("⚠️ No project data found to build vector database.")

@st.cache_resource
def initialize_embedding_model():
    """Initializes the sentence transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- UI Sidebar Setup ---
st.sidebar.title("Cortex Navigation")
page = st.sidebar.radio("Go to", ["Team Builder", "Data Management", "Analytics Dashboard", "Setup & Utilities"])

with st.sidebar:
    st.header("⚡ Performance")
    use_gpu = st.toggle("Enable GPU Acceleration", value=False)
    if use_gpu and not GPU_LIBRARIES_AVAILABLE:
        st.error("RAPIDS (cuDF, cuML) not found. GPU acceleration is unavailable. Please install via Conda.")
        use_gpu = False

# --- Initialize on first run ---
spark = initialize_spark_session(use_gpu)
initialize_data_sources(spark)
embedding_model = initialize_embedding_model()

# --- 3. AI & DATA HELPER FUNCTIONS ---
@st.cache_data
def get_builder_data(_spark):
    """Loads, processes, and caches the necessary data for the Team Builder."""
    employee_spark_df = get_spark_dataframe(_spark, EMPLOYEES_PARQUET)
    project_spark_df = get_spark_dataframe(_spark, PROJECTS_PARQUET)
    
    if employee_spark_df is None or project_spark_df is None: return None, None
    
    if 'Avg_Performance' not in employee_spark_df.columns:
        st.warning("`Avg_Performance` not found. Please run 'Pre-process' on the Setup page.")
        employee_spark_df = employee_spark_df.withColumn('Avg_Performance', coalesce(col('Avg_Performance'), lit(0)))

    return employee_spark_df.toPandas(), project_spark_df.toPandas()

def parse_request_with_ai(user_query, llm_model):
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template("Extract 'required_skills' (list), 'max_budget_per_hour' (number), and 'project_description' (string) from this request: '{query}'. Format as JSON.")
    chain = prompt | llm_model | parser
    return chain.invoke({"query": user_query})

def generate_justification_with_ai(team_df, analysis, query, llm_model):
    prompt = ChatPromptTemplate.from_template("You are an HR strategist. Justify why this team is optimal for the request: '{query}'.\n\nTeam:\n{team_details}\n\nAnalysis:\n{analysis}\n\nBe concise and data-driven.")
    chain = prompt | llm_model
    response = chain.invoke({"query": query, "team_details": team_df.to_string(index=False), "analysis": str(analysis)})
    return response.content if hasattr(response, 'content') else response

# --- 4. STREAMLIT UI PAGES ---

if page == "Team Builder":
    st.markdown("<h1 class='main-title'>🧠 Team-Formation-Cortex</h1>", unsafe_allow_html=True)
    st.markdown("Your AI-powered assistant for building optimal, data-driven project teams.")

    with st.sidebar:
        st.header("🤖 AI Configuration")
        ai_provider = st.selectbox("Choose AI Provider", ["Google Gemini", "Ollama (Local)"])
        if ai_provider == "Google Gemini":
            google_api_key = st.text_input("Enter Google API Key", type="password")
            gemini_model = st.selectbox("Choose Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
        else:
            ollama_model = st.text_input("Enter Ollama Model Name", value="llama3:8b")

    with st.form("project_form"):
        user_query = st.text_area("**Describe your new project:**", height=150)
        submitted = st.form_submit_button("Assemble Optimal Team")

    if submitted and user_query:
        llm = None
        try:
            if ai_provider == "Google Gemini":
                if not google_api_key: st.error("Please enter your Google API Key.")
                else: llm = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=google_api_key, temperature=0)
            else: llm = ChatOllama(model=ollama_model, temperature=0)
        except Exception as e: st.error(f"Failed to initialize AI model: {e}")

        if llm:
            with st.spinner("Cortex is analyzing..."):
                employees_df, projects_df = get_builder_data(spark)
                if employees_df is None:
                    st.error("Employee data not found. Please initialize data on the Setup page.")
                    st.stop()
                
                db_client = chromadb.PersistentClient(path="./chroma_db")
                collection = db_client.get_or_create_collection(name="projects")
                
                parsed_request = parse_request_with_ai(user_query, llm)
                required_skills = parsed_request.get('required_skills', [])
                max_budget = parsed_request.get('max_budget_per_hour', 0)
                description = parsed_request.get('project_description', '')
                
                query_embedding = embedding_model.encode(description, device='cuda' if use_gpu else 'cpu').tolist()
                similar_projects = collection.query(query_embeddings=[query_embedding], n_results=50)
                
                project_fit_scores = {}
                if similar_projects['ids'] and projects_df is not None:
                    similar_project_ids = [int(pid) for pid in similar_projects['ids'][0]]
                    relevant_employees = projects_df[projects_df['ProjectID'].isin(similar_project_ids)]
                    fit_counts = relevant_employees['EmployeeID'].value_counts()
                    project_fit_scores = (fit_counts / fit_counts.max()).to_dict()

                recommended_ids, analysis = find_optimal_team(
                    employees_df=employees_df, project_fit_scores=project_fit_scores,
                    required_skills=required_skills, max_budget_per_hour=max_budget
                )

                if recommended_ids:
                    recommended_team_df = employees_df[employees_df['EmployeeID'].isin(recommended_ids)]
                    justification = generate_justification_with_ai(recommended_team_df, analysis, user_query, llm)
                    st.success("✅ Optimal Team Assembled!")
                    st.markdown("### 🤖 AI-Generated Justification")
                    st.write(justification)
                    st.markdown("### 👥 Team Details")
                    st.dataframe(recommended_team_df[['Name', 'Age', 'Role', 'Skills', 'Cost_per_Hour', 'Avg_Performance']])
                else:
                    st.error(f"**Analysis Failed:** {analysis.get('status', 'Could not find a team.')}")

elif page == "Data Management":
    st.title("🗃️ Data Management Hub")
    emp_tab, proj_tab = st.tabs(["Manage Employees", "Manage Projects"])

    with emp_tab:
        st.header("👥 Employee Data")
        st.info(f"Data Source: `{EMPLOYEES_PARQUET}`")
        
        uploaded_emp_file = st.file_uploader("Upload `employees.csv` to Add or Update", type="csv", key="emp_uploader")
        if uploaded_emp_file:
            updates_pd_df = pd.read_csv(uploaded_emp_file)
            updates_df = spark.createDataFrame(updates_pd_df)

            if st.button("Process Employee Updates"):
                with st.spinner("Processing upsert operation with Spark..."):
                    existing_df = get_spark_dataframe(spark, EMPLOYEES_PARQUET)
                    
                    if existing_df:
                        merged_df = existing_df.join(updates_df, "EmployeeID", "full_outer").select(
                            col("EmployeeID"),
                            coalesce(updates_df.Name, existing_df.Name).alias("Name"),
                            coalesce(updates_df.Age, existing_df.Age).alias("Age"),
                            coalesce(updates_df.Role, existing_df.Role).alias("Role"),
                            coalesce(updates_df.Cost_per_Hour, existing_df.Cost_per_Hour).alias("Cost_per_Hour"),
                            coalesce(updates_df.Availability, existing_df.Availability).alias("Availability"),
                            coalesce(updates_df.Domain_Expertise, existing_df.Domain_Expertise).alias("Domain_Expertise"),
                            coalesce(updates_df.Skills, existing_df.Skills).alias("Skills")
                        )
                    else:
                        merged_df = updates_df

                    merged_df.write.mode("overwrite").parquet(EMPLOYEES_PARQUET)
                    st.success("✅ Employee data successfully updated!")
                    st.cache_data.clear()
                    st.rerun()

        st.subheader("✍️ Live Employee Editor")
        employee_df_spark = get_spark_dataframe(spark, EMPLOYEES_PARQUET)
        if employee_df_spark:
            edited_pd_df = st.data_editor(employee_df_spark.toPandas(), num_rows="dynamic", key="emp_editor")
            if st.button("Save Employee Changes"):
                 with st.spinner("Saving changes with Spark..."):
                    updated_spark_df = spark.createDataFrame(edited_pd_df)
                    updated_spark_df.write.mode("overwrite").parquet(EMPLOYEES_PARQUET)
                    st.success("✅ Employee changes saved!")
                    st.cache_data.clear()
                    st.rerun()

    with proj_tab:
        st.header("📋 Project History")
        st.info(f"Data Source: `{PROJECTS_PARQUET}`")
        
        uploaded_proj_file = st.file_uploader("Upload `projects_history.csv` to Add or Update", type="csv", key="proj_uploader")
        if uploaded_proj_file:
            updates_pd_df = pd.read_csv(uploaded_proj_file)
            updates_df = spark.createDataFrame(updates_pd_df)
            if st.button("Process Project Updates"):
                 with st.spinner("Processing upsert and rebuilding vector DB..."):
                    existing_df = get_spark_dataframe(spark, PROJECTS_PARQUET)
                    if existing_df:
                        updates_df.write.mode("overwrite").parquet(PROJECTS_PARQUET)
                    else:
                        updates_df.write.mode("overwrite").parquet(PROJECTS_PARQUET)
                    
                    st.success("✅ Project data successfully updated!")
                    rebuild_vector_database(spark, embedding_model)
                    st.cache_data.clear()
                    st.rerun()

        st.subheader("✍️ Live Project History Editor")
        project_df_spark = get_spark_dataframe(spark, PROJECTS_PARQUET)
        if project_df_spark:
            edited_proj_pd_df = st.data_editor(project_df_spark.toPandas(), num_rows="dynamic", key="proj_editor")
            if st.button("Save Project Changes"):
                with st.spinner("Saving changes and rebuilding vector DB..."):
                    updated_proj_spark_df = spark.createDataFrame(edited_proj_pd_df)
                    updated_proj_spark_df.write.mode("overwrite").parquet(PROJECTS_PARQUET)
                    st.success("✅ Project changes saved!")
                    rebuild_vector_database(spark, embedding_model)
                    st.cache_data.clear()
                    st.rerun()

elif page == "Analytics Dashboard":
    st.title("📊 Analytics Dashboard")
    employee_df_spark = get_spark_dataframe(spark, EMPLOYEES_PARQUET)

    if employee_df_spark is None:
        st.warning("Data not found. Please add data via the Setup & Utilities page.")
    else:
        if 'Avg_Performance' not in employee_df_spark.columns:
            st.warning("`Avg_Performance` not found. Run 'Pre-process' for full analytics.")
            df_pd = employee_df_spark.toPandas()
            df_pd['Avg_Performance'] = 0
        else:
            df_pd = employee_df_spark.toPandas()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution of Employee Roles")
            role_counts = df_pd['Role'].value_counts()
            fig_roles = px.pie(role_counts, values=role_counts.values, names=role_counts.index, title="Role Breakdown")
            st.plotly_chart(fig_roles, use_container_width=True)

        with col2:
            st.subheader("Cost Per Hour Distribution")
            fig_cost = px.histogram(df_pd, x="Cost_per_Hour", nbins=20, title="Frequency of Hourly Costs")
            st.plotly_chart(fig_cost, use_container_width=True)

        st.subheader("Employee Profile Clustering (K-Means)")
        if len(df_pd) >= 4:
            features_pd = df_pd[['Age', 'Cost_per_Hour', 'Avg_Performance']].fillna(0)
            
            if use_gpu and GPU_LIBRARIES_AVAILABLE:
                st.info("⚡ Running clustering on GPU with RAPIDS cuML...")
                features_gdf = cudf.from_pandas(features_pd)
                scaler = cuStandardScaler()
                scaled_features = scaler.fit_transform(features_gdf)
                kmeans = cuKMeans(n_clusters=4, random_state=42).fit(scaled_features)
                df_pd['Cluster'] = kmeans.labels_.to_numpy().astype(str)
                pca = cuPCA(n_components=2)
                reduced_features = pca.fit_transform(scaled_features).to_pandas().values
            else:
                st.info("Running clustering on CPU with scikit-learn...")
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features_pd)
                kmeans = KMeans(n_clusters=4, random_state=42, n_init=10).fit(scaled_features)
                df_pd['Cluster'] = kmeans.labels_.astype(str)
                pca = PCA(n_components=2)
                reduced_features = pca.fit_transform(scaled_features)

            df_pd['pca1'] = reduced_features[:, 0]
            df_pd['pca2'] = reduced_features[:, 1]
            
            fig_cluster = px.scatter(
                df_pd, x='pca1', y='pca2', color='Cluster',
                hover_name='Name', hover_data=['Role', 'Cost_per_Hour', 'Avg_Performance'],
                title="Employee Clusters"
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.warning("Not enough data to generate a cluster plot.")

        st.subheader("Ad-Hoc Analysis with Spark SQL")
        query = st.text_area("Enter your Spark SQL query:", value="SELECT Role, count(*) as count FROM employees GROUP BY Role ORDER BY count DESC", height=150)
        if st.button("🚀 Run Query"):
            try:
                # Use the latest data for the query
                employee_df_spark.createOrReplaceTempView("employees")
                project_df_spark = get_spark_dataframe(spark, PROJECTS_PARQUET)
                if project_df_spark: project_df_spark.createOrReplaceTempView("projects")
                
                with st.spinner("Executing Spark SQL query..."):
                    result_df = spark.sql(query).toPandas()
                    st.dataframe(result_df)
            except Exception as e:
                st.error(f"Invalid Query or Data Error: {e}")

elif page == "Setup & Utilities":
    st.title("⚙️ Setup & Utilities")

    st.header("🗳️ Mock Data Generation")
    st.warning(f"This will overwrite `{EMPLOYEES_CSV}` and `{PROJECTS_CSV}`.")
    if st.button("Generate New Mock Data"):
        generate_new_data()
        st.info("Data generated. You must now re-initialize the Parquet files.")
    
    st.header("⏩ Data Initialization")
    st.info("Reads CSVs and creates/overwrites the Parquet files.")
    if st.button("Initialize/Re-Initialize Parquet Files from CSVs"):
        if os.path.exists(EMPLOYEES_PARQUET): shutil.rmtree(EMPLOYEES_PARQUET)
        if os.path.exists(PROJECTS_PARQUET): shutil.rmtree(PROJECTS_PARQUET)
        initialize_data_sources(spark)
        st.success("Parquet files created.")
        st.cache_data.clear()
        st.cache_resource.clear()

    st.header("⚡ Data Pre-processing")
    st.info("Calculates performance metrics and enriches employee data.")
    if st.button("Pre-process and Enrich Data"):
        success, message = pre_process_and_enrich_data(spark)
        if success:
            st.success(message)
            st.cache_data.clear()
        else:
            st.error(message)

    st.header("🧠 Vector Database")
    st.info("Powers the AI's ability to find relevant project experience.")
    if st.button("Manually Rebuild Vector Database"):
        rebuild_vector_database(spark, embedding_model)

