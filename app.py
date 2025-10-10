# app.py
import os
import sys
import shutil
import time
import random
import pandas as pd
import streamlit as st
import plotly.express as px
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, max as spark_max
from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable

from data_generator import (
    generate_new_data,
    pre_process_and_enrich_data,
    compact_and_sort_delta,  # keep if present in your data_generator.py
    EMPLOYEES_DELTA, PROJECTS_DELTA, EMPLOYEES_CSV, PROJECTS_CSV
)
from optimizer import find_optimal_team

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

st.set_page_config(layout="wide", page_title="Team-Formation-Cortex", page_icon="🧠")
st.markdown("<style>div[data-testid='stHelp']{display:none!important}</style>", unsafe_allow_html=True)
st.help = lambda *a, **k: None

# Minimal UX polish + smooth scroll
st.markdown("""
<style>
  .stButton>button { border-radius: 16px; }
  .block-container { padding-top: 1.2rem; }
  .main .block-container { scroll-behavior: smooth; }
  .stApp { scroll-behavior: smooth; }
  .main-title {
      font-size: 3rem !important; font-weight: bold;
      background: -webkit-linear-gradient(45deg, #090088, #00ff95 80%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      animation: fadeIn 1s ease-in-out;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Spark / Delta init
# -----------------------------------------------------------------------------
@st.cache_resource
def initialize_spark_session():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    if sys.platform == "win32":
        hadoop_home_path = r"C:\hadoop"
        os.environ.setdefault("HADOOP_HOME", hadoop_home_path)
        bin_path = os.path.join(hadoop_home_path, "bin")
        if os.path.isdir(bin_path):
            os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + bin_path
    builder = (
        SparkSession.builder
        .appName("TeamFormationCortex")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

@st.cache_resource
def initialize_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

spark = initialize_spark_session()
embedding_model = initialize_embedding_model()

# -----------------------------------------------------------------------------
# Caching helpers (Spark vs Pandas split)
# -----------------------------------------------------------------------------
# Spark-returning helper must be resource-cached (non-pickle)
@st.cache_resource
def get_delta_dataframe(path: str, version: int | None = None):
    reader = spark.read.format("delta")
    if version is not None:
        reader = reader.option("versionAsOf", str(version))
    return reader.load(path)

# Pandas-returning helper can be data-cached (pickle-safe)
@st.cache_data
def read_delta_as_pandas(path: str, version: int | None = None) -> pd.DataFrame:
    reader = spark.read.format("delta")
    if version is not None:
        reader = reader.option("versionAsOf", str(version))
    return reader.load(path).toPandas()

@st.cache_data
def get_builder_data():
    emp_pdf = read_delta_as_pandas(EMPLOYEES_DELTA)
    proj_pdf = read_delta_as_pandas(PROJECTS_DELTA)
    if emp_pdf is None or proj_pdf is None or emp_pdf.empty or proj_pdf.empty:
        return None, None
    if "Avg_Performance" not in emp_pdf.columns:
        st.warning("`Avg_Performance` not found. Please run 'Pre-process' in Setup.")
        emp_pdf["Avg_Performance"] = 0.0
    return emp_pdf, proj_pdf

# -----------------------------------------------------------------------------
# AI helpers
# -----------------------------------------------------------------------------
def parse_request_with_ai(user_query, llm_model):
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "Extract 'required_skills' (list), 'max_budget_per_hour' (number), and 'project_description' (string) from this request: '{query}'. Format as JSON."
    )
    chain = prompt | llm_model | parser
    return chain.invoke({"query": user_query})

def generate_justification_with_ai(team_df: pd.DataFrame, analysis: dict, query: str, llm_model):
    prompt = ChatPromptTemplate.from_template(
        "You are an HR strategist. Justify why this team is optimal for the request: '{query}'.\n\nTeam:\n{team_details}\n\nAnalysis:\n{analysis}\n\nBe concise and data-driven."
    )
    chain = prompt | llm_model
    response = chain.invoke({"query": query, "team_details": team_df.to_string(index=False), "analysis": str(analysis)})
    return response.content if hasattr(response, "content") else response

# -----------------------------------------------------------------------------
# Vector DB (Chroma)
# -----------------------------------------------------------------------------
def rebuild_vector_db(embedding_model, domain_filter: str | None = None) -> tuple[bool, str, int]:
    try:
        proj_sdf = get_delta_dataframe(PROJECTS_DELTA)  # Spark via resource cache
        pdf = proj_sdf.select("ProjectID", "Project_Description", "Domain").distinct().toPandas()
        if domain_filter:
            pdf = pdf[pdf["Domain"] == domain_filter]
        if pdf.empty:
            return False, "No projects available to index.", 0

        client = chromadb.PersistentClient(path="./chroma_db")
        name = "projects" if not domain_filter else f"projects_{domain_filter.lower()}"
        for c in client.list_collections():
            if c.name == name:
                client.delete_collection(name=name)
                break
        collection = client.create_collection(name=name, metadata={"space": "cosine", "dims": 384})

        docs = pdf["Project_Description"].tolist()
        ids = pdf["ProjectID"].astype(str).tolist()
        emb = embedding_model.encode(docs, show_progress_bar=False).tolist()
        collection.add(ids=ids, embeddings=emb, documents=docs)
        return True, f"Indexed {collection.count()} projects into '{name}'.", int(collection.count())
    except Exception as e:
        return False, f"Vector rebuild failed: {e}", 0

def ensure_vector_ready():
    client = chromadb.PersistentClient(path="./chroma_db")
    name = "projects"
    try:
        col = client.get_or_create_collection(name=name, metadata={"space": "cosine"})
        return True, name, int(col.count())
    except Exception:
        return False, name, 0

# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
def page_setup_utilities():
    
    st.title("⚙️ Setup & Utilities")
    st.caption("Run one-time data tasks, enrichment, governance, and health checks.")

    # Health checks
    with st.expander("Health checks", expanded=False):
        cols = st.columns(4)
        with cols[0]:
            st.write("Spark/Delta:")
            try:
                _ = spark.version
                st.success("Spark OK")
            except Exception as e:
                st.error(f"Spark error: {e}")
        with cols[1]:
            st.write("Employees table:")
            st.success("Present") if os.path.exists(EMPLOYEES_DELTA) else st.warning("Missing")
        with cols[2]:
            st.write("Projects table:")
            st.success("Present") if os.path.exists(PROJECTS_DELTA) else st.warning("Missing")
        with cols[3]:
            ready, cname, ccount = ensure_vector_ready()
            st.success(f"Chroma '{cname}' ready ({ccount})") if ready else st.warning("Chroma not initialized")

    st.markdown("---")

    # Data generation
    st.header("🗳️ Mock Data Generation")
    st.warning(f"This overwrites `{EMPLOYEES_CSV}` and `{PROJECTS_CSV}`.")
    if st.button("Generate New Mock Data"):
        generate_new_data()
        st.info("CSV generated. Now initialize Delta tables below.")

    # Init Delta
    st.header("⏩ Initialize Delta Tables")
    part_choice = st.selectbox("Employees partition by", ["Domain_Expertise", "Role"], index=0)
    if st.button("Initialize/Re-Initialize Delta from CSVs"):
        if os.path.exists(EMPLOYEES_DELTA): shutil.rmtree(EMPLOYEES_DELTA)
        if os.path.exists(PROJECTS_DELTA): shutil.rmtree(PROJECTS_DELTA)
        try:
            df = spark.read.csv(EMPLOYEES_CSV, header=True, inferSchema=True)
            df.write.format("delta").partitionBy(part_choice).mode("overwrite").save(EMPLOYEES_DELTA)
            dfp = spark.read.csv(PROJECTS_CSV, header=True, inferSchema=True)
            dfp.write.format("delta").partitionBy("Domain").mode("overwrite").save(PROJECTS_DELTA)
            st.success(f"Delta tables initialized (Employees partitioned by {part_choice}).")
            get_builder_data.clear()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Initialization failed: {e}")

    # Enrichment (Option A: overwriteSchema + MERGE)
    st.header("⚡ Enrich Employees with Avg_Performance (Option A)")
    if st.button("Pre-process and Enrich Data"):
        success, msg = pre_process_and_enrich_data(spark)
        if success:
            st.success(msg)
            st.session_state["data_enriched"] = True
            get_builder_data.clear()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error(msg)

    # Vector DB
    st.header("🧠 Vector Database")
    colv1, colv2, colv3 = st.columns([2,2,2])
    with colv1:
        if st.button("Rebuild 'projects' Collection"):
            ok, msg, cnt = rebuild_vector_db(embedding_model)
            st.success(msg) if ok else st.error(msg)
    with colv2:
        domain_sel = st.selectbox("Optional domain for separate collection", ["", "Finance","Healthcare","Retail","EdTech","Media","GovTech"])
        if st.button("Rebuild Domain Collection"):
            if not domain_sel:
                st.warning("Select a domain first.")
            else:
                ok, msg, cnt = rebuild_vector_db(embedding_model, domain_filter=domain_sel)
                st.success(msg) if ok else st.error(msg)
    with colv3:
        ready, cname, ccount = ensure_vector_ready()
        st.info(f"Current collection '{cname}' count: {ccount}")

    # Delta maintenance (if you kept compact_and_sort_delta in data_generator.py)
    st.header("🧹 Delta Maintenance (OSS-friendly)")
    st.caption("Compact and sort to reduce small files and improve locality (emulates OPTIMIZE/Z-Order).")
    zo_cols_emp = st.multiselect("Employees: sort by (Z-Order-like)", ["Domain_Expertise","Role","Age","Cost_per_Hour"])
    zo_cols_proj = st.multiselect("Projects: sort by (Z-Order-like)", ["Domain","Project_Success_Score","Individual_Performance_Score"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Compact + Sort Employees"):
            try:
                ok, msg = compact_and_sort_delta(spark, EMPLOYEES_DELTA, partition_col="Domain_Expertise", zorder_like_cols=zo_cols_emp)
                st.success(msg) if ok else st.error(msg)
            except Exception as e:
                st.error(f"Employees compaction failed: {e}")
    with c2:
        if st.button("Compact + Sort Projects"):
            try:
                ok, msg = compact_and_sort_delta(spark, PROJECTS_DELTA, partition_col="Domain", zorder_like_cols=zo_cols_proj)
                st.success(msg) if ok else st.error(msg)
            except Exception as e:
                st.error(f"Projects compaction failed: {e}")

    # Micro-benchmark
    st.header("⏱️ Micro-benchmark")
    if st.button("Run Benchmark (role counts, cost histogram)"):
        try:
            t0 = time.time()
            edf = get_delta_dataframe(EMPLOYEES_DELTA)
            pdf = get_delta_dataframe(PROJECTS_DELTA)
            role_counts = edf.groupBy("Role").count().orderBy("count", ascending=False).count()
            _ = pdf.groupBy("Domain").count().count()
            dt = time.time() - t0
            st.success(f"Bench ran in {dt:.3f}s (two group-bys).")
        except Exception as e:
            st.error(f"Benchmark failed: {e}")

def page_team_builder():
    st.title("🧠 Team Builder")
    st.caption("Describe a project, parse constraints with an LLM, retrieve similar history, and optimize a team.")

    with st.sidebar:
        st.subheader("AI & Solver Config")
        provider = st.selectbox("LLM", ["Google Gemini", "Ollama (Local)"])
        if provider == "Google Gemini":
            google_api_key = st.text_input("Google API Key", type="password")
            model_name = st.selectbox("Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
        else:
            ollama_model = st.text_input("Ollama Model", value="llama3:8b")
        st.markdown("---")
        st.subheader("Optimization Weights")
        fit_weight = st.slider("Fit weight", 1, 5, 3, 1)
        perf_weight = st.slider("Performance weight", 1, 5, 2, 1)
        min_avg_age = st.slider("Min avg age", 25, 45, 30, 1)
        max_team_size = st.number_input("Max team size (0=unbounded)", min_value=0, value=0, step=1)

    with st.form("project_form"):
        user_query = st.text_area("Describe your new project:", height=140)
        submitted = st.form_submit_button("Assemble Optimal Team")

    if not submitted:
        return

    # LLM init
    llm = None
    try:
        if provider == "Google Gemini":
            if not google_api_key:
                st.error("Enter Google API Key.")
                return
            llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=google_api_key, temperature=0)
        else:
            llm = ChatOllama(model=ollama_model, temperature=0)
    except Exception as e:
        st.error(f"LLM init failed: {e}")
        return

    # Ensure fresh data after enrichment
    if st.session_state.get("data_enriched", False):
        get_builder_data.clear()
        st.session_state["data_enriched"] = False

    employees_df, projects_df = get_builder_data()
    if employees_df is None:
        st.error("Missing Delta tables. Initialize on Setup page.")
        return

    # Vector retrieval
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="projects")

    parsed = parse_request_with_ai(user_query, llm)
    required_skills = parsed.get("required_skills", []) or []
    max_budget = parsed.get("max_budget_per_hour", 0) or 0
    description = parsed.get("project_description", "") or user_query

    q_emb = embedding_model.encode(description).tolist()
    similar = collection.query(query_embeddings=[q_emb], n_results=50)

    project_fit_scores = {}
    if similar.get("ids") and projects_df is not None and len(similar["ids"]) > 0:
        similar_pids = [int(pid) for pid in similar["ids"][0]]
        relevant = projects_df[projects_df["ProjectID"].isin(similar_pids)]
        counts = relevant["EmployeeID"].value_counts()
        project_fit_scores = (counts / counts.max()).to_dict()

    rec_ids, analysis = find_optimal_team(
        employees_df=employees_df,
        project_fit_scores=project_fit_scores,
        required_skills=required_skills,
        max_budget_per_hour=max_budget,
        min_avg_age=float(min_avg_age),
        fit_weight=int(fit_weight),
        performance_weight=int(perf_weight),
        max_team_size=(int(max_team_size) if max_team_size > 0 else None),
        time_limit_s=10,
        workers=8,
    )

    if not rec_ids:
        st.error(analysis.get("status", "No team found."))
        return

    st.session_state["last_query"] = user_query
    st.session_state["recommended_team_ids"] = rec_ids

    team_df = employees_df[employees_df["EmployeeID"].isin(rec_ids)]
    justif = generate_justification_with_ai(team_df, analysis, user_query, llm)

    st.success("✅ Team Assembled")
    st.subheader("AI Justification")
    st.write(justif)
    st.subheader("Team Details")
    st.dataframe(team_df[["Name","Age","Role","Skills","Cost_per_Hour","Avg_Performance"]], use_container_width=True)

    st.info("Proceed to Project Lifecycle to finalize and write records atomically.")

def page_lifecycle():
    st.title("🔄 Project Lifecycle")
    st.caption("Finalize team, record performance, and append records atomically to Delta.")

    if "recommended_team_ids" not in st.session_state:
        st.warning("Generate a team in Team Builder first.")
        return

    if st.session_state.get("data_enriched", False):
        get_builder_data.clear()
        st.session_state["data_enriched"] = False

    employees_df, _ = get_builder_data()
    if employees_df is None:
        st.error("Missing employees table.")
        return

    rec_df = employees_df[employees_df["EmployeeID"].isin(st.session_state["recommended_team_ids"])]

    avail = employees_df[employees_df["Availability"] == "Available"]
    final_names = st.multiselect("Finalize team members:", options=avail["Name"], default=rec_df["Name"].tolist())
    final_df = employees_df[employees_df["Name"].isin(final_names)]
    st.dataframe(final_df[["Name","Role","Skills","Cost_per_Hour"]], use_container_width=True)

    st.subheader("Complete Project")
    with st.form("complete_form"):
        q = st.session_state.get("last_query", "")
        proj_name = st.text_input("Project Name", f"New Project based on: {q[:30]}…")
        proj_desc = st.text_area("Project Description", f"Completed project based on request: {q}")
        st.markdown("---")

        if not final_df.empty:
            st.markdown("Enter Final Performance Scores (0.0–1.0):")
            perf_scores = {}
            for _, row in final_df.iterrows():
                perf_scores[row["EmployeeID"]] = st.slider(
                    f"{row['Name']}", 0.0, 1.0, 0.85, 0.01, key=f"perf_{row['EmployeeID']}"
                )
        else:
            st.warning("No team members selected.")
            perf_scores = {}

        done = st.form_submit_button("✅ Complete & Save (ACID)")

    if not done:
        return
    if not perf_scores:
        st.error("Cannot complete with empty team.")
        return

    with st.spinner("Writing atomically to Delta..."):
        ptab = get_delta_dataframe(PROJECTS_DELTA)
        new_id_row = ptab.select(spark_max("ProjectID")).first()
        new_pid = (new_id_row[0] if new_id_row and new_id_row[0] is not None else 5000) + 1

        records = []
        avg_success = sum(perf_scores.values()) / len(perf_scores)
        for emp_id, score in perf_scores.items():
            row = final_df[final_df["EmployeeID"] == emp_id].iloc[0]
            records.append({
                "ProjectID": int(new_pid),
                "EmployeeID": int(emp_id),
                "Project_Name": proj_name,
                "Project_Description": proj_desc,
                "Tech_Stack_Used": row["Skills"],
                "Domain": row["Domain_Expertise"],
                "Project_Role": row["Role"],
                "Project_Success_Score": float(avg_success),
                "Individual_Performance_Score": float(score),
            })

        sdf = spark.createDataFrame(records)
        target_schema = ptab.schema
        for field in target_schema:
            sdf = sdf.withColumn(field.name, col(field.name).cast(field.dataType))
        sdf.write.format("delta").mode("append").save(PROJECTS_DELTA)

        st.success(f"✅ Saved Project {new_pid} and team records.")
        for k in ["last_query", "recommended_team_ids"]:
            st.session_state.pop(k, None)
        get_builder_data.clear()
        st.cache_data.clear()
        st.rerun()

def page_data_management():
    st.title("🗃️ Data Management (Delta)")
    st.caption("Time travel, history, MERGE updates, and live editing.")

    # History + time travel
    st.subheader("History & Time Travel (Employees)")
    if os.path.exists(EMPLOYEES_DELTA):
        try:
            dt = DeltaTable.forPath(spark, EMPLOYEES_DELTA)
            hist = dt.history().select("version","timestamp","operation","operationMetrics").orderBy("version", ascending=False).limit(10)
            st.dataframe(hist.toPandas(), use_container_width=True)
            versions = [r.version for r in hist.select("version").collect()]
            if len(versions) == 0:
                st.info("No versions yet.")
            elif len(versions) == 1:
                st.info(f"Single version: {versions[0]}")
                dfv = read_delta_as_pandas(EMPLOYEES_DELTA, version=versions[0])
                st.dataframe(dfv, use_container_width=True)
            else:
                c1, c2 = st.columns(2)
                with c1:
                    v1 = st.select_slider("Left version", options=versions, value=max(versions))
                    left = read_delta_as_pandas(EMPLOYEES_DELTA, version=v1)
                    st.dataframe(left, use_container_width=True)
                with c2:
                    v2 = st.select_slider("Right version", options=versions, value=min(versions))
                    right = read_delta_as_pandas(EMPLOYEES_DELTA, version=v2)
                    st.dataframe(right, use_container_width=True)
        except Exception as e:
            st.warning(f"History unavailable: {e}")
    else:
        st.info("Initialize employees table in Setup.")

    st.markdown("---")

    emp_tab, proj_tab = st.tabs(["Employees", "Projects"])

    with emp_tab:
        st.subheader("MERGE Upload (CSV)")
        up = st.file_uploader("Upload employees.csv", type="csv", key="emp_up")
        if up and st.button("MERGE Employees"):
            try:
                updates = spark.createDataFrame(pd.read_csv(up))
                d = DeltaTable.forPath(spark, EMPLOYEES_DELTA)
                (d.alias("emp").merge(updates.alias("u"), "emp.EmployeeID = u.EmployeeID")
                   .whenMatchedUpdateAll()
                   .whenNotMatchedInsertAll()
                   .execute())
                st.success("Employees merged.")
                get_builder_data.clear()
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"MERGE failed: {e}")

        st.subheader("Live Edit")
        try:
            sdf = get_delta_dataframe(EMPLOYEES_DELTA)
            edited = st.data_editor(sdf.toPandas(), num_rows="dynamic", use_container_width=True, key="emp_editor")
            if st.button("Save Employee Edits (MERGE)"):
                updates = spark.createDataFrame(edited)
                d = DeltaTable.forPath(spark, EMPLOYEES_DELTA)
                (d.alias("emp").merge(updates.alias("u"), "emp.EmployeeID = u.EmployeeID")
                   .whenMatchedUpdateAll()
                   .whenNotMatchedInsertAll()
                   .execute())
                st.success("Edits saved.")
                get_builder_data.clear()
                st.cache_data.clear()
                st.rerun()
        except Exception as e:
            st.error(f"Live edit failed: {e}")
    # Add a Projects history & time travel section like Employees
    st.subheader("History & Time Travel (Projects)")
    if os.path.exists(PROJECTS_DELTA):
        try:
            dtp = DeltaTable.forPath(spark, PROJECTS_DELTA)
            phist = dtp.history().select("version","timestamp","operation","operationMetrics").orderBy("version", ascending=False).limit(10)
            st.dataframe(phist.toPandas(), use_container_width=True)

            pversions = [r.version for r in phist.select("version").collect()]
            if len(pversions) == 0:
                st.info("No project table versions yet.")
            elif len(pversions) == 1:
                st.info(f"Single version: {pversions[0]}")
                pv = read_delta_as_pandas(PROJECTS_DELTA, version=pversions[0])
                st.dataframe(pv, use_container_width=True)
            else:
                c1, c2 = st.columns(2)
                with c1:
                    pv1 = st.select_slider("Projects left version", options=pversions, value=max(pversions))
                    leftp = read_delta_as_pandas(PROJECTS_DELTA, version=pv1)
                    st.dataframe(leftp, use_container_width=True)
                with c2:
                    pv2 = st.select_slider("Projects right version", options=pversions, value=min(pversions))
                    rightp = read_delta_as_pandas(PROJECTS_DELTA, version=pv2)
                    st.dataframe(rightp, use_container_width=True)
        except Exception as e:
            st.warning(f"Projects history unavailable: {e}")
    else:
        st.info("Initialize projects table in Setup.")

    with proj_tab:
        st.subheader("MERGE Upload (CSV)")
        up2 = st.file_uploader("Upload projects_history.csv", type="csv", key="proj_up")
        if up2 and st.button("MERGE Projects"):
            try:
                updates = spark.createDataFrame(pd.read_csv(up2))
                d = DeltaTable.forPath(spark, PROJECTS_DELTA)
                (d.alias("p").merge(updates.alias("u"),
                "p.ProjectID = u.ProjectID AND p.EmployeeID = u.EmployeeID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute())
                st.success("Projects merged.")
                ok, msg, cnt = rebuild_vector_db(embedding_model)
                st.info(msg if ok else f"Vector note: {msg}")
                get_builder_data.clear()
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"MERGE failed: {e}")

        st.subheader("Current Projects")
        try:
            proj_pdf_current = read_delta_as_pandas(PROJECTS_DELTA)
            st.dataframe(proj_pdf_current, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load current projects: {e}")
    

def page_analytics():
    st.title("📊 Analytics")
    edf_pdf = read_delta_as_pandas(EMPLOYEES_DELTA)
    if edf_pdf is None or edf_pdf.empty:
        st.warning("No employees table. Initialize data.")
        return
    if "Avg_Performance" not in edf_pdf.columns:
        st.warning("`Avg_Performance` missing. Run enrichment for full insights.")
        df = edf_pdf.copy()
        df["Avg_Performance"] = 0.0
    else:
        df = edf_pdf

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Roles Distribution")
        counts = df["Role"].value_counts()
        fig = px.pie(counts, values=counts.values, names=counts.index, title="Role Breakdown")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Cost/Hour Distribution")
        fig2 = px.histogram(df, x="Cost_per_Hour", nbins=25, title="Hourly Costs")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("K-Means Clustering")
    feats_all = ["Age","Cost_per_Hour","Avg_Performance"]
    sel_feats = st.multiselect("Features", feats_all, default=feats_all)
    k = st.slider("k (clusters)", 2, 8, 4, 1)
    if len(df) >= k and sel_feats:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        from sklearn.decomposition import PCA
        X = df[sel_feats].fillna(0.0).to_numpy()
        Xs = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit(Xs)
        df["Cluster"] = km.labels_.astype(str)
        sil = silhouette_score(Xs, km.labels_) if len(df) > k else float("nan")
        pca = PCA(n_components=2)
        red = pca.fit_transform(Xs)
        df["pca1"] = red[:,0]
        df["pca2"] = red[:,1]
        figc = px.scatter(df, x="pca1", y="pca2", color="Cluster",
                          hover_name="Name",
                          hover_data=["Role","Cost_per_Hour","Avg_Performance"],
                          title=f"Clusters (silhouette={sil:.3f})")
        st.plotly_chart(figc, use_container_width=True)
    else:
        st.info("Select features and ensure enough rows for clustering.")

    st.subheader("Ad-Hoc Spark SQL")
    templates = {
        "Top roles": "SELECT Role, count(*) AS cnt FROM employees GROUP BY Role ORDER BY cnt DESC",
        "Avg perf by role": "SELECT Role, avg(Avg_Performance) AS avg_perf FROM employees GROUP BY Role ORDER BY avg_perf DESC",
        "Domain staff counts": "SELECT Domain_Expertise AS Domain, count(*) AS cnt FROM employees GROUP BY Domain ORDER BY cnt DESC",
        "Project count per employee": "SELECT EmployeeID, count(DISTINCT ProjectID) AS projects FROM projects GROUP BY EmployeeID ORDER BY projects DESC LIMIT 20"
    }
    tname = st.selectbox("Template", list(templates.keys()), index=0)
    query = st.text_area("SQL", value=templates[tname], height=140)
    if st.button("Run SQL"):
        try:
            e = get_delta_dataframe(EMPLOYEES_DELTA)
            p = get_delta_dataframe(PROJECTS_DELTA)
            if e is not None: e.createOrReplaceTempView("employees")
            if p is not None: p.createOrReplaceTempView("projects")
            out = spark.sql(query).toPandas()
            st.dataframe(out, use_container_width=True)
        except Exception as e:
            st.error(f"Query failed: {e}")

# -----------------------------------------------------------------------------
# Navigation (st.Page / st.navigation)
# -----------------------------------------------------------------------------
def app_router():
    tb = st.Page(page_team_builder, title="Team Builder", icon="🧠", url_path="team")
    lc = st.Page(page_lifecycle, title="Project Lifecycle", icon="🔄", url_path="lifecycle")
    dm = st.Page(page_data_management, title="Data Management", icon="🗃️", url_path="data")
    an = st.Page(page_analytics, title="Analytics", icon="📊", url_path="analytics")
    su = st.Page(page_setup_utilities, title="Setup & Utilities", icon="⚙️", url_path="setup", default=True)
    nav = st.navigation([tb, lc, dm, an, su])
    nav.run()
    return None 
if __name__ == "__main__":
    app_router()