import chromadb
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession
import time

def main():
    """
    This script performs a one-time setup of the local vector database.
    It loads project data using Spark, generates embeddings for each project
    description using a sentence-transformer model, and stores them in ChromaDB.
    """
    # --- Step 1: Initialize Spark and Load Project Data ---
    spark = SparkSession.builder \
        .appName("TeamFormationCortex-VectorDB") \
        .master("local[*]") \
        .getOrCreate()
    
    print("Spark session initialized for Vector DB creation.")
    spark.sparkContext.setLogLevel("ERROR")

    projects_df = spark.read.csv("projects_history.csv", header=True, inferSchema=True)

    # To avoid redundant work, we only need one description per unique project.
    print("Fetching distinct project descriptions from Spark...")
    distinct_projects_df = projects_df.select("ProjectID", "Project_Description").distinct()
    
    # Collect the data to the driver node. This is feasible for our local setup.
    # For a true big data environment, this would be a distributed process.
    projects_to_embed = distinct_projects_df.collect()
    print(f"Found {len(projects_to_embed)} unique projects to process.")
    
    spark.stop()
    print("Spark session stopped.")

    # --- Step 2: Initialize Model and Vector Database ---

    # Load a pre-trained model from Hugging Face. 
    # 'all-MiniLM-L6-v2' is a great starting point - fast and effective.
    print("Loading sentence-transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully.")

    # Initialize a persistent ChromaDB client. 
    # This will store the database files in a 'chroma_db' directory.
    db_client = chromadb.PersistentClient(path="./chroma_db")

    # Create a new collection or get it if it already exists.
    # This makes the script safe to run multiple times.
    collection_name = "projects"
    collection = db_client.get_or_create_collection(name=collection_name)
    print(f"ChromaDB collection '{collection_name}' ready.")

    # --- Step 3: Generate and Store Embeddings ---

    print("Generating and storing embeddings... This may take a few minutes.")
    start_time = time.time()
    
    # Prepare data for ChromaDB in batches to be efficient
    project_ids = [str(row['ProjectID']) for row in projects_to_embed]
    documents = [row['Project_Description'] for row in projects_to_embed]
    
    # Generate embeddings for all documents at once
    embeddings = model.encode(documents, show_progress_bar=True)
    
    # Add the data to the ChromaDB collection
    # We use the ProjectID as the unique ID for each entry.
    collection.add(
        ids=project_ids,
        embeddings=embeddings.tolist(), # ChromaDB expects a list of lists
        documents=documents
    )
    
    end_time = time.time()
    print(f"\nSuccessfully added {collection.count()} embeddings to the database.")
    print(f"Process completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
