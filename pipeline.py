import os
from dotenv import load_dotenv

load_dotenv(override=True)

from ingestor import process_file
from graph_uploader import GraphUploader
from semantic_extractor import extract_and_store_semantics
from vector_uploader import upload_to_qdrant



uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

def run_full_pipeline(file_path):
    """Orchestrates the extraction and uploading of a real document."""
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return

    # Extract just the filename (e.g., 'report.pdf') to use as the Document ID
    document_name = os.path.basename(file_path)
    print(f"\n🚀 Starting AI Pipeline for: {document_name}")
    print("=" * 50)

    #  STEP 1: Computer Vision Extraction 
    print("\n[1/4] Running PyTorch Ingestor (LayoutLMv3 + TATR)...")
    document_data = process_file(file_path)
    
    #  STEP 2: Upload Spatial Structure to Neo4j 
    print("\n[2/4] Uploading Spatial Graph to Neo4j...")
    uploader = GraphUploader(uri, user, password)
    try:
        uploader.upload_document_data(document_name, document_data)
    finally:
        uploader.close()

    #  Combine words into paragraphs for Steps 3 & 4 
    #  take the individual words from PyTorch and stitch them back together
    full_raw_text = ""
    for page in document_data:
        full_raw_text += " ".join(page["text_data"]["words"]) + " "

    #  LLM Semantic Extraction to Neo4j 
    print("\n[3/4] Extracting Semantic Logic (LangChain -> Neo4j)...")
    extract_and_store_semantics(document_name, full_raw_text)

    #  Vector Embedding to Qdrant 
    print("\n[4/4] Embedding text to Vector Database (Qdrant)...")
    upload_to_qdrant(document_name, full_raw_text)

    print("\n✅ Pipeline Complete! The document is ready for the LangGraph Agent.")

if __name__ == "__main__":
    print("🧠 Multimodal Graph-RAG Pipeline")
    print("-" * 50)
    
    while True:
        # Dynamic input: Ask the user for any file on their computer
        target_file = input("\nEnter the path to your PDF or Image (or 'quit' to exit): ")
        
        if target_file.lower() in ['quit', 'exit', 'q']:
            print("Shutting down pipeline.")
            break
            
        run_full_pipeline(target_file)