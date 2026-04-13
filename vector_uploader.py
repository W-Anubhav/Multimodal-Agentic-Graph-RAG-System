import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


load_dotenv()


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "document_knowledge"

client = QdrantClient(url=QDRANT_URL)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. Create the Collection if it doesn't exist
# text-embedding-3-small outputs 1536 dimensions
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

def upload_to_qdrant(document_name, raw_text):
    """
    Chunks the text, embeds it, and stores it in Qdrant.
    """
    print(f"Preparing to embed and store: {document_name}...")
    
    # Split text into overlapping chunks to preserve context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Create LangChain Documents with metadata linking back to the source
    chunks = text_splitter.create_documents(
        texts=[raw_text],
        metadatas=[{"source": document_name}]
    )
    
    # Initialize the Vector Store connection
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    
    # Upload the chunks
    vector_store.add_documents(chunks)
    
    print(f"✅ Successfully uploaded {len(chunks)} chunks to Qdrant!")
    return True


if __name__ == "__main__":
    # Simulating a longer text extracted from the document
    sample_text = (
        "Nvidia reported a revenue increase of 10% in Q3 2024. "
        "Jensen Huang is the CEO. The new H100 chips are driving massive "
        "growth in the datacenter sector. " * 10 
    )
    
    upload_to_qdrant("Q3_Financial_Report.pdf", sample_text)