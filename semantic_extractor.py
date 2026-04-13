import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph


load_dotenv()


graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# 3. Initialize the LLM and the Graph Transformer
# Using a fast/cheap model for development. 
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini") 

#  definining what the LLM is allowed to extract so the graph doesn't become messy
graph_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Company", "Person", "Metric", "Date", "Product"],
    allowed_relationships=["REPORTED", "EMPLOYED_BY", "HAS_VALUE", "OCCURRED_ON", "PRODUCES"]
)

def extract_and_store_semantics(document_name, raw_text):
    """
    Takes reconstructed text, extracts meaningful graph relationships, 
    and saves them directly to your Neo4j database.
    """
    print(f"Extracting meaning from: {document_name}...")
    
    # Wrap the text in a LangChain Document
    doc = Document(
        page_content=raw_text,
        metadata={"source": document_name}
    )
    
    # Let the LLM figure out the Nodes and Relationships
    graph_documents = graph_transformer.convert_to_graph_documents([doc])
    
    # Push the semantic data directly to Neo4j
    # include_source=True automatically links these new facts back to a "Document" node
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True 
    )
    
    print(f"✅ Successfully added semantic nodes for {document_name}!")
    return graph_documents


if __name__ == "__main__":
   
    sample_text = "Nvidia reported a revenue increase of 10% in Q3 2024. Jensen Huang is the CEO."
    
   
    extracted_data = extract_and_store_semantics("Q3_Financial_Report.pdf", sample_text)
    
   
    print("\n--- Extracted Graph Data ---")
    for node in extracted_data[0].nodes:
        print(f"Node: {node.id} ({node.type})")
        
    print("\nRelationships:")
    for rel in extracted_data[0].relationships:
        print(f"{rel.source.id} --[{rel.type}]--> {rel.target.id}")