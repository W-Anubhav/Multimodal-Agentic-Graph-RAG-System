import os
import json 
from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv(override=True)
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

class GraphUploader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def upload_document_data(self, doc_name, document_data):
        """
        Takes the output from ingestor.py and uploads it to Neo4j.
        """
        
        # We loop through the data and stringify the bounding boxes AND labels
        # so Neo4j just sees them as standard text properties instead of illegal nested lists.
        for page in document_data:
            if 'text_data' in page and 'boxes' in page['text_data']:
                page['text_data']['boxes'] = [json.dumps(box) for box in page['text_data']['boxes']]
            
            if 'table_data' in page:
                if 'boxes' in page['table_data']:
                    page['table_data']['boxes'] = [json.dumps(box) for box in page['table_data']['boxes']]
                if 'labels' in page['table_data']:
                    page['table_data']['labels'] = [json.dumps(label) for label in page['table_data']['labels']]
        

        # The Cypher query to build the graph structure
        query = """
        // 1. Create the Main Document Node
        MERGE (d:Document {name: $doc_name})
        
        WITH d
        UNWIND $pages AS page_data
        
        // 2. Create Page Nodes connected to the Document
        MERGE (p:Page {page_number: page_data.page_number, document: $doc_name})
        MERGE (d)-[:HAS_PAGE]->(p)
        
        WITH p, page_data
        
        // 3. Create Text Nodes
        // We use FOREACH as a conditional loop to handle lists of text
        FOREACH (i IN range(0, size(page_data.text_data.words)-1) |
            MERGE (t:TextEntity {
                text: page_data.text_data.words[i],
                bbox: page_data.text_data.boxes[i],
                page: p.page_number
            })
            MERGE (p)-[:CONTAINS_TEXT]->(t)
        )
        
        // 4. Create Table Nodes
        FOREACH (j IN range(0, size(page_data.table_data.boxes)-1) |
            MERGE (tbl:TableEntity {
                label_id: page_data.table_data.labels[j], // <-- REMOVED toString()
                bbox: page_data.table_data.boxes[j],
                page: p.page_number
            })
            MERGE (p)-[:CONTAINS_TABLE]->(tbl)
        )
        """
        
        # Execute the query
        with self.driver.session() as session:
            session.execute_write(
                lambda tx: tx.run(query, doc_name=doc_name, pages=document_data)
            )
            print(f"✅ Successfully uploaded {doc_name} to Neo4j Aura!")


if __name__ == "__main__":
    # this simulates the output format from the ingestor.py
    mock_ingestor_output = [
        {
            "page_number": 1,
            "text_data": {
                "words": ["Revenue", "increased", "by", "10%"],
                "boxes": [[10, 10, 50, 20], [55, 10, 100, 20], [105, 10, 120, 20], [125, 10, 150, 20]]
            },
            "table_data": {
                "boxes": [[10, 50, 200, 300]], 
                "labels": [[6, 6, 6]] 
            }
        }
    ]

    # Run the uploader
    uploader = GraphUploader(URI, USERNAME, PASSWORD)
    try:
        uploader.upload_document_data("Q3_Financial_Report.pdf", mock_ingestor_output)
    finally:
        uploader.close()