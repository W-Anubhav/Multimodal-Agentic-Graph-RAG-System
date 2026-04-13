# Multimodal-Agentic-Graph-RAG-System
# Multimodal Agentic Graph-RAG Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph_DB-blue)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-Vision-red)

## 📌 Overview
Standard RAG (Retrieval-Augmented Generation) systems flatten complex PDFs into plain text, destroying spatial layouts, tables, and document structures. This often causes LLMs to hallucinate or fail when asked highly specific questions about financial reports, invoices, or visual data.

This project solves that by implementing a **Multimodal Agentic Graph-RAG Architecture**. It uses Computer Vision to "see" the document, stores the physical layout and structural relationships in a Graph Database, and stores the semantic narrative in a Vector Database. Finally, an autonomous AI Agent intelligently routes user queries to the correct database to fetch exact, hallucination-free answers.

## 🏗️ Architecture & Dual-Database Memory
* **The "Structural" Brain (Neo4j):** Maps the physical layout of the PDF. It stores pages, bounding boxes, and tables as nodes and edges. If a user asks, *"What is the exact revenue in the table on page 6?"*, the system queries Neo4j to find the precise coordinates and data without guessing.
* **The "Semantic" Brain (Qdrant):** Stores the heavy narrative text (e.g., the "Management Discussion") chopped into 500-character overlapping chunks and embedded as 1,536-dimensional vectors. It is used for answering contextual questions like, *"Why did the company struggle with supply chain issues?"*

## ⚙️ Tech Stack
* **Agent & Orchestration:** LangChain, LangGraph (ReAct Agent)
* **Graph Database:** Neo4j (Aura Cloud)
* **Vector Database:** Qdrant (Local Docker)
* **Computer Vision Ingestion:** PyTorch, Tesseract OCR, Poppler
* **LLM & Embeddings:** OpenAI (`gpt-4o`, `text-embedding-3-small`)
* **Frontend:** Chainlit

## 📂 Project Structure
* `app.py`: The Chainlit frontend interface.
* `pipeline.py`: The orchestrator that manages the ingestion workflow.
* `ingestor.py`: Uses Computer Vision (PyTorch/Tesseract) to extract raw text and bounding boxes from PDFs.
* `graph_uploader.py`: Sanitizes spatial data (JSON serialization) and uploads physical document layouts to Neo4j via Cypher queries.
* `semantic_extractor.py`: Uses an LLM Graph Transformer to extract explicit entities/relationships from text and store them in Neo4j.
* `vector_uploader.py`: Chunks text and uploads dense embeddings to Qdrant.
* `agent.py`: The LangGraph routing brain equipped with custom tools and engineered prompt guardrails.

## 🚀 Getting Started

### Prerequisites
1. Docker Desktop (for Qdrant)
2. Python 3.10+
3. Tesseract OCR & Poppler installed on your system (added to PATH).
4. A Neo4j Aura account (Free tier) and OpenAI API Key.

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Agentic-Graph-RAG.git](https://github.com/YourUsername/Agentic-Graph-RAG.git)
   cd Agentic-Graph-RAG
