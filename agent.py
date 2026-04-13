import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langgraph.prebuilt import create_react_agent


load_dotenv()

# 2. Connect to Databases

qdrant_client = QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="document_knowledge",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)


graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)



# (The Agent's "Hands")

@tool
def search_unstructured_text(query: str) -> str:
    """
    Use this tool FIRST for almost all questions. It searches the actual text of the PDF.
    Use this to find specific facts, financial numbers, metrics (like revenue, dividends, repurchases), 
    summaries, quotes, and semantic information.
    """
    docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def search_structured_graph(query: str) -> str:
    """
    Use this tool ONLY when the user asks about the physical structure or layout of the document.
    Examples: "What page is this on?", "Are there any tables?", "Where is the text located on the page?"
    DO NOT use this tool to search for specific financial facts, amounts, or general document knowledge.
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    # This automatically writes and executes a Cypher query based on the question
    chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True)
    try:
        return chain.invoke({"query": query})["result"]
    except Exception as e:
        return f"Graph search failed: {e}"

# 4. Initialize the LangGraph Agent
tools = [search_unstructured_text, search_structured_graph]
agent_llm = ChatOpenAI(temperature=0, model="gpt-4o")

# LangGraph's prebuilt ReAct agent handles the routing automatically!
agent_executor = create_react_agent(agent_llm, tools)

def ask_agent(question: str):
    """Sends a question to the agent and streams its thought process."""
    print(f"\n🗣️  User: {question}")
    print("-" * 50)
    
    # Run the agent state machine
    for step in agent_executor.stream({"messages": [("user", question)]}):
        for node_name, node_state in step.items():
            if node_name == "tools":
                # Print which database the agent decided to use
                used_tool = node_state["messages"][0].name
                print(f"🧠 Agent decided to route to: [{used_tool}]")
            elif node_name == "agent":
                # Print the final generated answer
                print(f"🤖 Agent Answer: {node_state['messages'][0].content}")


if __name__ == "__main__":
    print("🤖 Agent Terminal Started! Type 'exit' to quit.")
    print("-" * 50)
    
    while True:
        # 1. Get dynamic input from the user
        user_input = input("\n🗣️ You: ")
        
        # 2. Allow the user to quit the loop
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Shutting down agent...")
            break
            
        # 3. Pass the dynamic input to the agent
        ask_agent(user_input)