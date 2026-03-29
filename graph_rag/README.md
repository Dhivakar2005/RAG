# GraphRAG Local Implementation

This project is a completely offline, local AI assistant using Graph Retrieval-Augmented Generation (GraphRAG). It uses Python, a Neo4j Graph Database, and Microsoft's `phi3:mini` Model running on Ollama for local inference.

## Prerequisites

1. **Python 3.9+**
2. **Neo4j** (Either Neo4j Desktop, Neo4j Community Server, or Docker version)
3. **Ollama** installed on your machine.

---

## 🛠️ Step 1: Install Neo4j & Ollama

### Install Neo4j
You can download Neo4j Desktop from their website. Start a new local database instance.
By default, Neo4j uses the connection URI `bolt://localhost:7687` and user `neo4j`.
Be sure to set a password during database creation. Open `config.py` in this project and update the `NEO4J_PASSWORD` environment variable or the string directly in the code to match your database password!

*(Alternatively, to run via docker: `docker run -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest`)*

### Install Ollama
Download and install [Ollama](https://ollama.com). Open your terminal/command prompt and pull the local model needed for this application:
```bash
ollama pull phi3:mini
```
Ensure Ollama is running in the background.

---

## 📦 Step 2: Install Python Dependencies

Open a terminal in the root directory of this project (`graph_rag_local`), and optimally inside a virtual environment, run:

```bash
pip install -r requirements.txt
```

---

## 🚀 Step 3: Run the System

### 1. Ingest Data
A sample knowledge file is located at `data/knowledge.txt`. Run the ingestion pipeline, which splits the document into chunks, instructs the LLM to extract entities and connections to form a JSON object schema, and populates the Neo4j database graph:

```bash
python ingest.py
```

### 2. Chat with the GraphRAG Assistant
Once ingestion is successfully finished, you can query your knowledge base interactively. Run the query script:

```bash
python query.py
```

Type a question like: *"What is GraphRAG?"* or *"Who develops Phi-3?"*.
The CLI system will dynamically convert the entities inside the question, look up connected paths in your unified knowledge graph, and feed this relationship context cleanly back to the LLM to generate an answer.
