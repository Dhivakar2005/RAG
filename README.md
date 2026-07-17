# 🧠 Multi-Architecture RAG Systems

Welcome to the **RAG (Retrieval-Augmented Generation)** repository. This workspace contains three distinct implementations of Retrieval-Augmented Generation, showcasing different databases, LLM providers, and data architectures:

1. **🤖 Interactive Doc RAG (`rag.py`)**: A Streamlit-based UI that processes uploaded PDFs locally using Ollama (`llama3`) and an in-memory vector store.
2. **⚡ ChromaDB + Gemini RAG (`rag_with_chromadb.py`)**: A persistent command-line RAG application powered by ChromaDB and Google's Gemini API (`gemini-2.5-flash`).
3. **🕸️ GraphRAG Local (`graph_rag/`)**: An offline knowledge graph-based RAG pipeline using Neo4j and Ollama (`phi3:mini`).

---

## 🛠️ Global Prerequisites

Before running the applications, ensure you have:
* **Python 3.9+** installed.
* **Ollama** installed and running (for local models).
* **Neo4j** (Desktop or Docker) running (only required for GraphRAG).

---
## 📚 What You'll Learn

- Retrieval-Augmented Generation (RAG)
- Vector Databases
- Embeddings
- Semantic Search
- Knowledge Graphs
- Neo4j
- LangChain
- Ollama
- Google Gemini

  ---
## 🚀 1. Interactive Doc RAG (Streamlit + Ollama)

A highly visual, web-based UI that allows you to upload any research PDF and chat with it in real-time.

### Setup & Run
1. **Pull the Ollama model**:
   ```bash
   ollama pull llama3
   ```
2. **Install requirements**:
   Ensure you have the required packages:
   ```bash
   pip install streamlit langchain langchain-community langchain-text-splitters langchain-ollama pdfplumber
   ```
3. **Start the application**:
   ```bash
   streamlit run rag.py
   ```
4. **Use**: Open the local URL (usually `http://localhost:8501`), upload a PDF file from the UI, and start chatting.

---

## ⚡ 2. ChromaDB + Gemini RAG (Persistent CLI)

A fast, command-line interface using Google Gemini embeddings and text generation, saving document indices to a persistent Chroma database.

### Setup & Run
1. **Install requirements**:
   ```bash
   pip install chromadb google-generativeai pypdf
   ```
2. **Configure your API Key**:
   Open [rag_with_chromadb.py](file:///d:/03_Visual_Studio/Ai/RAG/rag_with_chromadb.py) and update the Google Gemini API key:
   ```python
   genai.configure(api_key="your_gemini_api_key_here")
   ```
3. **Add a PDF**:
   Place your PDF file in the `documents/` folder. By default, it looks for `documents/attention-is-all-you-need.pdf`.
4. **Run the script**:
   ```bash
   python rag_with_chromadb.py
   ```
5. **Use**: Ask questions in the terminal, or type `exit` to quit.

---

## 🕸️ 3. GraphRAG Local (Neo4j + Ollama)

An advanced, offline Graph-based RAG pipeline that builds structured entity-relationship graphs from text, stored in Neo4j, to answer complex multi-hop queries.

### Setup & Run
1. **Run Neo4j**:
   Ensure a local Neo4j database instance is running (port `7687`).
2. **Pull the Ollama model**:
   ```bash
   ollama pull phi3:mini
   ```
3. **Configure Connection**:
   Update Neo4j credentials in `graph_rag/config.py`.
4. **Navigate & Install Dependencies**:
   ```bash
   cd graph_rag
   pip install -r requirements.txt
   ```
5. **Ingest Knowledge File**:
   Extract entities and connections into Neo4j:
   ```bash
   python ingest.py
   ```
6. **Query the Graph**:
   ```bash
   python query.py
   ```

---

## 📂 Repository Structure

```directory
├── chroma/               # Local ChromaDB persistent storage
├── documents/            # PDF documents source directory
├── graph_rag/            # Neo4j GraphRAG implementation codebase
│   ├── data/             # Input data for GraphRAG
│   ├── ingest.py         # Graph construction script
│   ├── query.py          # Graph query interface
│   └── README.md         # Detailed GraphRAG readme
├── rag.py                # Streamlit App (Ollama + In-memory vector store)
└── rag_with_chromadb.py  # CLI App (Gemini + ChromaDB)
```
