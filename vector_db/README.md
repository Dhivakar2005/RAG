# RAG System in Python

A complete Retrieval-Augmented Generation (RAG) system built in Python using LangChain, ChromaDB, and local Ollama models.

## Architecture Pipeline
- **Document Loading:** Reads text files into documents
- **Chunking:** Splits documents (chunk_size=500, overlap=100)
- **Embeddings:** Vectorizes text strings using local `phi3:mini` via Ollama
- **Vector Database:** Stores generated vector bindings in local ChromaDB setup
- **Retrieval:** Extracts most relevant documents based on semantics
- **LLM Answer Generation:** Answers questions utilizing context through local `phi3:mini` with `temperature=0`

## Features
- Modular code structure separating ingestion entirely from querying.
- Context retention and sources validation per answer.
- Local vector database persistence.
- Infinite interactive CLI loop.

## Project Structure
```
rag_project/
├── data/
│   └── knowledge.txt     # Example knowledge base content
├── config.py             # Shared configuration variables
├── ingest.py             # Script to process text data and populate vector storage
├── query.py              # Script containing the interactive CLI chat loop
├── requirements.txt      # Python dependencies
└── README.md             # The current README 
```

## Setup Instructions

### 1. Set up a Virtual Environment
```bash
python -m venv venv
# On Windows use:
venv\Scripts\activate
# On Linux/MacOS use:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Provide Environment Configuration
Ensure you have [Ollama](https://ollama.com/) installed and running locally, and pull the required model:
```bash
ollama pull phi3:mini
```
(Optional) You can still set up a `.env` file if other configurations are added.

### 4. Build the Vector Database
Execute the ingest command to ingest raw data and construct the required chroma local db instance:
```bash
python ingest.py
```

### 5. Chat Interface
Run the query script to initiate the RAG evaluation:
```bash
python query.py
```
> Type `exit` or `quit` to cleanly exit the chat session.
