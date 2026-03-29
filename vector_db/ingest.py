import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import config

def main():
    """
    Ingest pipeline to process a text document, generate embeddings,
    and persist vectors to a local Chroma database.
    """
    print(f"Loading document from {config.KNOWLEDGE_FILE}...")
    
    # Ensure the knowledge file exists
    if not os.path.exists(config.KNOWLEDGE_FILE):
        print(f"Error: Knowledge file not found at {config.KNOWLEDGE_FILE}")
        return

    # 1. Load text documents
    try:
        loader = TextLoader(config.KNOWLEDGE_FILE, encoding="utf-8")
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s).")
    except Exception as e:
        print(f"Error loading document: {e}")
        return

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks "
          f"(chunk_size={config.CHUNK_SIZE}, chunk_overlap={config.CHUNK_OVERLAP}).")

    # 3. Generate embeddings
    # We use OllamaEmbeddings configured with phi3:mini
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    print(f"Generated embeddings configuration using model: {config.EMBEDDING_MODEL}")
    
    # Check if a database already exists; if so, clear it for a fresh ingest
    if os.path.exists(config.PERSIST_DIRECTORY):
        print(f"Found existing database at '{config.PERSIST_DIRECTORY}'. Clearing it to start fresh...")
        shutil.rmtree(config.PERSIST_DIRECTORY)

    # 4. Store vectors in ChromaDB natively (with local persistence)
    print(f"Persisting vector records to '{config.PERSIST_DIRECTORY}'...")
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=config.PERSIST_DIRECTORY
        )
        print("Ingestion complete. Vector database is ready.")
    except Exception as e:
        print(f"Error creating vector database: {e}")

if __name__ == "__main__":
    main()
