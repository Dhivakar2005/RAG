import os
import sys

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import config

def setup_rag_chain():
    """
    Sets up the RAG retrieval chain by loading the persisted Chroma DB,
    configuring the retriever, and setting up the LLM prompting structure.
    """
    # 1. Load embeddings model used during ingestion
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)

    # 2. Connect to the existing vector DB
    if not os.path.exists(config.PERSIST_DIRECTORY):
        print(f"Error: Vector DB not found at '{config.PERSIST_DIRECTORY}'.\nPlease run `python ingest.py` first.")
        sys.exit(1)

    vector_store = Chroma(
        persist_directory=config.PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

    # 3. Create a retriever to get top-k relevant chunks (default: top 3)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 4. Initialize the Chat Model (temperature=0 for factual responses)
    llm = ChatOllama(model=config.CHAT_MODEL, temperature=0)

    # 5. Create a prompt template for generating the answer
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    # 6. Build the LCEL generation chain
    rag_chain = prompt | llm | StrOutputParser()

    return retriever, rag_chain

def interactive_chat():
    """
    CLI loop for interactively predicting text outputs given context from Chroma.
    Supports asking multiple questions iteratively.
    """
    print("Initializing RAG System...")
    try:
        retriever, rag_chain = setup_rag_chain()
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return
    
    print("\n--- RAG System Ready ---")
    print("Ask any question related to the knowledge base.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            user_input = input("\nYou: ")
            
            # Check for termination command
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting chat. Goodbye!")
                break
            
            if not user_input.strip():
                continue

            # Query the model chain for an answer based on RAG context
            print("Thinking...")
            # Retrieve documents
            source_documents = retriever.invoke(user_input)
            context_text = "\n\n".join(doc.page_content for doc in source_documents)
            
            # Generate the answer using LCEL
            answer = rag_chain.invoke({
                "context": context_text,
                "question": user_input
            })

            print(f"\nAI: {answer}")
            print("\nSources:")
            # Display retrieved sources used to generate the context window
            if source_documents:
                for i, doc in enumerate(source_documents, 1):
                    # Preview the first 100 characters of the source document
                    content_preview = doc.page_content.replace('\n', ' ')[:100]
                    print(f"  {i}. {content_preview}...")
            else:
                print("  No sources retrieved.")

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nExiting chat. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    interactive_chat()
