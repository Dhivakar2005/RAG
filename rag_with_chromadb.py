import os
import chromadb
import google.generativeai as genai
from pypdf import PdfReader

genai.configure(api_key="your_gemini_api_key_here")

CHROMA_DIR = "./chroma"
COLLECTION_NAME = "RAG_Collections"
EMBED_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"

PDF_PATH = "documents/attention-is-all-you-need.pdf"  


chroma_client = chromadb.Client(
    settings=chromadb.config.Settings(
        persist_directory=CHROMA_DIR
    )
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)

def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunks


def embed_documents(texts):
    embeddings = []
    for text in texts:
        emb = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_document"
        )["embedding"]
        embeddings.append(emb)
    return embeddings

def embed_query(text):
    return genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query"
    )["embedding"]

pdf_text = read_pdf(PDF_PATH)
chunks = chunk_text(pdf_text)

ids = [f"pdf_chunk_{i}" for i in range(len(chunks))]
embeddings = embed_documents(chunks)

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings,
    metadatas=[{"source": PDF_PATH}] * len(chunks)
)

print(f"Inserted {len(chunks)} chunks from PDF.")

def rag_query(question, k=3):
    query_embedding = embed_query(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""

    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)

    return response.text

if __name__ == "__main__":
    print("Ask questions about the document.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        answer = rag_query(question)

        print("\nAnswer:")
        print("-" * 50)
        print(answer)
