import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

st.markdown("""
<style>
.stApp {
    background: #0D0D0D;
    font-family: 'Roboto', 'Segoe UI', sans-serif;
    color: #E0E0E0;
}

h1, h2, h3 {
    color: #00FFF7 !important;
    text-shadow: 0 0 8px rgba(0,255,247,0.7);
}

.stChatInput input {
    background: rgba(40,40,40,0.85) !important;
    color: #00FFF7 !important;
    border: 1px solid #00FFF7 !important;
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 4px 15px rgba(0,255,247,0.3);
    font-weight: 500;
}

.stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
    background: rgba(60,60,60,0.85) !important;
    border: 1px solid #FF3CFF !important;
    color: #FF3CFF !important;
    border-radius: 18px;
    padding: 16px;
    margin: 10px 0;
    box-shadow: 0 5px 20px rgba(255,60,255,0.3);
}

.stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
    background: rgba(30,30,30,0.9) !important;
    border: 1px solid #00FFF7 !important;
    color: #00FFF7 !important;
    border-radius: 18px;
    padding: 16px;
    margin: 10px 0;
    box-shadow: 0 5px 20px rgba(0,255,247,0.3);
}

.stChatMessage .avatar {
    background: linear-gradient(45deg, #FF3CFF, #00FFF7) !important;
    color: #0D0D0D !important;
    font-weight: bold;
    box-shadow: 0 0 10px rgba(255,60,255,0.6), 0 0 10px rgba(0,255,247,0.6);
}

.stChatMessage p, .stChatMessage div {
    color: inherit !important;
}

.stFileUploader {
    background: rgba(50,50,50,0.85);
    border-radius: 14px;
    padding: 18px;
    border: 1px solid #00FFF7;
    box-shadow: 0 4px 15px rgba(0,255,247,0.3);
    color: #00FFF7;
    font-weight: 500;
}

.stButton>button {
    background: linear-gradient(135deg, #FF3CFF, #00FFF7);
    color: #0D0D0D;
    font-weight: bold;
    border-radius: 14px;
    padding: 10px 20px;
    box-shadow: 0 6px 25px rgba(0,255,247,0.5);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stButton>button:hover {
    transform: scale(1.08);
    box-shadow: 0 8px 30px rgba(0,255,247,0.7);
}

::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-track {
    background: #0D0D0D;
}
::-webkit-scrollbar-thumb {
    background: #FF3CFF;
    border-radius: 5px;
}

.stChatInput input::placeholder {
    color: rgba(0,255,247,0.5) !important;
    font-style: italic;
}

.stChatMessage {
    backdrop-filter: blur(6px);
}
</style>
""", unsafe_allow_html=True)




PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""
PDF_STORAGE_PATH = 'documents/'
EMBEDDING_MODEL = OllamaEmbeddings(model="llama3")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="llama3")


def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})




st.title("🤖 DOC RAG")
st.markdown("#### Your Intelligent Document Assistant")
st.markdown("---")

uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False

)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully!")
    
    user_input = st.chat_input("Enter your question:")
    
    if user_input:
        with st.chat_message("User", avatar="🙍‍♂️"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("Assistant", avatar="🤖"):
            st.write(ai_response)