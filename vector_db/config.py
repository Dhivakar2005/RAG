import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants for project structure
DATA_DIR = "data"
PERSIST_DIRECTORY = "chroma_db"
KNOWLEDGE_FILE = os.path.join(DATA_DIR, "knowledge.txt")

# Constants for chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Model configurations
EMBEDDING_MODEL = "phi3:mini"
CHAT_MODEL = "phi3:mini"
