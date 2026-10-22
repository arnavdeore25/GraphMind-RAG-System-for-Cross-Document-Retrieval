import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = "gemini-flash-lite-latest"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SOURCE_DIR = "source"
CHROMA_DIR = "db/chroma_db"
COLLECTION = "graphmind"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphmind123")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5
GRAPH_HOPS = 2
