import os
from dotenv import load_dotenv

load_dotenv()

# ======================= LOAD ENV =======================
MODEL = os.getenv("MODEL", "llama3.2:3b")
TRAINED_DIR = os.getenv("TRAINED_DIR", "/backend/trained")
#DATABASE_URL = os.getenv("DATABASE_URL")

OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_BASE_URL = f"http://ollama:{OLLAMA_PORT}"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DB = os.getenv("DB_PATH", "/backend/db/history.db")