import os
from dotenv import load_dotenv

load_dotenv()

# ======================= LOAD ENV =======================
TRAINED_DIR = os.getenv("TRAINED_DIR", "/backend/trained")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_BASE_URL = f"http://ollama:{OLLAMA_PORT}"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DB = os.getenv("DB_PATH", "/backend/db/history.db")