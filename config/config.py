import os

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
SERPER_API_KEY  = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")

GROQ_MODEL      = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "models/embedding-001"

CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
TOP_K_RESULTS   = 3

SERPER_URL         = "https://google.serper.dev/search"
MAX_SEARCH_RESULTS = 3
