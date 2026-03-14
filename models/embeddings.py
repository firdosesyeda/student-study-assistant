import google.generativeai as genai
from config.config import GEMINI_API_KEY, EMBEDDING_MODEL

def get_embedding(text: str) -> list:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return []

def get_query_embedding(text: str) -> list:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        return result["embedding"]
    except Exception as e:
        print(f"[Query Embedding Error] {e}")
        return []