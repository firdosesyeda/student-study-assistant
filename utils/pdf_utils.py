import pdfplumber
from config.config import CHUNK_SIZE, CHUNK_OVERLAP

def extract_text_from_pdf(pdf_file) -> str:
    try:
        full_text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
        return full_text
    except Exception as e:
        print(f"[PDF Error] {e}")
        return ""

def split_into_chunks(text: str) -> list:
    try:
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks
    except Exception as e:
        print(f"[Chunking Error] {e}")
        return []