import math
from config.config import TOP_K_RESULTS

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    try:
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
        magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        return dot_product / (magnitude_a * magnitude_b)
    except Exception as e:
        print(f"[Cosine Similarity Error] {e}")
        return 0.0

def build_vector_store(chunks: list, embeddings: list) -> list:
    try:
        store = []
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:
                store.append({"chunk": chunk, "embedding": embedding})
        return store
    except Exception as e:
        print(f"[Vector Store Build Error] {e}")
        return []

def retrieve_relevant_chunks(query_embedding: list, vector_store: list) -> list:
    try:
        if not vector_store or not query_embedding:
            return []
        scored = []
        for item in vector_store:
            score = cosine_similarity(query_embedding, item["embedding"])
            scored.append((score, item["chunk"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:TOP_K_RESULTS]]
    except Exception as e:
        print(f"[Retrieval Error] {e}")
        return []