def build_rag_prompt(question: str, context_chunks: list, mode: str) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    if mode == "Concise":
        instruction = "Answer in 2-3 short sentences only. Be direct."
    else:
        instruction = "Answer in detail with examples. Use bullet points if needed."

    return f"""You are a helpful study assistant for students.
Use ONLY the context below to answer. If the answer is not in context, say:
"This topic is not covered in the uploaded document."

{instruction}

─── Context from uploaded notes ───
{context}
────────────────────────────────────

Student's Question: {question}

Answer:"""


def build_web_prompt(question: str, web_results: str, mode: str) -> str:
    if mode == "Concise":
        instruction = "Summarize in 2-3 short sentences only."
    else:
        instruction = "Give a detailed, well-structured answer based on search results."

    return f"""You are a helpful study assistant for students.
The uploaded notes didn't have the answer, so here are live web search results.

{instruction}

─── Web Search Results ───
{web_results}
──────────────────────────

Student's Question: {question}

Answer:"""