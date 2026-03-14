import requests
from config.config import SERPER_API_KEY, SERPER_URL, MAX_SEARCH_RESULTS

def search_web(query: str) -> str:
    try:
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {"q": query, "num": MAX_SEARCH_RESULTS}
        response = requests.post(SERPER_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()

        results = response.json().get("organic", [])
        if not results:
            return "No web results found."

        parts = []
        for i, r in enumerate(results[:MAX_SEARCH_RESULTS], 1):
            parts.append(f"{i}. **{r.get('title','')}**\n{r.get('snippet','')}\n🔗 {r.get('link','')}")
        return "\n\n".join(parts)

    except requests.exceptions.Timeout:
        return "❌ Web search timed out."
    except requests.exceptions.ConnectionError:
        return "❌ No internet connection."
    except Exception as e:
        return f"❌ Web Search Error: {str(e)}"