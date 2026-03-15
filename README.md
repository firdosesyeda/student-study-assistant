#  Student Study Assistant

An AI-powered chatbot that reads your study PDFs and answers questions intelligently.
Built with **Streamlit + Groq (LLaMA 3.3) + Gemini Embeddings + Serper Web Search**.

---

##  Use Case

Students often struggle to find answers quickly from lengthy notes and textbooks.
This chatbot solves that by:
- Reading your uploaded PDF notes
- Answering questions directly from your material
- Searching the web when PDF has no answer

---

##  Features

| Feature | Description |
|---|---|
| 📄 PDF Upload + RAG | Upload any PDF — chatbot reads and answers from it |
| 🧠 Smart Retrieval | Finds most relevant chunks using Cosine Similarity |
| 🌐 Live Web Search | Auto fallback to Google search via Serper API |
| ✍️ Response Modes | Concise (short) or Detailed (full explanation) |
| 💬 Chat History | Full conversation saved in session |
| 🎨 Dark UI | Beautiful dark purple gradient theme |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Streamlit | Frontend UI |
| Groq API (LLaMA 3.3 70B) | LLM for generating answers |
| Google Gemini | Text embeddings for RAG |
| Serper API | Live Google web search |
| pdfplumber | PDF text extraction |
| Cosine Similarity | Vector search (no external DB needed) |

---

## 📁 Project Structure
```
student_study_assistant/
├── config/
│   ├── __init__.py
│   └── config.py           ← API keys & settings
│
├── models/
│   ├── __init__.py
│   ├── llm.py              ← Groq LLM wrapper
│   └── embeddings.py       ← Gemini embedding model
│
├── utils/
│   ├── __init__.py
│   ├── pdf_utils.py        ← PDF reading & chunking
│   ├── vector_store.py     ← In-memory RAG vector store
│   ├── web_search.py       ← Serper live web search
│   └── prompt_builder.py   ← Builds prompts for LLM
│
├── app.py                  ← Main Streamlit app
├── requirements.txt
└── README.md
```

---

## Setup Guide

### Step 1 — Get Free API Keys

| API | Link | Free Limit |
|---|---|---|
| Groq API | https://console.groq.com/keys | Free tier available |
| Gemini API | https://aistudio.google.com/app/apikey | Free tier available |
| Serper API | https://serper.dev | 2500 searches/month free |

### Step 2 — Clone Repository
```bash
git clone https://github.com/firdoseyeda/student-study-assistant.git
cd student-study-assistant
```

### Step 3 — Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 4 — Install Packages
```bash
pip install -r requirements.txt
```

### Step 5 — Add API Keys

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY    = "your-groq-key-here"
GEMINI_API_KEY  = "your-gemini-key-here"
SERPER_API_KEY  = "your-serper-key-here"
```

### Step 6 — Run App
```bash
streamlit run app.py
```

Open browser at `http://localhost:8501` 🎉

---

##  How RAG Works
```
PDF Upload
    ↓
Extract Text (pdfplumber)
    ↓
Split into Chunks (500 chars each)
    ↓
Embed each Chunk → Vectors (Gemini)
    ↓
Store in Memory

── When you ask a question ──

Question → Embed Question
    ↓
Compare with stored chunks (Cosine Similarity)
    ↓
Pick Top 3 most relevant chunks
    ↓
Send chunks + question to Groq LLaMA
    ↓
Get Smart Answer! ✅
```

---

## Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect GitHub repo
4. Set `app.py` as main file
5. Add secrets in **Settings → Secrets**:
```toml
GROQ_API_KEY    = "your-groq-key"
GEMINI_API_KEY  = "your-gemini-key"
SERPER_API_KEY  = "your-serper-key"
```
6. Click **Deploy!**

---

## 📦 Requirements
```
streamlit>=1.32.0
google-generativeai>=0.5.0
pdfplumber>=0.10.0
requests>=2.31.0
groq>=0.4.0
```

---

## ⚙️ Configuration

Edit `config/config.py` to change:

| Setting | Default | Description |
|---|---|---|
| CHUNK_SIZE | 500 | Characters per chunk |
| CHUNK_OVERLAP | 50 | Overlap between chunks |
| TOP_K_RESULTS | 3 | Chunks to retrieve |
| MAX_SEARCH_RESULTS | 3 | Web results to use |

---

## 🙏 Acknowledgements

Built for **NeoStats AI Engineer Challenge**

> "Upload your notes → Ask anything → Get smart answers!"

Built with ❤️ using Streamlit + Groq + Google Gemini
