import streamlit as st
import time
from models.llm        import get_gemini_response
from models.embeddings import get_embedding, get_query_embedding
from utils.pdf_utils      import extract_text_from_pdf, split_into_chunks
from utils.vector_store   import build_vector_store, retrieve_relevant_chunks
from utils.web_search     import search_web
from utils.prompt_builder import build_rag_prompt, build_web_prompt

st.set_page_config(page_title="📚 Student Study Assistant", page_icon="📚", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    * { font-family: 'Poppins', sans-serif; }

    /* Main background - dark purple gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e, #2d2b55) !important;
        border-right: 1px solid #4a4080;
    }
    [data-testid="stSidebar"] * { color: #e0d7ff !important; }
    [data-testid="stSidebar"] .stFileUploader {
        background: rgba(255,255,255,0.05);
        border: 1px dashed #7c6fcf;
        border-radius: 12px;
        padding: 10px;
    }
    [data-testid="stSidebar"] .stRadio label { color: #c4b5fd !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #a78bfa !important; }

    /* Title area */
    .title-container {
        text-align: center;
        padding: 30px 0 10px 0;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 8px;
    }
    .emoji-logo {
        font-size: 4rem;
        display: block;
        margin-bottom: 10px;
    }

    /* Chat bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        padding: 14px 18px;
        border-radius: 20px 20px 4px 20px;
        margin: 10px 0 4px auto;
        max-width: 78%;
        word-wrap: break-word;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
        font-size: 0.95rem;
    }
    .bot-bubble {
        background: linear-gradient(135deg, #1e1b4b, #2e2b6b);
        color: #e2e8f0;
        padding: 14px 18px;
        border-radius: 20px 20px 20px 4px;
        margin: 10px 0 4px 0;
        max-width: 82%;
        border: 1px solid #4338ca;
        word-wrap: break-word;
        box-shadow: 0 4px 15px rgba(67, 56, 202, 0.3);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .source-badge {
        font-size: 0.72rem;
        color: #7c6fcf;
        margin-bottom: 12px;
        padding-left: 4px;
    }

    /* Empty chat message */
    .empty-chat {
        text-align: center;
        color: #4a4580;
        margin-top: 60px;
        font-size: 1rem;
    }
    .empty-chat span {
        font-size: 3rem;
        display: block;
        margin-bottom: 10px;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.07) !important;
        border: 2px solid #4338ca !important;
        border-radius: 14px !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 12px 16px !important;
    }
    .stTextInput > div > div > input::placeholder { color: #6366f1 !important; }
    .stTextInput > div > div > input:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.2) !important;
    }

    /* Ask button */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border-radius: 14px !important;
        border: none !important;
        width: 100% !important;
        height: 50px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6d28d9, #4338ca) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6) !important;
        transform: translateY(-1px);
    }

    /* Divider */
    hr { border-color: #2d2b55 !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #1a1a3e; }
    ::-webkit-scrollbar-thumb { background: #4338ca; border-radius: 3px; }

    /* Warning/Info/Success boxes */
    .stWarning, .stInfo, .stSuccess, .stError {
        border-radius: 10px !important;
    }

    /* Progress bar */
    .stProgress > div > div { background: linear-gradient(90deg, #7c3aed, #60a5fa) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "vector_store"  not in st.session_state: st.session_state.vector_store  = []
if "pdf_processed" not in st.session_state: st.session_state.pdf_processed = False
if "pdf_name"      not in st.session_state: st.session_state.pdf_name      = ""

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class='title-container'>
    <span class='emoji-logo'>📚</span>
    <p class='main-title'>Student Study Assistant</p>
    <p class='subtitle'>✨ Upload your notes → Ask anything → Get smart answers!</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("### 📄 Upload Study Material")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_uploader")

    if uploaded_pdf and uploaded_pdf.name != st.session_state.pdf_name:
        with st.spinner("📖 Reading and indexing your PDF..."):
            try:
                raw_text = extract_text_from_pdf(uploaded_pdf)
                if not raw_text.strip():
                    st.error("❌ Could not extract text. Try another file.")
                else:
                    chunks = split_into_chunks(raw_text)
                    embeddings = []
                    progress_bar = st.progress(0)
                    for i, chunk in enumerate(chunks):
                        embeddings.append(get_embedding(chunk))
                        progress_bar.progress((i + 1) / len(chunks))
                    st.session_state.vector_store  = build_vector_store(chunks, embeddings)
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name      = uploaded_pdf.name
                    st.success(f"✅ '{uploaded_pdf.name}' indexed!\n({len(chunks)} chunks ready)")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if st.session_state.pdf_processed:
        st.info(f"📗 Loaded: **{st.session_state.pdf_name}**")
    else:
        st.warning("⚠️ No PDF uploaded.\nWeb search will be used.")

    st.markdown("---")
    st.markdown("### ✍️ Response Mode")
    response_mode = st.radio(
        "",
        options=["⚡ Concise", "📖 Detailed"],
        index=0
    )
    response_mode = "Concise" if "Concise" in response_mode else "Detailed"

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("<p style='color:#4a4580;font-size:0.75rem;text-align:center;'>Built with ❤️ using<br>Streamlit + Groq + Gemini</p>", unsafe_allow_html=True)

# ── Chat Display ────────────────────────────────────────────────
chat_box = st.container()
with chat_box:
    if not st.session_state.chat_history:
        st.markdown("""
        <div class='empty-chat'>
            <span>💬</span>
            Ask your first question below!<br>
            <small style='color:#3d3b6e;'>Upload a PDF or just ask anything...</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-bubble'>🧑‍🎓 {msg['text']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-bubble'>🤖 {msg['text']}</div>", unsafe_allow_html=True)
                if msg.get("source"):
                    st.markdown(f"<div class='source-badge'>📌 Source: {msg['source']}</div>", unsafe_allow_html=True)

# ── Input ───────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    user_question = st.text_input(
        "",
        placeholder="💭 Ask me anything...",
        label_visibility="collapsed",
        key="question_input"
    )
with col2:
    ask_clicked = st.button("Ask ➤")

# ── Main Logic ──────────────────────────────────────────────────
if ask_clicked and user_question.strip():
    question = user_question.strip()
    st.session_state.chat_history.append({"role": "user", "text": question})

    with st.spinner("🔮 Thinking..."):
        answer = ""
        source = ""

        # Try RAG first
        if st.session_state.pdf_processed and st.session_state.vector_store:
            try:
                query_emb = get_query_embedding(question)
                if query_emb:
                    relevant_chunks = retrieve_relevant_chunks(query_emb, st.session_state.vector_store)
                    if relevant_chunks:
                        prompt = build_rag_prompt(question, relevant_chunks, response_mode)
                        answer = get_gemini_response(prompt)
                        source = f"📄 {st.session_state.pdf_name}"
            except Exception as e:
                st.warning(f"RAG failed: {e}. Trying web search...")

        # Fallback to web search
        if not answer or "not covered in the uploaded document" in answer.lower():
            try:
                web_results = search_web(question)
                prompt      = build_web_prompt(question, web_results, response_mode)
                answer      = get_gemini_response(prompt)
                source      = "🌐 Live Web Search"
            except Exception as e:
                answer = f"❌ Sorry, I couldn't get an answer: {str(e)}"
                source = "Error"

    # Typing animation effect
    st.session_state.chat_history.append({"role": "bot", "text": answer, "source": source})
    st.rerun()

elif ask_clicked:
    st.warning("⚠️ Please type a question first!")

# ── Footer ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
import streamlit as st
import time
from models.llm        import get_gemini_response
from models.embeddings import get_embedding, get_query_embedding
from utils.pdf_utils      import extract_text_from_pdf, split_into_chunks
from utils.vector_store   import build_vector_store, retrieve_relevant_chunks
from utils.web_search     import search_web
from utils.prompt_builder import build_rag_prompt, build_web_prompt

st.set_page_config(page_title="📚 Student Study Assistant", page_icon="📚", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    * { font-family: 'Poppins', sans-serif; }

    /* Main background - dark purple gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e, #2d2b55) !important;
        border-right: 1px solid #4a4080;
    }
    [data-testid="stSidebar"] * { color: #e0d7ff !important; }
    [data-testid="stSidebar"] .stFileUploader {
        background: rgba(255,255,255,0.05);
        border: 1px dashed #7c6fcf;
        border-radius: 12px;
        padding: 10px;
    }
    [data-testid="stSidebar"] .stRadio label { color: #c4b5fd !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #a78bfa !important; }

    /* Title area */
    .title-container {
        text-align: center;
        padding: 30px 0 10px 0;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 8px;
    }
    .emoji-logo {
        font-size: 4rem;
        display: block;
        margin-bottom: 10px;
    }

    /* Chat bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        padding: 14px 18px;
        border-radius: 20px 20px 4px 20px;
        margin: 10px 0 4px auto;
        max-width: 78%;
        word-wrap: break-word;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
        font-size: 0.95rem;
    }
    .bot-bubble {
        background: linear-gradient(135deg, #1e1b4b, #2e2b6b);
        color: #e2e8f0;
        padding: 14px 18px;
        border-radius: 20px 20px 20px 4px;
        margin: 10px 0 4px 0;
        max-width: 82%;
        border: 1px solid #4338ca;
        word-wrap: break-word;
        box-shadow: 0 4px 15px rgba(67, 56, 202, 0.3);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .source-badge {
        font-size: 0.72rem;
        color: #7c6fcf;
        margin-bottom: 12px;
        padding-left: 4px;
    }

    /* Empty chat message */
    .empty-chat {
        text-align: center;
        color: #4a4580;
        margin-top: 60px;
        font-size: 1rem;
    }
    .empty-chat span {
        font-size: 3rem;
        display: block;
        margin-bottom: 10px;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.07) !important;
        border: 2px solid #4338ca !important;
        border-radius: 14px !important;
        color: white !important;
        font-size: 1rem !important;
        padding: 12px 16px !important;
    }
    .stTextInput > div > div > input::placeholder { color: #6366f1 !important; }
    .stTextInput > div > div > input:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.2) !important;
    }

    /* Ask button */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important;
        border-radius: 14px !important;
        border: none !important;
        width: 100% !important;
        height: 50px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6d28d9, #4338ca) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6) !important;
        transform: translateY(-1px);
    }

    /* Divider */
    hr { border-color: #2d2b55 !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #1a1a3e; }
    ::-webkit-scrollbar-thumb { background: #4338ca; border-radius: 3px; }

    /* Warning/Info/Success boxes */
    .stWarning, .stInfo, .stSuccess, .stError {
        border-radius: 10px !important;
    }

    /* Progress bar */
    .stProgress > div > div { background: linear-gradient(90deg, #7c3aed, #60a5fa) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "vector_store"  not in st.session_state: st.session_state.vector_store  = []
if "pdf_processed" not in st.session_state: st.session_state.pdf_processed = False
if "pdf_name"      not in st.session_state: st.session_state.pdf_name      = ""

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class='title-container'>
    <span class='emoji-logo'>📚</span>
    <p class='main-title'>Student Study Assistant</p>
    <p class='subtitle'>✨ Upload your notes → Ask anything → Get smart answers!</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("### 📄 Upload Study Material")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_pdf and uploaded_pdf.name != st.session_state.pdf_name:
        with st.spinner("📖 Reading and indexing your PDF..."):
            try:
                raw_text = extract_text_from_pdf(uploaded_pdf)
                if not raw_text.strip():
                    st.error("❌ Could not extract text. Try another file.")
                else:
                    chunks = split_into_chunks(raw_text)
                    embeddings = []
                    progress_bar = st.progress(0)
                    for i, chunk in enumerate(chunks):
                        embeddings.append(get_embedding(chunk))
                        progress_bar.progress((i + 1) / len(chunks))
                    st.session_state.vector_store  = build_vector_store(chunks, embeddings)
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_name      = uploaded_pdf.name
                    st.success(f"✅ '{uploaded_pdf.name}' indexed!\n({len(chunks)} chunks ready)")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if st.session_state.pdf_processed:
        st.info(f"📗 Loaded: **{st.session_state.pdf_name}**")
    else:
        st.warning("⚠️ No PDF uploaded.\nWeb search will be used.")

    st.markdown("---")
    st.markdown("### ✍️ Response Mode")
    response_mode = st.radio(
        "",
        options=["⚡ Concise", "📖 Detailed"],
        index=0
    )
    response_mode = "Concise" if "Concise" in response_mode else "Detailed"

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("<p style='color:#4a4580;font-size:0.75rem;text-align:center;'>Built with ❤️ using<br>Streamlit + Groq + Gemini</p>", unsafe_allow_html=True)

# ── Chat Display ────────────────────────────────────────────────
chat_box = st.container()
with chat_box:
    if not st.session_state.chat_history:
        st.markdown("""
        <div class='empty-chat'>
            <span>💬</span>
            Ask your first question below!<br>
            <small style='color:#3d3b6e;'>Upload a PDF or just ask anything...</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-bubble'>🧑‍🎓 {msg['text']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-bubble'>🤖 {msg['text']}</div>", unsafe_allow_html=True)
                if msg.get("source"):
                    st.markdown(f"<div class='source-badge'>📌 Source: {msg['source']}</div>", unsafe_allow_html=True)

# ── Input ───────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([5, 1])
with col1:
    user_question = st.text_input(
        "",
        placeholder="💭 Ask me anything...",
        label_visibility="collapsed",
        key="question_input"
    )
with col2:
    ask_clicked = st.button("Ask ➤")

# ── Main Logic ──────────────────────────────────────────────────
if ask_clicked and user_question.strip():
    question = user_question.strip()
    st.session_state.chat_history.append({"role": "user", "text": question})

    with st.spinner("🔮 Thinking..."):
        answer = ""
        source = ""

        # Try RAG first
        if st.session_state.pdf_processed and st.session_state.vector_store:
            try:
                query_emb = get_query_embedding(question)
                if query_emb:
                    relevant_chunks = retrieve_relevant_chunks(query_emb, st.session_state.vector_store)
                    if relevant_chunks:
                        prompt = build_rag_prompt(question, relevant_chunks, response_mode)
                        answer = get_gemini_response(prompt)
                        source = f"📄 {st.session_state.pdf_name}"
            except Exception as e:
                st.warning(f"RAG failed: {e}. Trying web search...")

        # Fallback to web search
        if not answer or "not covered in the uploaded document" in answer.lower():
            try:
                web_results = search_web(question)
                prompt      = build_web_prompt(question, web_results, response_mode)
                answer      = get_gemini_response(prompt)
                source      = "🌐 Live Web Search"
            except Exception as e:
                answer = f"❌ Sorry, I couldn't get an answer: {str(e)}"
                source = "Error"

    # Typing animation effect
    st.session_state.chat_history.append({"role": "bot", "text": answer, "source": source})
    st.rerun()

elif ask_clicked:
    st.warning("⚠️ Please type a question first!")

# ── Footer ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
