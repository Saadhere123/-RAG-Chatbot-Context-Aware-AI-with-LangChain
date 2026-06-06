import streamlit as st
import os
import time
from utils.rag_engine import RAGEngine
from utils.document_loader import load_documents_from_folder, load_sample_corpus
from langchain_core.documents import Document

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e3a;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #7c5cbf;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #1a0a2e 0%, #0f1a2e 50%, #0a1a1a 100%);
    border: 1px solid #2a1a4e;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(124,92,191,0.08) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(0,200,180,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    background: linear-gradient(90deg, #9b72cf, #00c8b4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.main-header p {
    color: #8888aa;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    margin: 0.4rem 0 0;
}

/* Chat messages */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding: 0.5rem 0;
}

.msg-user {
    background: linear-gradient(135deg, #1e1440, #1a1a2e);
    border: 1px solid #3a2a6e;
    border-radius: 16px 16px 4px 16px;
    padding: 1rem 1.4rem;
    margin-left: 10%;
    color: #d0c8f0;
    font-size: 0.95rem;
    line-height: 1.6;
}

.msg-bot {
    background: linear-gradient(135deg, #0a1a2e, #0a2020);
    border: 1px solid #1a3a4e;
    border-radius: 16px 16px 16px 4px;
    padding: 1rem 1.4rem;
    margin-right: 10%;
    color: #c8e8e8;
    font-size: 0.95rem;
    line-height: 1.7;
}

.msg-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.msg-user .msg-label { color: #9b72cf; }
.msg-bot  .msg-label { color: #00c8b4; }

/* Sources */
.sources-box {
    background: #0a0a18;
    border: 1px solid #1e1e3a;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-top: 0.8rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #6688aa;
}
.sources-box strong { color: #4488cc; }

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #0f1a0f;
    border: 1px solid #1a3a1a;
    border-radius: 20px;
    padding: 4px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #44cc88;
}
.status-dot {
    width: 6px; height: 6px;
    background: #44cc88;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* Input */
.stTextInput input, .stTextArea textarea {
    background: #0f0f1a !important;
    border: 1px solid #2a2a4e !important;
    color: #e8e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #7c5cbf !important;
    box-shadow: 0 0 0 2px rgba(124,92,191,0.15) !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #3d1f7a, #1f4a7a) !important;
    border: 1px solid #5a3aaa !important;
    color: #e8e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background: linear-gradient(135deg, #5a3aaa, #2a6aaa) !important;
    border-color: #7c5cbf !important;
    transform: translateY(-1px) !important;
}

/* Metrics */
.metric-card {
    background: #0f0f1a;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #9b72cf;
    font-family: 'Space Mono', monospace;
}
.metric-card .label {
    font-size: 0.7rem;
    color: #6666aa;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* File uploader */
.stFileUploader {
    background: #0f0f1a !important;
    border: 1px dashed #2a2a5e !important;
    border-radius: 12px !important;
}

/* Divider */
hr { border-color: #1e1e3a !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a4e; border-radius: 3px; }

/* Chat input area fix */
.stChatInput { background: #0f0f1a !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 RAG Chatbot")
    st.markdown("---")

    # API Key
    st.markdown("### 🔑 Configuration")
    api_key = st.text_input(
        "Grok API Key (xAI)",
        type="password",
        placeholder="xai-...",
        help="Your Grok xAI API key — get it from console.x.ai"
    )

    st.markdown("---")
    st.markdown("### 📚 Knowledge Base")

    kb_option = st.radio(
        "Source",
        ["Sample Corpus (Built-in)", "Upload Documents", "Enter Text"],
        index=0
    )

    uploaded_files = None
    custom_text = None

    if kb_option == "Upload Documents":
        uploaded_files = st.file_uploader(
            "Upload .txt or .pdf files",
            type=["txt", "pdf"],
            accept_multiple_files=True,
        )
    elif kb_option == "Enter Text":
        custom_text = st.text_area(
            "Paste your knowledge base text",
            height=200,
            placeholder="Paste any text here — articles, docs, notes..."
        )

    # Model settings
    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    model_name = st.selectbox(
       "Groq / LLaMA Model",
       ["llama3-70b-8192", "llama-3.1-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
        help="Free LLaMA models via Groq API"
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    top_k = st.slider("Top-K Retrieval", 1, 8, 3)
    memory_window = st.slider("Memory Window (turns)", 2, 10, 5)

    st.markdown("---")

    build_btn = st.button("🚀 Build Knowledge Base", use_container_width=True)

    if build_btn:
        if not api_key:
            st.error("Please enter your Grok API key.")
        else:
            with st.spinner("Loading documents & building vector store…"):
                try:
                    # Gather documents
                    if kb_option == "Sample Corpus (Built-in)":
                        docs, n_docs = load_sample_corpus()
                    elif kb_option == "Upload Documents":
                        if not uploaded_files:
                            st.warning("Please upload files first.")
                            st.stop()
                        docs, n_docs = load_documents_from_folder(uploaded_files)
                    elif kb_option == "Enter Text":
                        if not custom_text or not custom_text.strip():
                            st.warning("Please enter text first.")
                            st.stop()
                        docs = [Document(page_content=custom_text, metadata={"source": "user_input"})]
                        n_docs = 1
                    else:
                        st.warning("No documents provided.")
                        st.stop()

                    engine = RAGEngine(
                        api_key=api_key,
                        model_name=model_name,
                        temperature=temperature,
                        top_k=top_k,
                        memory_window=memory_window,
                    )
                    n_chunks = engine.build_vectorstore(docs)

                    st.session_state.rag_engine = engine
                    st.session_state.vectorstore_ready = True
                    st.session_state.doc_count = n_docs
                    st.session_state.chunk_count = n_chunks
                    st.session_state.messages = []
                    st.success(f"✅ Ready! {n_docs} docs → {n_chunks} chunks")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Stats
    if st.session_state.vectorstore_ready:
        st.markdown("---")
        st.markdown("### 📊 Stats")
        c1, c2 = st.columns(2)
        c1.markdown(f"""
        <div class='metric-card'>
            <div class='value'>{st.session_state.doc_count}</div>
            <div class='label'>Docs</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""
        <div class='metric-card'>
            <div class='value'>{st.session_state.chunk_count}</div>
            <div class='label'>Chunks</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"<br><div class='metric-card'><div class='value'>{len(st.session_state.messages)//2}</div><div class='label'>Turns</div></div>", unsafe_allow_html=True)

    if st.session_state.messages:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.rag_engine:
                st.session_state.rag_engine.clear_memory()
            st.rerun()

# ─── Main Area ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
  <h1>🧠 RAG Chatbot</h1>
  <p>Retrieval-Augmented Generation · LangChain · Grok xAI · LLaMA · FAISS Vector Search</p>
</div>
""", unsafe_allow_html=True)

# Status row
col_s1, col_s2, col_s3 = st.columns([2, 2, 4])
with col_s1:
    if st.session_state.vectorstore_ready:
        st.markdown("<div class='status-badge'><div class='status-dot'></div>Vector Store Active</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-badge' style='border-color:#3a1a1a;color:#cc4444;'><div class='status-dot' style='background:#cc4444;'></div>Not Initialized</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Chat History ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class='msg-user'>
            <div class='msg-label'>👤 You</div>
            {msg['content']}
        </div>""", unsafe_allow_html=True)
    else:
        sources_html = ""
        if msg.get("sources"):
            src_list = "<br>".join(f"• {s}" for s in msg["sources"])
            sources_html = f"<div class='sources-box'><strong>📎 Sources:</strong><br>{src_list}</div>"
        st.markdown(f"""
        <div class='msg-bot'>
            <div class='msg-label'>🤖 Assistant</div>
            {msg['content']}
            {sources_html}
        </div>""", unsafe_allow_html=True)

# ─── Chat Input ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if not st.session_state.vectorstore_ready:
    st.info("👈 Configure your API key and build a knowledge base in the sidebar to start chatting.")
else:
    user_input = st.chat_input("Ask anything about your knowledge base…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("🔍 Retrieving & reasoning…"):
            try:
                answer, sources = st.session_state.rag_engine.query(user_input)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {str(e)}",
                    "sources": [],
                })
        st.rerun()
