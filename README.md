# 🧠 RAG Chatbot — Context-Aware AI with LangChain

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot built with:
- **LangChain** — orchestration, chains, memory
- **Groq API** — Free LLaMA 3 models (llama3-70b-8192)
- **TF-IDF Embeddings** — 100% local, no internet required
- **FAISS** — local vector store with MMR retrieval
- **Streamlit** — interactive chat UI

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get buy Groq API Key 
1. Go to **https://console.groq.com**
2. Sign up (free)
3. Go to **API Keys** → Create API Key
4. Copy your key (starts with `gsk_...`)

### 3. Run the app
```bash
python -m streamlit run app.py
```

### 4. Use the chatbot
1. Enter your **Groq API key** (`gsk_...`) in the sidebar
2. Choose a knowledge base (built-in sample, upload files, or paste text)
3. Click **🚀 Build Knowledge Base**
4. Start chatting!

---

## 📁 Project Structure
rag_chatbot/
├── app.py                    # Streamlit UI
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
├── utils/
│   ├── rag_engine.py         # RAG pipeline (embed → store → retrieve → generate)
│   └── document_loader.py    # Document loaders (sample corpus + file uploads)
├── data/                     # Drop extra .txt / .pdf files here
└── vectorstore/              # FAISS index saved here (auto-created)

## 🏗️ Architecture
User Query
│
▼
Chat History (last N turns)
│
▼
TF-IDF Embedder (local, no download)
│
▼
FAISS Vector Store (MMR Retrieval)
│
▼
Top-K Relevant Chunks + Context
│
▼
Groq API — LLaMA 3 (llama3-70b-8192)
│
▼
Answer + Source Citations

---

## ⚙️ Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| Model | llama3-70b-8192 | Free LLaMA 3 via Groq |
| Temperature | 0.3 | Response creativity (0=deterministic) |
| Top-K | 3 | Number of chunks retrieved per query |
| Memory Window | 5 | Conversation turns to remember |
| Chunk Size | 800 | Characters per document chunk |
| Chunk Overlap | 100 | Overlap between consecutive chunks |

---

## 📚 Sample Knowledge Base Topics

The built-in corpus covers:
- **Artificial Intelligence** — ML, deep learning, NLP, computer vision
- **Large Language Models** — transformers, RAG, prompt engineering, embeddings
- **Python Programming** — frameworks, packaging, language features
- **Vector Databases** — FAISS, Pinecone, HNSW, similarity metrics
- **LangChain Framework** — chains, agents, LCEL, memory
- **Cloud Computing** — AWS/GCP/Azure, serverless, containers

---

## 🧩 Skills Demonstrated

- ✅ **Conversational AI** with persistent context memory
- ✅ **Local TF-IDF Embeddings** — no internet, no API cost
- ✅ **Vector similarity search** with FAISS + MMR
- ✅ **RAG pipeline** — retrieval-augmented generation
- ✅ **LLM integration** via LangChain + Groq API (FREE)
- ✅ **Streamlit deployment** with polished dark UI
- ✅ **File upload support** for .txt and .pdf documents

---

## 📝 Notes

- Groq API is credit card required $5 doller credit
- TF-IDF embeddings are **100% local** — no HuggingFace download needed
- FAISS vector store is in-memory — rebuilds on each session
- For production, replace FAISS with Pinecone/Weaviate and add authentication

