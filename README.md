# 🧠 RAG Chatbot — Context-Aware AI with LangChain

A production-grade **Retrieval-Augmented Generation (RAG)** chatbot built with:
- **LangChain** — orchestration, chains, memory
- **OpenAI** — embeddings (`sentence-transformers/all-MiniLM-L6-v2 (local)`) + chat (`gpt-3.5-turbo` / `gpt-4o`)
- **FAISS** — local vector store with MMR retrieval
- **Streamlit** — interactive chat UI

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up your API key
```bash
cp .env.example .env
# Edit .env and add your Grok xAI API key
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Use the chatbot
1. Enter your **Grok xAI API key** in the sidebar
2. Choose a knowledge base (built-in sample, upload files, or paste text)
3. Click **🚀 Build Knowledge Base**
4. Start chatting!

---

## 📁 Project Structure

```
rag_chatbot/
├── app.py                    # Streamlit UI
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
├── utils/
│   ├── rag_engine.py         # RAG pipeline (embed → store → retrieve → generate)
│   └── document_loader.py    # Document loaders (sample corpus + file uploads)
├── data/                     # Drop extra .txt / .pdf files here
└── vectorstore/              # FAISS index saved here (auto-created)
```

---

## 🏗️ Architecture

```
User Query
    │
    ▼
ConversationBufferWindowMemory  ←── Chat History (last N turns)
    │
    ▼
ConversationalRetrievalChain
    │
    ├── Condense Question (with history) ──► Standalone Question
    │
    ▼
FAISS Vector Store  (MMR Retrieval)
    │  sentence-transformers/all-MiniLM-L6-v2 (local)
    ▼
Top-K Relevant Chunks
    │
    ▼
ChatOpenAI (GPT-3.5/4o)
    │  System prompt + context + history
    ▼
Answer + Source Citations
```

---

## ⚙️ Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| Model | gpt-3.5-turbo | LLM for generation |
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
- ✅ **Document embedding** using OpenAI's embedding API
- ✅ **Vector similarity search** with FAISS + MMR
- ✅ **RAG pipeline** — retrieval-augmented generation
- ✅ **LLM integration** via LangChain + OpenAI
- ✅ **Streamlit deployment** with polished dark UI
- ✅ **File upload support** for .txt and .pdf documents

---

## 📝 Notes

- The FAISS vector store is in-memory; rebuild on each session or use `save_vectorstore()` to persist
- For production, replace FAISS with Pinecone/Weaviate and add authentication
- Costs ~$0.001–0.005 per query with gpt-3.5-turbo + sentence-transformers/all-MiniLM-L6-v2 (local)
