from __future__ import annotations
import os
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

GROK_BASE_URL = "https://api.groq.com/openai/v1"


class TfidfEmbedder(Embeddings):
    """100% local embeddings — no internet, no download needed."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=384)
        self._all_texts = []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._all_texts = texts
        vectors = self.vectorizer.fit_transform(texts).toarray().astype(float)
        return normalize(vectors).tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self.vectorizer.transform([text]).toarray().astype(float)
        return normalize(vec)[0].tolist()


class RAGEngine:

    def __init__(
        self,
        api_key: str,
        model_name: str = "grok-3-mini",
        temperature: float = 0.3,
        top_k: int = 3,
        memory_window: int = 5,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chat_history: list = []
        self.memory_window = memory_window

        self.embeddings = TfidfEmbedder()

        self.llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=GROK_BASE_URL,
            temperature=temperature,
            streaming=False,
        )

        self.vectorstore = None
        self.retriever = None

    def build_vectorstore(self, docs: List[Document]) -> int:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.top_k, "fetch_k": self.top_k * 3},
        )
        return len(chunks)

    def query(self, question: str) -> Tuple[str, List[str]]:
        if self.retriever is None:
            raise RuntimeError("Vector store not built. Call build_vectorstore() first.")

        retrieved_docs = self.retriever.invoke(question)

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        history_text = ""
        recent = self.chat_history[-(self.memory_window * 2):]
        for msg in recent:
            if isinstance(msg, HumanMessage):
                history_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Assistant: {msg.content}\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant powered by Grok (xAI).
Use the retrieved context below as your PRIMARY source.
If the answer is not in the context, say so honestly.

Context:
{context}

Chat History:
{history}"""),
            ("human", "{question}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        answer = chain.invoke({
            "context": context,
            "history": history_text,
            "question": question,
        })

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

        sources, seen = [], set()
        for doc in retrieved_docs:
            src = doc.metadata.get("source", "unknown")
            snippet = doc.page_content[:120].replace("\n", " ").strip() + "…"
            label = f"[{src}] {snippet}"
            if label not in seen:
                seen.add(label)
                sources.append(label)

        return answer, sources

    def clear_memory(self):
        self.chat_history = []

    def save_vectorstore(self, path: str = "vectorstore/index"):
        if self.vectorstore:
            self.vectorstore.save_local(path)

    def load_vectorstore(self, path: str = "vectorstore/index"):
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.top_k, "fetch_k": self.top_k * 3},
        )