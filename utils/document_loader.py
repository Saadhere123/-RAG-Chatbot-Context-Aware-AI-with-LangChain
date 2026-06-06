from __future__ import annotations

import io
from typing import List, Tuple

from langchain_core.documents import Document


SAMPLE_CORPUS = {
    "artificial_intelligence": """
Artificial Intelligence (AI) is the simulation of human intelligence processes by machines,
especially computer systems. AI research has been defined as the field of study of intelligent
agents — any device that perceives its environment and takes actions that maximize its chance of
successfully achieving its goals.

Machine learning is a subset of AI that provides systems the ability to automatically learn and
improve from experience without being explicitly programmed. Deep learning is a subset of machine
learning using multi-layered neural networks that learn representations of data.

Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret,
and manipulate human language. Applications include chatbots, language translation, sentiment
analysis, and text summarization.

Computer vision enables machines to interpret and make decisions based on visual data. It powers
autonomous vehicles, facial recognition, medical image analysis, and object detection systems.

Reinforcement learning is an area where an agent learns to make decisions by interacting with
an environment to maximize cumulative rewards. It has achieved superhuman performance in games
like Chess, Go, and complex video games.
""",

    "large_language_models": """
Large Language Models (LLMs) are AI systems trained on vast amounts of text data using
transformer architectures. GPT-4, Claude, LLaMA, and Gemini are examples of state-of-the-art LLMs.

Transformer architecture, introduced in Attention is All You Need (2017), uses self-attention
mechanisms to process sequential data in parallel rather than sequentially.

Retrieval-Augmented Generation (RAG) combines LLMs with external knowledge retrieval. Instead of
relying solely on parametric memory, RAG retrieves relevant documents from a vector database and
uses them as context for generation. This reduces hallucinations and keeps knowledge current.

Prompt engineering is the practice of designing effective prompts to guide LLM behavior.
Techniques include chain-of-thought prompting, few-shot learning, and role-based instructions.

Vector embeddings represent text as dense numerical vectors in high-dimensional space. Semantically
similar texts have embeddings that are close together, enabling similarity search.
""",

    "python_programming": """
Python is a high-level, interpreted programming language known for its simplicity and readability.
Created by Guido van Rossum and released in 1991, Python emphasizes code readability with
significant indentation and clear syntax.

Key Python features include dynamic typing, automatic memory management, comprehensive standard
library, and support for multiple paradigms: procedural, object-oriented, and functional.

Popular Python frameworks: Django and Flask for web development; NumPy, Pandas, and Matplotlib
for data science; TensorFlow, PyTorch, and scikit-learn for machine learning; FastAPI for APIs.

Python's package manager pip and virtual environments enable dependency management.
The Python Package Index (PyPI) hosts over 400,000 packages.
""",

    "vector_databases": """
Vector databases are specialized databases designed to store, index, and query high-dimensional
vectors (embeddings). They are fundamental to AI applications like semantic search, recommendation
systems, and RAG pipelines.

Popular vector databases include FAISS (Facebook AI Similarity Search), Pinecone, Weaviate,
Chroma, and Qdrant.

Approximate Nearest Neighbor (ANN) algorithms like HNSW and IVF enable fast retrieval from
millions of vectors.

Similarity metrics used in vector search: cosine similarity, Euclidean distance, and dot product.
""",

    "langchain_framework": """
LangChain is an open-source framework for building applications powered by language models.
It provides abstractions for chaining LLM calls, integrating tools, managing memory, and
building complex AI workflows.

Core LangChain components: Chains, Agents, Memory, Retrievers, and Tools.

LangChain Expression Language (LCEL) is a declarative way to compose chains using the pipe
operator. It supports streaming, parallel execution, and easy debugging.

LangChain integrates with dozens of vector stores, LLM providers, document loaders, and text
splitters.
""",

    "cloud_computing": """
Cloud computing delivers computing services over the internet to offer faster innovation,
flexible resources, and economies of scale. Major providers include AWS, Google Cloud Platform,
and Microsoft Azure.

Service models: Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and
Software as a Service (SaaS).

Serverless computing lets developers run code without managing servers. AWS Lambda, Google Cloud
Functions, and Azure Functions automatically scale.

Containerization with Docker packages applications into portable containers. Kubernetes
orchestrates containers at scale.
""",
}


def load_sample_corpus() -> Tuple[List[Document], int]:
    docs = []
    for topic, content in SAMPLE_CORPUS.items():
        docs.append(Document(
            page_content=content.strip(),
            metadata={"source": topic.replace("_", " ").title(), "type": "sample"},
        ))
    return docs, len(docs)


def load_documents_from_folder(uploaded_files) -> Tuple[List[Document], int]:
    docs = []

    for file in uploaded_files:
        content = ""
        filename = file.name

        if filename.endswith(".txt"):
            content = file.read().decode("utf-8", errors="replace")

        elif filename.endswith(".pdf"):
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(file.read()))
                content = "\n\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except ImportError:
                content = f"[pypdf not installed. Run: pip install pypdf]\nFilename: {filename}"

        if content.strip():
            docs.append(Document(
                page_content=content,
                metadata={"source": filename, "type": "uploaded"},
            ))

    return docs, len(docs)