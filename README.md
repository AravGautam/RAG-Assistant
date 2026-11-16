# 🚀 Modular Production RAG Application

A complete, production-grade **Retrieval-Augmented Generation (RAG)** application built in Python using **FastAPI**, **Streamlit**, **Pydantic v2 (`pydantic-settings`)**, **ChromaDB**, **Hugging Face Sentence Transformers (Local & Free)**, and **Google Gemini LLM**.

Designed for enterprise readiness with a strictly decoupled architecture: dedicated modules for document loading, text cleaning, chunking, embedding, vector storage, retrieval, reranking, prompt construction, LLM generation, and guardrails.

---

## ✨ Features

- **Full-Screen Interactive UI (`app.py`)**: Seamless inline document upload, instant indexing, and full-width conversational chat with page-level citations.
- **Production-Ready FastAPI Backend**: Asynchronous REST endpoints, Dependency Injection, Pydantic v2 schemas, Lifespan lifecycle management, and OpenAPI Swagger documentation (`/docs`).
- **Real-Time Token Streaming**: Server-Sent Events (SSE) via `/api/v1/chat/stream` for low-latency streaming answers with trailing citations.
- **100% Free Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` locally on CPU. No rate limits or API bills for vectorizing documents.
- **Persistent ChromaDB Vector Store**: Automatic disk persistence preserving indexed collections across server restarts.
- **Multi-Format Document Parsing**: Built-in support for `.pdf` (via `pypdf`), `.md` (Markdown), and `.txt` files.
- **Strict Grounding & Anti-Hallucination**: System prompts and guardrails enforce strict grounding in the context, preventing fabrication and citing exact page numbers.

---

## 🏗️ Architecture & Folder Structure

```text
Major_Project/
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Production dependencies
├── README.md                        # Documentation
├── app.py                           # Full-screen interactive Streamlit UI
│
├── config/
│   ├── __init__.py
│   ├── settings.py                  # Pydantic v2 BaseSettings (Gemini, Embeddings, Chroma)
│   └── logging_config.py            # Structured logging configuration
│
└── src/
    ├── __init__.py
    │
    ├── ingestion/
    │   ├── __init__.py
    │   ├── loaders.py               # Document loaders (PDF, Markdown, Plain Text)
    │   ├── cleaners.py              # Text cleaning & normalization
    │   └── chunkers.py              # Recursive chunking with metadata enrichment
    │
    ├── embeddings/
    │   ├── __init__.py
    │   └── embedder.py              # Hugging Face Sentence-Transformers embedder wrapper
    │
    ├── vectorstore/
    │   ├── __init__.py
    │   ├── base.py                  # BaseVectorStore abstract interface
    │   └── client.py                # ChromaVectorStore persistent implementation
    │
    ├── retrieval/
    │   ├── __init__.py
    │   ├── retriever.py             # Vector retriever with smart summary detection
    │   ├── reranker.py              # Score normalization & deduplication
    │   └── query_transform.py       # Query hygiene & normalization
    │
    ├── generation/
    │   ├── __init__.py
    │   ├── prompts.py               # Grounded prompt templates & context formatters
    │   ├── llm.py                   # Google Gemini LLM wrapper (Sync & Async stream)
    │   └── guardrails.py            # Output verification & refusal checks
    │
    ├── pipeline/
    │   ├── __init__.py
    │   └── rag_pipeline.py          # Unified RAGPipeline orchestrator
    │
    └── api/
        ├── __init__.py
        ├── main.py                  # FastAPI application factory & routes
        ├── dependencies.py          # Dependency injection providers
        ├── schemas.py               # Pydantic v2 request/response models
        └── routes/
            ├── __init__.py
            ├── chat.py              # POST /chat & POST /chat/stream
            └── documents.py         # POST /upload, GET /list, DELETE /{filename}
```

---

## ⚡ Quick Start

### 1. Setup Environment
Ensure Python 3.10+ is installed:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Key
Create your `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Edit `.env` and insert your free **Google Gemini API Key** (obtainable from [Google AI Studio](https://aistudio.google.com/)):
```env
GOOGLE_API_KEY=AIzaSy...
```

### 3. Launch the Application

- **Start the UI (Streamlit)**:
  ```bash
  streamlit run app.py
  ```
  Open: **[http://localhost:8501](http://localhost:8501)**

- **Start the Backend API (FastAPI)**:
  ```bash
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
  ```
  Open Swagger Docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**
