"""
📚 Modular Production RAG Application - Streamlit Web UI
Universal Hybrid Frontend: Supports both FastAPI backend and standalone/Streamlit Cloud direct mode.
"""
import os
import json
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple

import requests
import streamlit as st

from src.config.settings import get_settings

# ==========================================
# Page Configuration & Clean Styling
# ==========================================
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* Hide Streamlit sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    /* Hide Streamlit Deploy button & menu */
    [data-testid="stDeployButton"],
    .stDeployButton,
    #MainMenu {
        display: none !important;
    }

    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 0.98rem;
        color: #475569;
        margin-bottom: 1rem;
    }
    .badge-online {
        display: inline-block;
        padding: 4px 12px;
        background-color: #DCFCE7;
        color: #166534;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .doc-banner {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 1.2rem;
    }
    .upload-card {
        background-color: #F8FAFC;
        border: 2px dashed #CBD5E1;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Hybrid Architecture: FastAPI & In-Process Fallback
# ==========================================
settings = get_settings()
API_BASE_URL = f"http://localhost:{settings.port}"


def run_async_safe(coro):
    """Safely execute an asynchronous coroutine in Streamlit without event loop collisions."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return executor.submit(asyncio.run, coro).result()


def check_fastapi_backend() -> bool:
    """Check if the local FastAPI server is reachable."""
    try:
        res = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        return res.status_code == 200
    except Exception:
        return False


# Determine operating mode
HAS_FASTAPI_SERVER = check_fastapi_backend()


@st.cache_resource(show_spinner="Initializing in-process RAG pipeline...")
def get_direct_pipeline():
    """Fallback in-process pipeline for Streamlit Community Cloud and standalone mode."""
    from src.embeddings.embedder import Embedder
    from src.vectorstore.client import ChromaVectorStore
    from src.generation.llm import GeminiLLM
    from src.pipeline.rag_pipeline import RAGPipeline

    embedder = Embedder.get_instance(
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
    )
    vector_store = ChromaVectorStore(
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
        embedder=embedder,
    )

    api_key = get_active_gemini_key()
    llm = GeminiLLM(api_key=api_key) if api_key else None

    return RAGPipeline(
        vector_store=vector_store,
        llm=llm,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.retrieval_top_k,
        similarity_threshold=settings.similarity_threshold,
    )


def get_active_gemini_key() -> str:
    """Retrieve Gemini API key from session state, environment, or Streamlit Cloud Secrets."""
    if "user_api_key" in st.session_state and st.session_state.user_api_key:
        return st.session_state.user_api_key
    if settings.google_api_key:
        return settings.google_api_key
    if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
        return st.secrets["GOOGLE_API_KEY"]
    return ""


# Initialize in-process pipeline if FastAPI is not detected
direct_pipeline = None if HAS_FASTAPI_SERVER else get_direct_pipeline()


# ==========================================
# Document & Query Handlers (Universal)
# ==========================================
def get_indexed_documents() -> Tuple[List[str], int]:
    """Retrieve indexed documents in either FastAPI mode or Direct mode."""
    if HAS_FASTAPI_SERVER:
        try:
            res = requests.get(f"{API_BASE_URL}/api/v1/documents", timeout=5)
            if res.status_code == 200:
                data = res.json()
                return data.get("documents", []), data.get("total_chunks", 0)
        except Exception:
            pass
    if direct_pipeline:
        return direct_pipeline.list_documents(), direct_pipeline.count()
    return [], 0


def upload_document_universal(file_bytes: bytes, filename: str, replace_existing: bool) -> Dict[str, Any]:
    """Index a document via FastAPI or direct pipeline."""
    if HAS_FASTAPI_SERVER:
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        params = {"clear_existing": "true" if replace_existing else "false"}
        res = requests.post(f"{API_BASE_URL}/api/v1/documents/upload", files=files, params=params, timeout=60)
        if res.status_code == 201:
            return res.json()
        raise RuntimeError(f"FastAPI upload error ({res.status_code}): {res.text}")
    else:
        return direct_pipeline.index_document(file_bytes=file_bytes, filename=filename, clear_existing=replace_existing)


def delete_all_documents_universal() -> int:
    """Clear all documents via FastAPI or direct pipeline."""
    if HAS_FASTAPI_SERVER:
        res = requests.delete(f"{API_BASE_URL}/api/v1/documents/clear-all", timeout=5)
        if res.status_code == 200:
            return res.json().get("chunks_cleared", 0)
    if direct_pipeline:
        return direct_pipeline.clear_all()
    return 0


def query_rag_universal(question: str, target_doc: Optional[str] = None) -> Dict[str, Any]:
    """Query RAG assistant via FastAPI or direct pipeline."""
    filter_dict = {"filename": target_doc} if target_doc else None

    if HAS_FASTAPI_SERVER:
        payload = {"query": question, "top_k": 5}
        if filter_dict:
            payload["filter"] = filter_dict
        res = requests.post(f"{API_BASE_URL}/api/v1/chat", json=payload, timeout=60)
        if res.status_code == 200:
            return res.json()
        raise RuntimeError(f"FastAPI chat error ({res.status_code}): {res.text}")
    else:
        # Update LLM with active key
        active_key = get_active_gemini_key()
        if not active_key:
            raise ValueError("Google Gemini API Key is required. Please provide it in the API Key settings.")
        from src.generation.llm import GeminiLLM
        direct_pipeline.llm = GeminiLLM(api_key=active_key)

        return run_async_safe(direct_pipeline.query(question=question, top_k=5, filter_dict=filter_dict))


# Initialize UI Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

active_docs, total_chunks = get_indexed_documents()
active_key = get_active_gemini_key()


# ==========================================
# 1. Header & System Status
# ==========================================
top_col1, top_col2 = st.columns([3, 1])

with top_col1:
    st.markdown('<div class="main-header">📚 PDF RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload documents and ask questions with verified, page-level citations.</div>',
        unsafe_allow_html=True,
    )

with top_col2:
    mode_label = "FastAPI Backend" if HAS_FASTAPI_SERVER else "Streamlit Cloud Engine"
    st.markdown(
        f'<div style="text-align: right;"><span class="badge-online">🟢 Online ({mode_label})</span></div>',
        unsafe_allow_html=True,
    )

# API Key Alert & Inline Setting (if missing)
if not active_key:
    with st.expander("🔑 Google Gemini API Key Required", expanded=True):
        st.info("A free Google Gemini API Key is needed to generate grounded answers. Get one from [Google AI Studio](https://aistudio.google.com/).")
        key_input = st.text_input("Enter Gemini API Key", type="password", placeholder="AIzaSy...")
        if key_input:
            st.session_state.user_api_key = key_input.strip()
            st.rerun()


# ==========================================
# 2. Inline Document Ingestion (Main Screen)
# ==========================================
if active_docs and not st.session_state.show_uploader:
    current_doc = active_docs[0]

    with st.container():
        st.markdown('<div class="doc-banner">', unsafe_allow_html=True)
        b_col1, b_col2, b_col3 = st.columns([3, 1, 1])

        with b_col1:
            st.markdown(
                f"📄 Active Document: **{current_doc}** &nbsp;·&nbsp; "
                f"**{total_chunks}** chunks indexed &nbsp;·&nbsp; "
                f"<span style='color:#16A34A; font-weight:600;'>Ready for questions</span>",
                unsafe_allow_html=True,
            )

        with b_col2:
            if st.button("📤 Upload New PDF", use_container_width=True):
                st.session_state.show_uploader = True
                st.rerun()

        with b_col3:
            if st.button("🗑️ Clear Document", use_container_width=True):
                delete_all_documents_universal()
                st.session_state.messages = []
                st.session_state.show_uploader = True
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

else:
    with st.container():
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.subheader("📄 Upload Document")
        st.caption("Upload a PDF, Markdown, or text file. It will be chunked and indexed into ChromaDB.")

        up_col1, up_col2 = st.columns([3, 1])
        with up_col1:
            uploaded_file = st.file_uploader(
                "Choose file",
                type=["pdf", "md", "txt"],
                label_visibility="collapsed",
            )
        with up_col2:
            replace_docs = st.checkbox("🧹 Clean Mode (Replace old files)", value=True)
            if active_docs and st.button("Cancel", use_container_width=True):
                st.session_state.show_uploader = False
                st.rerun()

        if uploaded_file is not None:
            if st.button("🚀 Index & Start Chatting", type="primary", use_container_width=True):
                with st.spinner(f"Indexing '{uploaded_file.name}' into vector store..."):
                    try:
                        stats = upload_document_universal(
                            file_bytes=uploaded_file.getvalue(),
                            filename=uploaded_file.name,
                            replace_existing=replace_docs,
                        )
                        chunk_count = stats.get("total_chunks", 0)
                        st.success(f"Indexed **{chunk_count}** chunks from **{uploaded_file.name}**!")
                        st.session_state.messages = []
                        st.session_state.show_uploader = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to index document: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 3. Main Chat Conversation Area
# ==========================================
if not active_docs:
    st.info("👆 Please upload a document above to activate the AI assistant.")
else:
    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("citations"):
                with st.expander(f"📄 View Source Citations ({len(message['citations'])} references)"):
                    for cite in message["citations"]:
                        st.markdown(
                            f"**Source {cite['source_id']} — {cite['filename']} (Page {cite['page']})** "
                            f"| Relevance: `{cite.get('score', 0.0):.2f}`"
                        )
                        st.markdown(f"> {cite['content']}")
                        st.divider()

    # Chat Input Box
    user_query = st.chat_input("Ask any question or request a summary of your document...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating answer..."):
                try:
                    result = query_rag_universal(
                        question=user_query,
                        target_doc=active_docs[0] if active_docs else None,
                    )

                    answer_text = result.get("answer", "")
                    citations = result.get("citations", [])

                    st.markdown(answer_text)

                    if citations:
                        with st.expander(f"📄 View Source Citations ({len(citations)} references)"):
                            for cite in citations:
                                st.markdown(
                                    f"**Source {cite['source_id']} — {cite['filename']} (Page {cite['page']})** "
                                    f"| Relevance: `{cite.get('score', 0.0):.2f}`"
                                )
                                st.markdown(f"> {cite['content']}")
                                st.divider()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "citations": citations,
                    })

                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
