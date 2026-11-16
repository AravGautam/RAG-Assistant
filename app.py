"""
📚 Modular Production RAG Application - Streamlit Web UI (Single-Page Layout)
Clean, modern full-screen interface with inline document ingestion (No Sidebar).
"""
import os
import json
import requests
import streamlit as st
from typing import List, Dict, Any, Optional

from src.config.settings import get_settings

# Page Configuration: Full width, no sidebar
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling for full-width layout
st.markdown("""
<style>
    /* Hide Streamlit sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"] {
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
        padding: 3px 10px;
        background-color: #DCFCE7;
        color: #166534;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-offline {
        display: inline-block;
        padding: 3px 10px;
        background-color: #FEE2E2;
        color: #991B1B;
        border-radius: 9999px;
        font-size: 0.8rem;
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

settings = get_settings()
API_BASE_URL = f"http://localhost:{settings.port}"


def check_backend_health() -> Optional[Dict[str, Any]]:
    """Check if the FastAPI backend is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def fetch_indexed_documents() -> Dict[str, Any]:
    """Fetch the list of indexed documents from the backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/documents", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"documents": [], "total_documents": 0, "total_chunks": 0}


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

# Backend state check
backend_health = check_backend_health()
doc_info = fetch_indexed_documents() if backend_health else {"documents": [], "total_documents": 0, "total_chunks": 0}
active_docs = doc_info.get("documents", [])
total_chunks = doc_info.get("total_chunks", 0)


# ==========================================
# 1. Header & Navigation (Main Screen)
# ==========================================
top_col1, top_col2 = st.columns([3, 1])

with top_col1:
    st.markdown('<div class="main-header">📚 PDF RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload documents and ask questions with verified, page-level citations.</div>',
        unsafe_allow_html=True,
    )

with top_col2:
    if backend_health:
        st.markdown(
            f'<div style="text-align: right;"><span class="badge-online">🟢 FastAPI Connected (v{backend_health.get("version", "1.0.0")})</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align: right;"><span class="badge-offline">🔴 Backend Offline</span></div>',
            unsafe_allow_html=True,
        )


# ==========================================
# 2. Inline Document Ingestion (Main Screen)
# ==========================================

# Case A: Document is already active -> Display compact active bar
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
                requests.delete(f"{API_BASE_URL}/api/v1/documents/clear-all")
                st.session_state.messages = []
                st.session_state.show_uploader = True
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# Case B: No document active OR user clicked "Upload New PDF"
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
                if not backend_health:
                    st.error("Cannot index: FastAPI backend is offline.")
                else:
                    with st.spinner(f"Indexing '{uploaded_file.name}' into vector store..."):
                        try:
                            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
                            params = {"clear_existing": "true" if replace_docs else "false"}
                            res = requests.post(
                                f"{API_BASE_URL}/api/v1/documents/upload",
                                files=files,
                                params=params,
                                timeout=60,
                            )
                            if res.status_code == 201:
                                data = res.json()
                                st.success(f"Indexed **{data['total_chunks']}** chunks from **{data['filename']}**!")
                                st.session_state.messages = []
                                st.session_state.show_uploader = False
                                st.rerun()
                            else:
                                st.error(f"Upload failed ({res.status_code}): {res.text}")
                        except Exception as e:
                            st.error(f"Error communicating with backend: {str(e)}")

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

    # Chat Input Box at bottom of screen
    user_query = st.chat_input("Ask any question or request a summary of your document...")

    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            if not backend_health:
                st.error("Cannot query: FastAPI backend is offline.")
            else:
                with st.spinner("Searching document and generating answer..."):
                    try:
                        payload = {"query": user_query, "top_k": 5}
                        if active_docs:
                            payload["filter"] = {"filename": active_docs[0]}

                        response = requests.post(f"{API_BASE_URL}/api/v1/chat", json=payload, timeout=60)

                        if response.status_code == 200:
                            data = response.json()
                            answer_text = data.get("answer", "")
                            citations = data.get("citations", [])

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

                            # Save to chat history
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer_text,
                                "citations": citations,
                            })

                        else:
                            st.error(f"Error ({response.status_code}): {response.text}")

                    except Exception as e:
                        st.error(f"Failed to query backend: {str(e)}")
