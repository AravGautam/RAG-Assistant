"""
📚 Modular Production RAG Application - Streamlit Web UI
Enterprise Edition with Dual-Pane "NotebookLM" Layout:
- Left Column (45%): Interactive Document / PDF Viewer with Page-level Citations & Highlighting
- Right Column (55%): Sleek Conversational Chat Feed with Real-time Token Streaming
- One-Click View Toggle: [ 📖 Split View ] ↔ [ 💬 Full Chat ]
- Hybrid Semantic + BM25 Lexical Retrieval with Reciprocal Rank Fusion
- Multi-Turn Conversational Memory & Contextual Query Rewriting
- 1-Click Starter Prompts & Document Insights
- Markdown Transcript & Citations Report Export
"""
import os
import re
import json
import time
import base64
import asyncio
import urllib.parse
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components

# Safe import for PyMuPDF with fallbacks for cloud deployment
try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    try:
        import fitz as pymupdf
        HAS_PYMUPDF = True
    except ImportError:
        pymupdf = None
        HAS_PYMUPDF = False

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    pypdf = None
    HAS_PYPDF = False

# Sync Streamlit Cloud secrets to os.environ so settings & modules pick them up automatically
try:
    if hasattr(st, "secrets"):
        for _k, _v in st.secrets.items():
            if isinstance(_v, (str, int, float, bool)) and _k not in os.environ:
                os.environ[_k] = str(_v)
except Exception:
    pass

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
    /* Hide Streamlit default chrome */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    [data-testid="stDeployButton"],
    .stDeployButton,
    #MainMenu {
        display: none !important;
    }

    .main-header {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.1rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 0.92rem;
        color: #64748B;
        margin-bottom: 0.6rem;
    }
    .badge-online {
        display: inline-block;
        padding: 3px 10px;
        background-color: #DCFCE7;
        color: #166534;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .doc-banner {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 0.8rem;
    }
    .doc-viewer-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .page-canvas-wrapper {
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .highlight-badge {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.83rem;
        margin-bottom: 10px;
        color: #92400E;
    }
    .upload-card {
        background-color: #F8FAFC;
        border: 2px dashed #CBD5E1;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 1.2rem;
    }
    .starter-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 1rem;
    }
    .metric-pill {
        display: inline-block;
        color: #64748B;
        font-size: 0.78rem;
        margin-top: 4px;
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


HAS_FASTAPI_SERVER = check_fastapi_backend()


def get_active_gemini_key() -> str:
    """Retrieve Gemini API key strictly from environment variables or Streamlit Cloud Secrets."""
    if hasattr(st, "secrets"):
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                return str(st.secrets["GOOGLE_API_KEY"]).strip()
        except Exception:
            pass
    if settings.google_api_key:
        return settings.google_api_key.strip()
    return os.getenv("GOOGLE_API_KEY", "").strip()


@st.cache_resource(show_spinner=False)
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
    llm = (
        GeminiLLM(
            api_key=api_key,
            model=settings.gemini_model,
            temperature=settings.gemini_temperature,
        )
        if api_key
        else None
    )

    return RAGPipeline(
        vector_store=vector_store,
        llm=llm,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.retrieval_top_k,
        similarity_threshold=settings.similarity_threshold,
    )


direct_pipeline = None if HAS_FASTAPI_SERVER else get_direct_pipeline()


# ==========================================
# Document & Query Handlers
# ==========================================
def save_uploaded_file(file_bytes: bytes, filename: str, replace_existing: bool) -> str:
    """Save raw file bytes to data/raw for local PDF viewer rendering."""
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    if replace_existing:
        for f in os.listdir(raw_dir):
            p = os.path.join(raw_dir, f)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
    target_path = os.path.join(raw_dir, filename)
    with open(target_path, "wb") as f:
        f.write(file_bytes)
    return target_path


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
    pipeline = direct_pipeline or get_direct_pipeline()
    return pipeline.list_documents(), pipeline.count()


def upload_document_universal(file_bytes: bytes, filename: str, replace_existing: bool) -> Dict[str, Any]:
    """Index a document via FastAPI or direct pipeline."""
    save_uploaded_file(file_bytes, filename, replace_existing)
    if HAS_FASTAPI_SERVER:
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        params = {"clear_existing": "true" if replace_existing else "false"}
        res = requests.post(f"{API_BASE_URL}/api/v1/documents/upload", files=files, params=params, timeout=60)
        if res.status_code == 201:
            return res.json()
        raise RuntimeError(f"FastAPI upload error ({res.status_code}): {res.text}")
    else:
        pipeline = direct_pipeline or get_direct_pipeline()
        return pipeline.index_document(file_bytes=file_bytes, filename=filename, clear_existing=replace_existing)


def delete_all_documents_universal() -> int:
    """Clear all documents via FastAPI or direct pipeline."""
    raw_dir = os.path.join("data", "raw")
    if os.path.exists(raw_dir):
        for f in os.listdir(raw_dir):
            p = os.path.join(raw_dir, f)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
    if HAS_FASTAPI_SERVER:
        res = requests.delete(f"{API_BASE_URL}/api/v1/documents/clear-all", timeout=5)
        if res.status_code == 200:
            return res.json().get("chunks_cleared", 0)
    pipeline = direct_pipeline or get_direct_pipeline()
    return pipeline.clear_all()


def fluid_word_stream(raw_stream, delay: float = 0.012):
    """
    Smoothly streams text word-by-word with natural typing cadence,
    emulating the ChatGPT typing effect.
    """
    for chunk in raw_stream:
        # Split chunk into words to produce natural human typing cadence
        words = chunk.split(" ")
        for idx, word in enumerate(words):
            suffix = " " if idx < len(words) - 1 else ""
            yield word + suffix
            if delay > 0:
                time.sleep(delay)


def query_rag_stream_universal(
    question: str,
    target_doc: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    status_callback: Optional[Any] = None,
) -> Tuple[Any, List[Dict[str, Any]], str]:
    """
    Stream answer tokens directly using direct pipeline for instant real-time response.
    """
    filter_dict = {"filename": target_doc} if target_doc else None
    pipeline = direct_pipeline or get_direct_pipeline()

    active_key = get_active_gemini_key()
    if not active_key:
        raise ValueError("GOOGLE_API_KEY environment variable is missing. Add it to your .env file or Streamlit Cloud Secrets.")

    from src.generation.llm import GeminiLLM
    pipeline.llm = GeminiLLM(
        api_key=active_key,
        model=settings.gemini_model,
        temperature=settings.gemini_temperature,
    )

    return pipeline.query_stream_sync(
        question=question,
        top_k=settings.retrieval_top_k,
        filter_dict=filter_dict,
        chat_history=chat_history,
        status_callback=status_callback,
    )


def generate_chat_report(messages: List[Dict[str, Any]], docs: List[str], chunk_count: int) -> str:
    """Compile conversation and citations into a clean Markdown report."""
    lines = [
        "# 📚 PDF RAG Assistant — Conversation Report",
        f"**Date Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Indexed Documents:** {', '.join(docs) if docs else 'None'}",
        f"**Total Chunks In Knowledge Base:** {chunk_count}",
        "",
        "---",
        "",
    ]
    for idx, msg in enumerate(messages, 1):
        if msg["role"] == "user":
            lines.append(f"### Q: {msg['content']}")
        else:
            lines.append(f"**Answer:**\n{msg['content']}\n")
            if msg.get("citations"):
                lines.append("**Source Citations:**")
                for cite in msg["citations"]:
                    lines.append(f"- *[{cite['filename']} - Page {cite['page']}]* {cite['content'][:140]}...")
                lines.append("")
            lines.append("---")
            lines.append("")
    return "\n".join(lines)


# ==========================================
# Initialize UI Session State
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "📖 Split View"

if "pdf_page" not in st.session_state:
    st.session_state.pdf_page = 1

if "active_snippet" not in st.session_state:
    st.session_state.active_snippet = None

if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

active_docs, total_chunks = get_indexed_documents()
if active_docs and (not st.session_state.selected_doc or st.session_state.selected_doc not in active_docs):
    st.session_state.selected_doc = active_docs[0]


def set_pdf_page_jump(page: int, filename: Optional[str] = None, snippet: Optional[str] = None):
    """Safely updates session state keys to jump the document viewer to a page."""
    try:
        page_num = max(1, int(float(page)))
    except (ValueError, TypeError):
        page_num = 1
    st.session_state.pdf_page = page_num
    st.session_state.view_mode = "📖 Split View"
    if filename:
        st.session_state.selected_doc = filename
    if snippet:
        st.session_state.active_snippet = snippet.strip()[:300]


# ==========================================
# 1. Header & System Status Bar
# ==========================================
top_col1, top_col2 = st.columns([2.8, 1.2])

with top_col1:
    st.markdown('<div class="main-header">📚 PDF RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Dual-Pane NotebookLM Layout &middot; Real-time Streaming &middot; Verified Page Citations</div>',
        unsafe_allow_html=True,
    )

with top_col2:
    st.markdown(
        f'<div style="text-align: right; margin-bottom: 6px;"><span class="badge-online">🟢 Online ({"FastAPI" if HAS_FASTAPI_SERVER else "In-Process"})</span></div>',
        unsafe_allow_html=True,
    )
    if active_docs:
        view_idx = 0 if st.session_state.view_mode == "📖 Split View" else 1
        view_choice = st.radio(
            "Layout Mode",
            options=["📖 Split View", "💬 Full Chat"],
            index=view_idx,
            horizontal=True,
            label_visibility="collapsed",
            key=f"layout_mode_radio_{st.session_state.view_mode}",
        )
        st.session_state.view_mode = view_choice


# ==========================================
# 2. Document Banner
# ==========================================
selected_target_doc: Optional[str] = st.session_state.selected_doc

if active_docs and not st.session_state.show_uploader:
    with st.container():
        st.markdown('<div class="doc-banner">', unsafe_allow_html=True)
        doc_info = (
            f"📄 Active Document: <strong>{active_docs[0]}</strong>"
            if len(active_docs) == 1
            else f"📚 <strong>{len(active_docs)} Documents Indexed</strong>"
        )
        st.markdown(
            f"{doc_info} &nbsp;·&nbsp; "
            f"<strong>{total_chunks}</strong> chunks indexed &nbsp;·&nbsp; "
            f"<span style='color:#16A34A; font-weight:600;'>Ready for questions</span>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

elif not active_docs or st.session_state.show_uploader:
    with st.container():
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.subheader("📄 Upload Document")
        st.caption("Upload a PDF, Markdown, or text file. It will be indexed into Hybrid BM25 + Vector Search.")

        up_col1, up_col2 = st.columns([3, 1])
        with up_col1:
            uploaded_file = st.file_uploader(
                "Choose file",
                type=["pdf", "md", "txt"],
                label_visibility="collapsed",
            )
        with up_col2:
            replace_docs = st.checkbox("🧹 Clean Mode (Replace old files)", value=True if not active_docs else False)
            if active_docs and st.button("Cancel", use_container_width=True):
                st.session_state.show_uploader = False
                st.rerun()

        if uploaded_file is not None:
            if st.button("🚀 Index & Start Chatting", type="primary", use_container_width=True):
                with st.spinner(f"Indexing '{uploaded_file.name}' into vector store & BM25 index..."):
                    try:
                        stats = upload_document_universal(
                            file_bytes=uploaded_file.getvalue(),
                            filename=uploaded_file.name,
                            replace_existing=replace_docs,
                        )
                        chunk_count = stats.get("total_chunks", 0)
                        st.session_state.selected_doc = uploaded_file.name
                        st.session_state.pdf_page = 1
                        st.session_state.active_snippet = None
                        st.session_state.show_uploader = False
                        st.success(f"Indexed **{chunk_count}** chunks from **{uploaded_file.name}**!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to index document: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 3. Interactive Document Viewer (Left Pane)
# ==========================================
def render_document_viewer(docs: List[str], current_target_doc: Optional[str]):
    """Renders the left column document previewer with page jumping and citation highlighting."""
    target_name = current_target_doc or (docs[0] if docs else None)
    if not target_name:
        st.info("No document selected.")
        return

    raw_path = os.path.join("data", "raw", target_name)
    ext = os.path.splitext(target_name)[1].lower()

    st.markdown('<div class="doc-viewer-card">', unsafe_allow_html=True)

    if ext == ".pdf" and os.path.exists(raw_path):
        try:
            doc = None
            total_pages = 1
            if HAS_PYMUPDF:
                try:
                    doc = pymupdf.open(raw_path)
                    total_pages = max(1, len(doc))
                except Exception:
                    doc = None
            
            if doc is None and HAS_PYPDF:
                try:
                    reader = pypdf.PdfReader(raw_path)
                    total_pages = max(1, len(reader.pages))
                except Exception:
                    total_pages = 1

            # Ensure page bounds
            if st.session_state.pdf_page < 1:
                st.session_state.pdf_page = 1
            elif st.session_state.pdf_page > total_pages:
                st.session_state.pdf_page = total_pages

            cur_page = st.session_state.pdf_page

            # Viewer Header & Navigation Controls
            vc1, vc2, vc3, vc4 = st.columns([1, 1.8, 1, 1.2])
            with vc1:
                if st.button("◀ Prev", disabled=(cur_page <= 1), use_container_width=True, key=f"viewer_prev_{cur_page}"):
                    set_pdf_page_jump(cur_page - 1)
                    st.session_state.active_snippet = None
                    st.rerun()
            with vc2:
                st.markdown(
                    f"<div style='text-align: center; font-weight:600; padding-top:4px;'>Page {cur_page} of {total_pages}</div>",
                    unsafe_allow_html=True,
                )
            with vc3:
                if st.button("Next ▶", disabled=(cur_page >= total_pages), use_container_width=True, key=f"viewer_next_{cur_page}"):
                    set_pdf_page_jump(cur_page + 1)
                    st.session_state.active_snippet = None
                    st.rerun()
            with vc4:
                jump_to = st.number_input(
                    "Go to",
                    min_value=1,
                    max_value=total_pages,
                    value=cur_page,
                    step=1,
                    label_visibility="collapsed",
                    key=f"viewer_page_input_{cur_page}",
                )
                if jump_to != cur_page:
                    set_pdf_page_jump(jump_to)
                    st.session_state.active_snippet = None
                    st.rerun()

            # Active Citation Banner
            if st.session_state.get("active_snippet"):
                st.markdown(
                    f'<div class="highlight-badge">📌 <strong>Focused Citation on Page {cur_page}:</strong><br/>'
                    f'<em>"{st.session_state.active_snippet}"</em></div>',
                    unsafe_allow_html=True,
                )

            def render_native_pdf(t_name: str, r_path: str, page_num: int):
                if HAS_FASTAPI_SERVER:
                    encoded_name = urllib.parse.quote(t_name)
                    raw_url = f"http://localhost:8000/api/v1/documents/raw/{encoded_name}?p={page_num}&ts={int(time.time()*1000)}#page={page_num}&view=FitH,0,0"
                    components.html(
                        f"""
                        <iframe id="pdf_frame_{page_num}" 
                                src="{raw_url}" 
                                width="100%" 
                                height="700px" 
                                style="border: 1px solid #CBD5E1; border-radius: 8px;">
                        </iframe>
                        """,
                        height=710,
                    )
                else:
                    with open(r_path, "rb") as f:
                        b64_pdf = base64.b64encode(f.read()).decode("utf-8")
                    components.html(
                        f"""
                        <iframe id="pdf_frame_{page_num}_{int(time.time()*1000)}" 
                                src="data:application/pdf;base64,{b64_pdf}#page={page_num}&view=FitH,0,0" 
                                width="100%" 
                                height="700px" 
                                type="application/pdf" 
                                style="border: 1px solid #CBD5E1; border-radius: 8px;">
                        </iframe>
                        """,
                        height=710,
                    )

            if doc is not None:
                # Document Tabs: High-Res Slide View vs Embedded PDF
                tab_slide, tab_embed = st.tabs(["🖼️ High-Res Slide View", "📜 Native PDF Embed"])

                with tab_slide:
                    page_obj = doc.load_page(cur_page - 1)

                    # Search and highlight active citation text on the slide
                    if st.session_state.get("active_snippet"):
                        try:
                            clean_text = re.sub(r'[\r\n\t]+', ' ', st.session_state.active_snippet).strip()
                            words = [w for w in clean_text.split() if len(w) > 3][:6]
                            search_phrase = " ".join(words[:4]) if words else clean_text[:25]
                            rects = page_obj.search_for(search_phrase)
                            if not rects and words:
                                rects = page_obj.search_for(words[0])
                            for r in rects:
                                annot = page_obj.add_highlight_annot(r)
                                if annot:
                                    annot.set_colors(stroke=(1.0, 0.82, 0.0))
                                    annot.update()
                        except Exception:
                            pass

                    pix = page_obj.get_pixmap(dpi=140)
                    st.markdown('<div class="page-canvas-wrapper">', unsafe_allow_html=True)
                    st.image(pix.tobytes("png"), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with tab_embed:
                    render_native_pdf(target_name, raw_path, cur_page)
            else:
                render_native_pdf(target_name, raw_path, cur_page)

        except Exception as e:
            st.error(f"Error rendering PDF: {str(e)}")

    elif ext in [".md", ".txt"] and os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
            text_content = f.read()
        st.markdown(f"#### 📄 {target_name}")
        st.text_area("Document Content", value=text_content, height=650, disabled=True)

    else:
        st.info(f"Viewing metadata for **{target_name}**.")
        st.caption("Raw file preview will appear here once saved in local storage.")

    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 4. Chat Feed & Query Stream (Right Pane / Full)
# ==========================================
def render_chat_interface(docs: List[str], current_target_doc: Optional[str]):
    """Renders the conversational chat interface with bottom-pinned search box, bottom options bar, and seamless streaming."""
    if not docs:
        st.info("👆 Please upload a document above to activate the AI assistant.")
        return

    # 1. Scrollable Chat Container (Takes up main height, all messages stay inside)
    chat_container = st.container(height=620, border=False)

    with chat_container:
        # Render all past messages
        for msg_idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message.get("citations"):
                    with st.expander(f"📄 View Source Citations ({len(message['citations'])} references)"):
                        for cite_idx, cite in enumerate(message["citations"]):
                            c_col1, c_col2 = st.columns([3.5, 1.2])
                            with c_col1:
                                st.markdown(
                                    f"**Source {cite['source_id']} — {cite['filename']} (Page {cite['page']})** "
                                    f"| Relevance: `{cite.get('score', 0.0):.2f}`"
                                )
                                st.markdown(f"> {cite['content']}")
                            with c_col2:
                                if st.button(
                                    f"🔍 Jump to P.{cite['page']}",
                                    key=f"btn_jump_{msg_idx}_{cite_idx}",
                                    use_container_width=True,
                                ):
                                    set_pdf_page_jump(int(cite["page"]), cite.get("filename"), cite.get("content"))
                                    st.rerun()
                            st.divider()

                if message.get("metric"):
                    st.markdown(f'<div class="metric-pill">{message["metric"]}</div>', unsafe_allow_html=True)

    # 2. Bottom Action Toolbar (Directly beneath the chat container, above chat input)
    act_c1, act_c2, act_c3, act_c4 = st.columns([1.6, 1.4, 1.2, 3.8])

    with act_c1:
        with st.popover("📎 Upload PDF", use_container_width=True):
            st.markdown("##### 📄 Upload Document")
            pop_file = st.file_uploader(
                "Select file",
                type=["pdf", "md", "txt"],
                key="pop_uploader_input",
                label_visibility="collapsed",
            )
            pop_clean = st.checkbox("Replace existing", value=False, key="pop_clean_check")
            if pop_file is not None:
                if st.button("🚀 Index Document", type="primary", use_container_width=True, key="pop_idx_btn"):
                    with st.spinner(f"Indexing '{pop_file.name}'..."):
                        upload_document_universal(pop_file.getvalue(), pop_file.name, replace_existing=pop_clean)
                        st.session_state.selected_doc = pop_file.name
                        st.session_state.pdf_page = 1
                        st.rerun()

    with act_c2:
        if st.session_state.messages:
            report_text = generate_chat_report(st.session_state.messages, docs, total_chunks)
            st.download_button(
                "📥 Export",
                data=report_text,
                file_name=f"chat_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
                key="bottom_export_btn",
            )
        else:
            st.button("📥 Export", disabled=True, use_container_width=True, key="bottom_export_dis")

    with act_c3:
        if st.button("🗑️ Clear", use_container_width=True, key="bottom_clear_btn", help="Clear conversation"):
            st.session_state.messages = []
            st.session_state.active_snippet = None
            st.rerun()

    with act_c4:
        if len(docs) > 1:
            scope_val = st.selectbox(
                "Scope",
                options=["All Documents"] + docs,
                index=0,
                label_visibility="collapsed",
                key="bottom_scope_dropdown",
            )
            current_target_doc = None if scope_val == "All Documents" else scope_val

    # 3. Bottom Chat Input (Permanently anchored at the very bottom)
    input_query = st.chat_input("Ask any question or follow-up about your document...")
    active_prompt = input_query or st.session_state.get("pending_query")

    if active_prompt:
        st.session_state.pending_query = None

        # Render active interaction inside chat_container
        with chat_container:
            with st.chat_message("user"):
                st.markdown(active_prompt)

            with st.chat_message("assistant"):
                start_time = time.time()
                try:
                    with st.spinner("Searching document & generating answer..."):
                        token_stream, citations, effective_query = query_rag_stream_universal(
                            question=active_prompt,
                            target_doc=current_target_doc,
                            chat_history=st.session_state.messages,
                        )
                        token_iter = iter(token_stream)
                        first_chunk = None
                        try:
                            first_chunk = next(token_iter)
                        except StopIteration:
                            pass

                    def combined_stream():
                        if first_chunk is not None:
                            yield first_chunk
                        for chunk in token_iter:
                            yield chunk

                    full_answer = st.write_stream(fluid_word_stream(combined_stream(), delay=0.008))
                    elapsed = time.time() - start_time

                    if citations:
                        with st.expander(f"📄 View Source Citations ({len(citations)} references)"):
                            for cite_idx, cite in enumerate(citations):
                                c_col1, c_col2 = st.columns([3.5, 1.2])
                                with c_col1:
                                    st.markdown(
                                        f"**Source {cite['source_id']} — {cite['filename']} (Page {cite['page']})** "
                                        f"| Relevance: `{cite.get('score', 0.0):.2f}`"
                                    )
                                    st.markdown(f"> {cite['content']}")
                                with c_col2:
                                    if st.button(
                                        f"🔍 Jump to P.{cite['page']}",
                                        key=f"live_jump_{cite_idx}",
                                        use_container_width=True,
                                    ):
                                        set_pdf_page_jump(int(cite["page"]), cite.get("filename"), cite.get("content"))
                                        st.rerun()
                                st.divider()

                    metric_str = f"⚡ {elapsed:.1f}s &nbsp;·&nbsp; 🟢 Fully Grounded &nbsp;·&nbsp; 📄 {len(citations)} citations"
                    if effective_query != active_prompt:
                        metric_str += f' &nbsp;·&nbsp; 🔄 <i>Context: "{effective_query}"</i>'

                    st.markdown(f'<div class="metric-pill">{metric_str}</div>', unsafe_allow_html=True)

                    st.session_state.messages.append({"role": "user", "content": active_prompt})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                        "citations": citations,
                        "metric": metric_str,
                    })

                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")


# ==========================================
# 5. Dual-Pane Layout Assembly
# ==========================================
if active_docs and st.session_state.view_mode == "📖 Split View":
    pane_col1, pane_col2 = st.columns([45, 55], gap="medium")
    with pane_col1:
        render_document_viewer(active_docs, selected_target_doc)
    with pane_col2:
        render_chat_interface(active_docs, selected_target_doc)
else:
    render_chat_interface(active_docs, selected_target_doc)
