"""
RAG Pipeline Orchestrator coordinating ingestion, retrieval, reranking, and generation.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator, Callable

from src.ingestion.loaders import DocumentLoader
from src.ingestion.cleaners import clean_documents
from src.ingestion.chunkers import DocumentChunker
from src.vectorstore.base import BaseVectorStore
from src.vectorstore.client import ChromaVectorStore
from src.retrieval.query_transform import QueryTransformer
from src.retrieval.retriever import VectorRetriever
from src.retrieval.reranker import Reranker
from src.generation.prompts import get_rag_prompt_template, format_context
from src.generation.llm import GeminiLLM
from src.generation.guardrails import Guardrails

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Unified production RAG pipeline.
    """

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        llm: Optional[GeminiLLM] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4,
        similarity_threshold: float = 0.0,
    ) -> None:
        self.vector_store = vector_store or ChromaVectorStore()
        self.llm = llm
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.retriever = VectorRetriever(
            vector_store=self.vector_store,
            top_k=top_k,
            score_threshold=similarity_threshold,
        )
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def index_document(
        self,
        file_bytes: bytes,
        filename: str,
        clear_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Load, clean, chunk, and index a document into the vector store.

        Args:
            file_bytes: Raw bytes of the document.
            filename: Name of the uploaded file.
            clear_existing: If True, wipe previous documents before indexing this one.

        Returns:
            Dictionary containing document ingestion statistics.
        """
        logger.info("Starting ingestion pipeline for file: '%s' (clear_existing=%s)", filename, clear_existing)

        if clear_existing:
            self.clear_all()

        # 1. Load document
        raw_docs = DocumentLoader.load_from_bytes(file_bytes, filename)

        # 2. Clean and sanitize text
        cleaned_docs = clean_documents(raw_docs)

        # 3. Chunk documents with metadata
        chunks, stats = self.chunker.split_documents(cleaned_docs)

        # 4. Index into vector store
        self.vector_store.add_documents(chunks)

        stats["total_pages"] = raw_docs[0].metadata.get("total_pages", len(raw_docs)) if raw_docs else 0
        logger.info("Finished indexing '%s': %d chunks created", filename, len(chunks))
        return stats

    def index_file_from_path(self, file_path: str, clear_existing: bool = False) -> Dict[str, Any]:
        """
        Ingest a file directly from a filesystem path.
        """
        with open(file_path, "rb") as f:
            content = f.read()
        import os
        filename = os.path.basename(file_path)
        return self.index_document(content, filename, clear_existing=clear_existing)

    async def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full RAG generation pipeline asynchronously.

        Returns:
            Dict containing answer, citations, model name, and grounding status.
        """
        if not self.llm:
            raise ValueError("LLM is not initialized. Ensure GOOGLE_API_KEY is configured.")

        # 1. Transform and contextualize query with chat history
        clean_query = QueryTransformer.contextualize_query(
            question, chat_history=chat_history, llm=self.llm
        )

        # 2. Retrieve candidates
        k_val = top_k or self.top_k
        raw_candidates = self.retriever.retrieve(clean_query, k=k_val * 2, filter_dict=filter_dict)

        # 3. Rerank & deduplicate
        ranked_docs = Reranker.rerank(
            raw_candidates, min_score=self.similarity_threshold, top_n=max(k_val, 6)
        )

        # 4. Format context & citations
        context_str, citations = format_context(ranked_docs)

        if not ranked_docs:
            return {
                "answer": "I could not find any relevant information in the uploaded documents.",
                "citations": [],
                "model": self.llm.model,
                "is_grounded": False,
                "effective_query": clean_query,
            }

        # 5. Build prompt messages
        prompt_template = get_rag_prompt_template()
        messages = prompt_template.format_messages(context=context_str, question=clean_query)

        # 6. Generate answer
        raw_answer = await self.llm.agenerate(messages)

        # 7. Apply guardrails
        validated_answer, is_grounded = Guardrails.validate_and_filter(raw_answer, citations)

        return {
            "answer": validated_answer,
            "citations": citations,
            "model": self.llm.model,
            "is_grounded": is_grounded,
            "effective_query": clean_query,
        }

    def query_stream_sync(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[Any, List[Dict[str, Any]], str]:
        """
        Synchronously stream RAG tokens for Streamlit st.write_stream().

        Returns:
            Tuple of (token_iterator, citations, effective_query).
        """
        if not self.llm:
            raise ValueError("LLM is not initialized. Ensure GOOGLE_API_KEY is configured.")

        # 1. Transform & contextualize query
        if status_callback:
            status_callback("🧠 Step 1: Contextualizing query with conversation history...")
        effective_query = QueryTransformer.contextualize_query(
            question, chat_history=chat_history, llm=self.llm
        )

        # 2. Retrieve candidates
        if status_callback:
            target_info = f" in '{filter_dict['filename']}'" if (filter_dict and "filename" in filter_dict and filter_dict["filename"]) else ""
            status_callback(f"🔎 Step 2: Searching ChromaDB vector store & BM25 keyword index{target_info}...")
        k_val = top_k or self.top_k
        raw_candidates = self.retriever.retrieve(effective_query, k=k_val * 2, filter_dict=filter_dict)

        # 3. Rerank & deduplicate
        if status_callback:
            status_callback(f"📊 Step 3: Reranking {len(raw_candidates)} candidate chunks & verifying citations...")
        ranked_docs = Reranker.rerank(
            raw_candidates, min_score=self.similarity_threshold, top_n=max(k_val, 6)
        )

        # 4. Format context & citations
        context_str, citations = format_context(ranked_docs)

        if not ranked_docs:
            if status_callback:
                status_callback("⚠️ No matching passages found in the uploaded documents.")
            def empty_stream():
                yield "I could not find any relevant information in the uploaded documents."
            return empty_stream(), [], effective_query

        if status_callback:
            status_callback(f"✅ Step 4: Grounded context ready ({len(citations)} source passages) — starting generation...")

        # 5. Format prompt
        prompt_template = get_rag_prompt_template()
        messages = prompt_template.format_messages(context=context_str, question=effective_query)

        # 6. Stream tokens
        token_stream = self.llm.stream(messages)
        return token_stream, citations, effective_query

    async def query_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[AsyncIterator[str], List[Dict[str, Any]]]:
        """
        Stream RAG answer tokens asynchronously.
        """
        if not self.llm:
            raise ValueError("LLM is not initialized. Ensure GOOGLE_API_KEY is configured.")

        clean_query = QueryTransformer.contextualize_query(
            question, chat_history=chat_history, llm=self.llm
        )
        k_val = top_k or self.top_k
        raw_candidates = self.retriever.retrieve(clean_query, k=k_val * 2, filter_dict=filter_dict)

        ranked_docs = Reranker.rerank(
            raw_candidates, min_score=self.similarity_threshold, top_n=max(k_val, 6)
        )

        context_str, citations = format_context(ranked_docs)

        if not ranked_docs:
            async def empty_stream():
                yield "I could not find any relevant information in the uploaded documents."
            return empty_stream(), []

        prompt_template = get_rag_prompt_template()
        messages = prompt_template.format_messages(context=context_str, question=clean_query)

        token_stream = self.llm.astream(messages)
        return token_stream, citations

    def delete_document(self, filename: str) -> int:
        """
        Remove document chunks by filename.
        """
        return self.vector_store.delete_by_source(filename)

    def clear_all(self) -> int:
        """
        Remove all indexed documents and reset vector store.
        """
        return self.vector_store.clear_all()

    def list_documents(self) -> List[str]:
        """
        List distinct filenames indexed in the vector store.
        """
        return self.vector_store.list_sources()

    def count(self) -> int:
        """
        Total chunks indexed.
        """
        return self.vector_store.count()
