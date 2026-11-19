"""
Retriever component interfacing between query processing, vector store, and BM25 search.
Implements Hybrid Dense + Lexical Retrieval with Reciprocal Rank Fusion (RRF).
"""
import logging
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document

from src.vectorstore.base import BaseVectorStore
from src.retrieval.bm25 import BM25Retriever

logger = logging.getLogger(__name__)

SUMMARY_KEYWORDS = [
    "summar",
    "overview",
    "outline",
    "agenda",
    "main points",
    "key points",
    "key takeaways",
    "table of contents",
    "what is this pdf",
    "what is this document",
    "what is the document about",
    "what is the pdf about",
    "explain this pdf",
    "explain this document",
    "tell me about this",
    "what does this contain",
]


class HybridRetriever:
    """
    Hybrid Retriever combining:
    1. Dense Vector Similarity (ChromaDB + sentence-transformers)
    2. BM25 Lexical Keyword Matching
    3. Reciprocal Rank Fusion (RRF)
    4. Structural / Summary query detection
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        top_k: int = 4,
        score_threshold: float = 0.0,
    ) -> None:
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold
        self._bm25 = BM25Retriever()
        self._bm25_doc_count = -1

    def _sync_bm25_index(self, filter_dict: Optional[Dict[str, Any]] = None) -> None:
        """Ensure BM25 index reflects current vector store documents."""
        try:
            current_count = self.vector_store.count()
            if current_count != self._bm25_doc_count or current_count == 0:
                all_docs = self.vector_store.get_all_documents()
                self._bm25.index_documents(all_docs)
                self._bm25_doc_count = current_count
                logger.info("Synchronized BM25 index with %d documents", len(all_docs))
        except Exception as e:
            logger.warning("Failed to synchronize BM25 index: %s", str(e))

    @classmethod
    def is_summary_query(cls, query: str) -> bool:
        """
        Check if the user is asking for a summary, overview, or document-level explanation.
        """
        lower_query = query.lower().strip()
        for kw in SUMMARY_KEYWORDS:
            if kw in lower_query:
                return True
        return False

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid retrieval combining dense vector search and BM25 keyword search
        using Reciprocal Rank Fusion (RRF).
        """
        limit = k or top_k or self.top_k
        target_file = filter_dict.get("filename") if filter_dict else None

        results: List[Tuple[Document, float]] = []

        # 1. Structural / Summary Query Handling
        if self.is_summary_query(query):
            logger.info("Detected summary query: '%s'. Fetching initial document chunks.", query)
            initial_chunks = self.vector_store.get_initial_chunks(
                filename=target_file, max_chunks=6
            )
            results.extend(initial_chunks)

        # 2. Dense Vector Similarity Search
        try:
            dense_matches = self.vector_store.similarity_search_with_score(
                query=query,
                k=limit * 2,
                filter_dict=filter_dict,
            )
        except Exception as e:
            logger.warning("Dense vector search failed: %s", str(e))
            dense_matches = []

        # 3. BM25 Lexical Keyword Search
        self._sync_bm25_index(filter_dict)
        try:
            bm25_matches = self._bm25.search(
                query=query,
                k=limit * 2,
                filter_dict=filter_dict,
            )
        except Exception as e:
            logger.warning("BM25 lexical search failed: %s", str(e))
            bm25_matches = []

        # 4. Visual Cross-Modal CLIP Search
        visual_matches: List[Document] = []
        try:
            from src.embeddings.image_embedder import ImageEmbedder
            query_img_emb = ImageEmbedder.get_instance().embed_text(query)
            if query_img_emb:
                img_hits = self.vector_store.search_images(
                    query_embedding=query_img_emb,
                    k=limit,
                    filter_dict=filter_dict,
                )
                for hit in img_hits:
                    if hit.get("distance", 1.0) < 0.45:
                        meta = hit.get("metadata", {})
                        doc_text = hit.get("document", "") or f"Visual diagram / image in {meta.get('filename')} (Page {meta.get('page_number', 1)})"
                        visual_matches.append(Document(page_content=doc_text, metadata=meta))
        except Exception as e:
            logger.debug("Visual CLIP search skipped: %s", str(e))

        # 5. Reciprocal Rank Fusion (RRF)
        # RRF Score = sum(weight / (60 + rank))
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Dense rank scoring (weight = 1.0)
        for rank, (doc, _) in enumerate(dense_matches):
            doc_key = doc.page_content.strip()[:120]
            doc_map[doc_key] = doc
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + (1.0 / (60.0 + rank))

        # BM25 rank scoring (weight = 0.8)
        for rank, (doc, _) in enumerate(bm25_matches):
            doc_key = doc.page_content.strip()[:120]
            doc_map[doc_key] = doc
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + (0.8 / (60.0 + rank))

        # Visual CLIP rank scoring (weight = 1.0)
        for rank, doc in enumerate(visual_matches):
            doc_key = doc.page_content.strip()[:120]
            doc_map[doc_key] = doc
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + (1.0 / (60.0 + rank))

        # Merge ranked hybrid results
        sorted_hybrid = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        for doc_key, score in sorted_hybrid[:limit * 2]:
            # Convert RRF score into distance-like metric for downstream compatibility (smaller distance = better)
            # Distance approximation: max(0.0, 1.0 - (score * 30.0))
            sim_dist = max(0.01, round(1.0 - (score * 25.0), 4))
            results.append((doc_map[doc_key], sim_dist))

        logger.debug(
            "Hybrid retrieval returned %d total candidates (Dense: %d, BM25: %d)",
            len(results),
            len(dense_matches),
            len(bm25_matches),
        )
        return results


# Backward compatibility alias
VectorRetriever = HybridRetriever
