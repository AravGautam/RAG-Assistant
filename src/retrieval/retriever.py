"""
Retriever component interfacing between query processing and vector store.
"""
import re
import logging
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document

from src.vectorstore.base import BaseVectorStore

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
]


class VectorRetriever:
    """
    Retrieves the most relevant document chunks from the vector store with
    intelligent broad-query and summarization awareness.
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
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve chunks with distance/relevance scores.

        For broad or summarization queries, automatically includes introductory chunks
        (e.g. title slide, agenda, abstract) to ensure a complete and grounded overview.
        """
        limit = k or self.top_k
        target_file = filter_dict.get("filename") if filter_dict else None

        results: List[Tuple[Document, float]] = []

        # If summary/overview request: pull initial structural chunks first
        if self.is_summary_query(query):
            logger.info("Detected summary query: '%s'. Fetching initial document chunks.", query)
            initial_chunks = self.vector_store.get_initial_chunks(
                filename=target_file, max_chunks=6
            )
            results.extend(initial_chunks)

        # Standard vector similarity search
        vector_matches = self.vector_store.similarity_search_with_score(
            query=query,
            k=limit,
            filter_dict=filter_dict,
        )
        results.extend(vector_matches)

        logger.debug("Retrieved %d candidates for query: '%s'", len(results), query[:50])
        return results
