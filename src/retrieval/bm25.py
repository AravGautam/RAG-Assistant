"""
BM25 (Best Matching 25) Keyword Retrieval Engine.
Provides deterministic lexical search to complement semantic vector search in Hybrid RAG.
"""
import math
import re
from typing import List, Tuple, Dict, Optional, Set, Any
from langchain_core.documents import Document


def _tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into clean lowercase words."""
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())


class BM25Retriever:
    """
    Self-contained in-memory Okapi BM25 indexer.
    Zero external heavy dependencies, high speed lexical search.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.documents: List[Document] = []
        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_lengths: List[int] = []
        self.doc_token_counts: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

    def index_documents(self, documents: List[Document]) -> None:
        """Build the BM25 inverted index from a list of Document objects."""
        self.documents = documents
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            self.avgdl = 0.0
            self.doc_lengths = []
            self.doc_token_counts = []
            self.idf = {}
            return

        total_length = 0
        self.doc_lengths = []
        self.doc_token_counts = []
        df: Dict[str, int] = {}

        for doc in documents:
            tokens = _tokenize(doc.page_content)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            # Frequency count for this doc
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.doc_token_counts.append(counts)

            # Document frequency
            for unique_token in counts.keys():
                df[unique_token] = df.get(unique_token, 0) + 1

        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

        # Calculate BM25 smoothed IDF for each term
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search documents by BM25 keyword match score.

        Returns:
            List of (Document, bm25_score) tuples sorted in descending order of score.
        """
        from typing import Any
        if self.corpus_size == 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        filter_filename = filter_dict.get("filename") if filter_dict else None

        scores: List[Tuple[int, float]] = []

        for idx, doc in enumerate(self.documents):
            # Apply filename filter if specified
            if filter_filename:
                doc_file = doc.metadata.get("filename") or doc.metadata.get("source")
                if doc_file != filter_filename:
                    continue

            doc_len = self.doc_lengths[idx]
            counts = self.doc_token_counts[idx]
            score = 0.0

            for q_term in query_tokens:
                if q_term not in counts:
                    continue
                tf = counts[q_term]
                idf = self.idf.get(q_term, 0.0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl or 1.0)))
                score += idf * (numerator / denominator)

            if score > 0.0:
                scores.append((idx, score))

        # Sort descending by BM25 score
        scores.sort(key=lambda x: x[1], reverse=True)

        return [(self.documents[idx], round(score, 4)) for idx, score in scores[:k]]
