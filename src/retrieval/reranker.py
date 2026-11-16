"""
Reranking and score normalization for candidate document chunks.
"""
from typing import List, Tuple
from langchain_core.documents import Document


class Reranker:
    """
    Reranks, deduplicates, and filters retrieved candidates.
    """

    @classmethod
    def rerank(
        cls,
        candidates: List[Tuple[Document, float]],
        min_score: float = 0.0,
        top_n: int = 4,
    ) -> List[Tuple[Document, float]]:
        """
        Deduplicate and normalize candidates based on relevance score.

        Args:
            candidates: List of (Document, raw_score) from retriever.
            min_score: Minimum normalized score threshold (0.0 to 1.0).
            top_n: Maximum number of chunks to retain.

        Returns:
            Sorted and filtered list of (Document, normalized_score) tuples.
        """
        if not candidates:
            return []

        # Deduplicate based on chunk content
        seen_texts = set()
        deduped: List[Tuple[Document, float]] = []

        for doc, raw_score in candidates:
            content_key = doc.page_content.strip()[:100]
            if content_key in seen_texts:
                continue
            seen_texts.add(content_key)

            # Chroma returns L2/cosine distance (smaller is better).
            # Convert to similarity score in [0, 1] range: 1 / (1 + distance)
            sim_score = round(1.0 / (1.0 + max(0.0, float(raw_score))), 4)

            if sim_score >= min_score:
                deduped.append((doc, sim_score))

        # Sort descending by normalized similarity score
        deduped.sort(key=lambda x: x[1], reverse=True)

        return deduped[:top_n]

