"""
Query transformation and hygiene utilities before vector search.
"""
import re


class QueryTransformer:
    """
    Sanitizes and normalizes incoming user queries before retrieval.
    """

    @classmethod
    def transform(cls, query: str) -> str:
        """
        Normalize and clean the query string.

        Args:
            query: Raw user input question.

        Returns:
            Sanitized query string.
        """
        if not query:
            return ""

        # Strip whitespace
        cleaned = query.strip()

        # Collapse redundant inner whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Remove repetitive punctuation (e.g. '???')
        cleaned = re.sub(r"\?{2,}", "?", cleaned)
        cleaned = re.sub(r"!{2,}", "!", cleaned)

        return cleaned

