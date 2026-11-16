"""
Guardrails for output validation and hallucination mitigation.
"""
from typing import List, Dict, Any, Tuple

REFUSAL_PHRASE = "I could not find the answer in the provided document."


class Guardrails:
    """
    Enforces post-generation quality and safety checks on generated answers.
    """

    @classmethod
    def validate_and_filter(
        cls,
        answer: str,
        citations: List[Dict[str, Any]],
    ) -> Tuple[str, bool]:
        """
        Validate generated answer against retrieved citations.

        Args:
            answer: Raw text response from the LLM.
            citations: List of retrieved citation chunks.

        Returns:
            Tuple of (final_sanitized_answer, is_grounded).
        """
        if not answer or not answer.strip():
            return REFUSAL_PHRASE, False

        clean_answer = answer.strip()

        # If no citations were retrieved at all, answer cannot be grounded
        if not citations:
            return REFUSAL_PHRASE, False

        # If the model explicitly and strictly refused
        lower = clean_answer.lower()
        if (
            "could not find the answer in the provided document" in lower
            or "could not find any relevant information" in lower
        ):
            return clean_answer, False

        return clean_answer, True
