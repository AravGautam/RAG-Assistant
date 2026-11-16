"""
Query transformation, normalization, and conversational contextualization utilities.
"""
import re
import logging
from typing import List, Dict, Optional, Any
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

FOLLOW_UP_INDICATORS = {
    "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "the latter", "the former", "the first", "the second", "the third",
    "first one", "second one", "third one", "what about", "how about",
    "tell me more", "explain more", "elaborate", "why so", "why is that",
    "give an example", "more details", "what else",
}


class QueryTransformer:
    """
    Sanitizes, normalizes, and contextualizes user queries with multi-turn chat history.
    """

    @classmethod
    def transform(cls, query: str) -> str:
        """
        Normalize and clean a query string.
        """
        if not query:
            return ""

        cleaned = query.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\?{2,}", "?", cleaned)
        cleaned = re.sub(r"!{2,}", "!", cleaned)
        return cleaned

    @classmethod
    def is_follow_up(cls, query: str) -> bool:
        """
        Check if a query is likely a conversational follow-up referencing earlier turns.
        """
        lower = query.lower().strip()
        tokens = set(re.findall(r"\b\w+\b", lower))

        if len(tokens) <= 4:
            return True

        for indicator in FOLLOW_UP_INDICATORS:
            if indicator in lower:
                return True
        return False

    @classmethod
    def contextualize_query(
        cls,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        llm: Optional[Any] = None,
    ) -> str:
        """
        Rewrite a conversational follow-up question into a standalone document search query
        using recent chat history.
        """
        cleaned_query = cls.transform(query)

        if not chat_history or len(chat_history) < 2 or not llm:
            return cleaned_query

        if not cls.is_follow_up(cleaned_query):
            return cleaned_query

        try:
            # Take last 4 turns (2 user, 2 assistant)
            recent_turns = chat_history[-4:]
            history_lines = []
            for turn in recent_turns:
                role = "User" if turn.get("role") == "user" else "Assistant"
                content = str(turn.get("content", ""))[:200]
                history_lines.append(f"{role}: {content}")
            history_str = "\n".join(history_lines)

            system_prompt = (
                "You are an expert search query reformulator for document retrieval. "
                "Given a short conversation history and a follow-up user query, rewrite the follow-up query "
                "into a single, standalone search query that includes all necessary context from earlier turns. "
                "DO NOT answer the question. Output ONLY the rewritten standalone question, nothing else."
            )
            user_prompt = f"Chat History:\n{history_str}\n\nFollow-up Question: {cleaned_query}\n\nStandalone Search Query:"

            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            rewritten = llm.generate(messages).strip()

            # Clean output
            rewritten = re.sub(r"^Standalone (Search )?Query:\s*", "", rewritten, flags=re.IGNORECASE)
            rewritten = rewritten.strip(' "\'')

            if rewritten and len(rewritten) > 3:
                logger.info("Contextualized query: '%s' -> '%s'", cleaned_query, rewritten)
                return rewritten
        except Exception as e:
            logger.warning("Query contextualization skipped due to error: %s", str(e))

        return cleaned_query
