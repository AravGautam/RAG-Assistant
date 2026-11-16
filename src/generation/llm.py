"""
Google Gemini LLM wrapper supporting sync, async, and streaming responses with automatic quota fallback.
"""
import logging
from typing import List, Iterator, AsyncIterator, Optional, Any
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_MODEL = "gemini-flash-lite-latest"


def _extract_text(content: Any) -> str:
    """Safely extract plain string from langchain message content (handles str, list of dicts, or ContentBlock)."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts)
    elif hasattr(content, "text"):
        return str(content.text)
    return str(content)


class LLMError(Exception):
    """Exception raised when LLM generation fails."""
    pass


class GeminiLLM:
    """
    Interface for Google Gemini language models with full streaming capabilities and
    automatic model fallback on quota / 429 exhaustion.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-flash-lite-latest",
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = 2048,
    ) -> None:
        if not api_key:
            raise LLMError(
                "Google API Key is missing. Provide GOOGLE_API_KEY in .env or settings."
            )

        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        self._client = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        logger.info("Initialized GeminiLLM (model='%s', temp=%.2f)", self.model, self.temperature)

    def _get_fallback_client(self) -> ChatGoogleGenerativeAI:
        """Create a client for the high-quota fallback model."""
        fallback_model = DEFAULT_FALLBACK_MODEL if self.model != DEFAULT_FALLBACK_MODEL else "gemini-flash-latest"
        logger.warning("Switching model from '%s' to fallback '%s' due to quota limit", self.model, fallback_model)
        self.model = fallback_model
        self._client = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return self._client

    def generate(self, messages: List[BaseMessage]) -> str:
        """Execute synchronous generation with automatic quota fallback."""
        try:
            response = self._client.invoke(messages)
            return _extract_text(response.content)
        except Exception as e:
            err_str = str(e).lower()
            if "resource_exhausted" in err_str or "429" in err_str or "quota" in err_str:
                logger.warning("Primary model quota exhausted. Retrying with fallback model...")
                client = self._get_fallback_client()
                response = client.invoke(messages)
                return _extract_text(response.content)
            logger.error("Gemini generation error: %s", str(e))
            raise LLMError(f"Gemini generation failed: {str(e)}") from e

    async def agenerate(self, messages: List[BaseMessage]) -> str:
        """Execute asynchronous generation with automatic quota fallback."""
        try:
            response = await self._client.ainvoke(messages)
            return _extract_text(response.content)
        except Exception as e:
            err_str = str(e).lower()
            if "resource_exhausted" in err_str or "429" in err_str or "quota" in err_str:
                logger.warning("Primary model quota exhausted. Retrying with fallback model...")
                client = self._get_fallback_client()
                response = await client.ainvoke(messages)
                return _extract_text(response.content)
            logger.error("Gemini async generation error: %s", str(e))
            raise LLMError(f"Gemini async generation failed: {str(e)}") from e

    def stream(self, messages: List[BaseMessage]) -> Iterator[str]:
        """Synchronous token streaming generator with automatic quota fallback."""
        try:
            for chunk in self._client.stream(messages):
                text = _extract_text(chunk.content)
                if text:
                    yield text
        except Exception as e:
            err_str = str(e).lower()
            if "resource_exhausted" in err_str or "429" in err_str or "quota" in err_str:
                logger.warning("Primary model quota exhausted in stream. Switching to fallback...")
                client = self._get_fallback_client()
                for chunk in client.stream(messages):
                    text = _extract_text(chunk.content)
                    if text:
                        yield text
            else:
                logger.error("Gemini streaming error: %s", str(e))
                raise LLMError(f"Gemini streaming failed: {str(e)}") from e

    async def astream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        """Asynchronous token streaming generator for FastAPI SSE streaming."""
        try:
            async for chunk in self._client.astream(messages):
                text = _extract_text(chunk.content)
                if text:
                    yield text
        except Exception as e:
            err_str = str(e).lower()
            if "resource_exhausted" in err_str or "429" in err_str or "quota" in err_str:
                logger.warning("Primary model quota exhausted in astream. Switching to fallback...")
                client = self._get_fallback_client()
                async for chunk in client.astream(messages):
                    text = _extract_text(chunk.content)
                    if text:
                        yield text
            else:
                logger.error("Gemini async streaming error: %s", str(e))
                raise LLMError(f"Gemini async streaming failed: {str(e)}") from e
