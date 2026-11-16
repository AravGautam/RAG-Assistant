"""
Google Gemini LLM wrapper supporting sync, async, and streaming responses.
"""
import logging
from typing import List, Iterator, AsyncIterator, Optional, Any
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Exception raised when LLM generation fails."""
    pass


class GeminiLLM:
    """
    Interface for Google Gemini language models with full streaming capabilities.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = 2048,
    ) -> None:
        if not api_key:
            raise LLMError(
                "Google API Key is missing. Provide GOOGLE_API_KEY in .env or settings."
            )

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        self._client = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        logger.info("Initialized GeminiLLM (model='%s', temp=%.2f)", self.model, self.temperature)

    def generate(self, messages: List[BaseMessage]) -> str:
        """
        Execute synchronous generation.
        """
        try:
            response = self._client.invoke(messages)
            return str(response.content)
        except Exception as e:
            logger.error("Gemini generation error: %s", str(e))
            raise LLMError(f"Gemini generation failed: {str(e)}") from e

    async def agenerate(self, messages: List[BaseMessage]) -> str:
        """
        Execute asynchronous generation.
        """
        try:
            response = await self._client.ainvoke(messages)
            return str(response.content)
        except Exception as e:
            logger.error("Gemini async generation error: %s", str(e))
            raise LLMError(f"Gemini async generation failed: {str(e)}") from e

    def stream(self, messages: List[BaseMessage]) -> Iterator[str]:
        """
        Synchronous token streaming generator.
        """
        try:
            for chunk in self._client.stream(messages):
                if chunk.content:
                    yield str(chunk.content)
        except Exception as e:
            logger.error("Gemini streaming error: %s", str(e))
            raise LLMError(f"Gemini streaming failed: {str(e)}") from e

    async def astream(self, messages: List[BaseMessage]) -> AsyncIterator[str]:
        """
        Asynchronous token streaming generator for FastAPI SSE streaming.
        """
        try:
            async for chunk in self._client.astream(messages):
                if chunk.content:
                    yield str(chunk.content)
        except Exception as e:
            logger.error("Gemini async streaming error: %s", str(e))
            raise LLMError(f"Gemini async streaming failed: {str(e)}") from e

