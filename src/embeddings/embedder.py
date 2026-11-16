"""
Local Hugging Face embeddings wrapper.
"""
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings


class Embedder:
    """
    Singleton wrapper for free local Hugging Face Sentence-Transformers embeddings.
    Runs locally on CPU with zero API costs.
    """
    _instance: Optional["Embedder"] = None
    _embeddings: Optional[HuggingFaceEmbeddings] = None

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )

    @classmethod
    def get_instance(
        cls,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> "Embedder":
        """
        Thread-safe singleton getter to avoid reloading model weights.
        """
        if cls._instance is None:
            cls._instance = cls(model_name=model_name, device=device)
        return cls._instance

    @property
    def langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Expose underlying LangChain compatible HuggingFaceEmbeddings object.
        """
        return self._embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single query string.
        """
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors for a batch of text chunks.
        """
        return self._embeddings.embed_documents(texts)

    @property
    def dimension(self) -> int:
        """
        Embedding vector dimensionality (384 for all-MiniLM-L6-v2).
        """
        sample = self.embed_query("dimension_probe")
        return len(sample)

