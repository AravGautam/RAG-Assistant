"""
FastAPI dependency injection providers.
"""
from functools import lru_cache
from typing import Optional
from fastapi import Depends

from config.settings import Settings, get_settings
from src.embeddings.embedder import Embedder
from src.vectorstore.client import ChromaVectorStore
from src.generation.llm import GeminiLLM
from src.pipeline.rag_pipeline import RAGPipeline

# Module-level singletons
_VECTOR_STORE: Optional[ChromaVectorStore] = None
_PIPELINE: Optional[RAGPipeline] = None


def get_app_settings() -> Settings:
    """
    Dependency returning application settings.
    """
    return get_settings()


def get_embedder(settings: Settings = Depends(get_app_settings)) -> Embedder:
    """
    Dependency returning the shared Hugging Face Embedder instance.
    """
    return Embedder.get_instance(
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
    )


def get_vector_store(
    settings: Settings = Depends(get_app_settings),
    embedder: Embedder = Depends(get_embedder),
) -> ChromaVectorStore:
    """
    Dependency returning the shared ChromaVectorStore instance.
    """
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = ChromaVectorStore(
            persist_directory=settings.chroma_persist_directory,
            collection_name=settings.chroma_collection_name,
            embedder=embedder,
        )
    return _VECTOR_STORE


def get_llm(settings: Settings = Depends(get_app_settings)) -> Optional[GeminiLLM]:
    """
    Dependency returning initialized Gemini LLM if API key is present.
    """
    if not settings.google_api_key:
        return None
    return GeminiLLM(
        api_key=settings.google_api_key,
        model=settings.gemini_model,
        temperature=settings.gemini_temperature,
        max_output_tokens=settings.gemini_max_output_tokens,
    )


def get_pipeline(
    settings: Settings = Depends(get_app_settings),
    vector_store: ChromaVectorStore = Depends(get_vector_store),
    llm: Optional[GeminiLLM] = Depends(get_llm),
) -> RAGPipeline:
    """
    Dependency returning the unified RAGPipeline instance.
    """
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = RAGPipeline(
            vector_store=vector_store,
            llm=llm,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            top_k=settings.retrieval_top_k,
            similarity_threshold=settings.similarity_threshold,
        )
    else:
        # Update LLM if API key became available or changed
        _PIPELINE.llm = llm
    return _PIPELINE

