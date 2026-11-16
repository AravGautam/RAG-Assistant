"""
Application settings using Pydantic v2 Settings.
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application settings loaded from environment variables or .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application Configuration
    app_name: str = "Modular Production RAG Application"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # Google Gemini LLM Settings
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.2
    gemini_max_output_tokens: Optional[int] = 2048

    # Free Local Hugging Face Embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # ChromaDB Vector Store
    chroma_persist_directory: str = "data/chroma_db"
    chroma_collection_name: str = "rag_documents"

    # Chunking & Document Ingestion
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval Configuration
    retrieval_top_k: int = 4
    similarity_threshold: float = 0.0


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached instance of the application settings.
    """
    return Settings()

