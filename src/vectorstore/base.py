"""
Abstract Base Class for Vector Stores.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document


class BaseVectorStore(ABC):
    """
    Abstract interface for vector database clients.
    """

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        """
        pass

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 4, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform vector similarity search.
        """
        pass

    @abstractmethod
    def similarity_search_with_score(
        self, query: str, k: int = 4, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform vector similarity search returning chunks and similarity distance/scores.
        """
        pass

    @abstractmethod
    def get_initial_chunks(
        self, filename: Optional[str] = None, max_chunks: int = 6
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve introductory/first chunks of a document for summary queries.
        """
        pass

    @abstractmethod
    def get_all_documents(self, filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Retrieve all document chunks, optionally filtered by metadata.
        """
        pass

    @abstractmethod
    def delete_by_source(self, filename: str) -> int:
        """
        Remove all document chunks associated with a specific filename.
        """
        pass

    @abstractmethod
    def clear_all(self) -> int:
        """
        Wipe all documents from the vector store.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Return the total number of chunks stored in the vector store.
        """
        pass

    @abstractmethod
    def list_sources(self) -> List[str]:
        """
        List distinct filenames indexed in the vector store.
        """
        pass

    def add_image_embedding(
        self,
        image_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        document_text: str = "",
    ) -> None:
        """Add a multimodal image embedding."""
        pass

    def search_images(
        self,
        query_embedding: List[float],
        k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search image embeddings."""
        return []

    def count_images(self, filename: Optional[str] = None) -> int:
        """Count image embeddings."""
        return 0
