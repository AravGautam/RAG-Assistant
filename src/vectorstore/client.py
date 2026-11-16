"""
ChromaDB persistent vector store implementation.
"""
import os
import logging
from typing import List, Tuple, Dict, Any, Optional, Set
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from src.vectorstore.base import BaseVectorStore
from src.embeddings.embedder import Embedder

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    Production-grade ChromaDB vector store with disk persistence.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "rag_documents",
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedder = embedder or Embedder.get_instance()

        os.makedirs(self.persist_directory, exist_ok=True)

        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedder.langchain_embeddings,
            persist_directory=self.persist_directory,
        )
        logger.info(
            "Initialized ChromaVectorStore (collection='%s', dir='%s', chunks=%d)",
            self.collection_name,
            self.persist_directory,
            self.count(),
        )

    @property
    def raw_vectorstore(self) -> Chroma:
        """
        Expose underlying LangChain Chroma instance.
        """
        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to Chroma with metadata. Automatically purges existing chunks
        for the same filename first to prevent duplicate entries.
        """
        if not documents:
            return []

        filename = documents[0].metadata.get("filename")
        if filename:
            # Purge previous version of this document if present
            self.delete_by_source(filename)

        doc_ids = self._vectorstore.add_documents(documents)
        logger.info("Indexed %d chunks into collection '%s'", len(doc_ids), self.collection_name)
        return doc_ids

    def similarity_search(
        self, query: str, k: int = 4, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for top-k similar chunks.
        """
        return self._vectorstore.similarity_search(query=query, k=k, filter=filter_dict)

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for top-k similar chunks with cosine distance scores.
        """
        return self._vectorstore.similarity_search_with_score(
            query=query, k=k, filter=filter_dict
        )

    def get_initial_chunks(
        self, filename: Optional[str] = None, max_chunks: int = 6
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve introductory/first chunks of a document sorted by page and chunk index.
        Ideal for document summarization, agendas, and overviews.
        """
        try:
            where_filter = {"filename": filename} if filename else None
            records = self._vectorstore._collection.get(
                where=where_filter,
                include=["documents", "metadatas"],
            )

            raw_docs = records.get("documents", [])
            raw_metas = records.get("metadatas", [])

            if not raw_docs:
                return []

            parsed_chunks: List[Tuple[int, int, Document]] = []
            for text, meta in zip(raw_docs, raw_metas):
                page = int(meta.get("page_number", meta.get("page", 1)))
                chunk_idx = int(meta.get("chunk_index", 1))
                doc = Document(page_content=text, metadata=meta or {})
                parsed_chunks.append((page, chunk_idx, doc))

            # Sort by page number, then chunk index
            parsed_chunks.sort(key=lambda x: (x[0], x[1]))

            # Return top initial chunks with 0.0 distance (max relevance)
            return [(item[2], 0.0) for item in parsed_chunks[:max_chunks]]

        except Exception as e:
            logger.error("Failed to retrieve initial chunks: %s", str(e))
            return []

    def delete_by_source(self, filename: str) -> int:
        """
        Remove all chunks associated with a specific file.
        """
        try:
            records = self._vectorstore._collection.get(
                where={"filename": filename},
                include=["metadatas"],
            )
            ids_to_delete = records.get("ids", [])
            if ids_to_delete:
                self._vectorstore._collection.delete(ids=ids_to_delete)
                logger.info("Deleted %d chunks for file '%s'", len(ids_to_delete), filename)
                return len(ids_to_delete)
            return 0
        except Exception as e:
            logger.error("Error deleting chunks for '%s': %s", filename, str(e))
            return 0

    def clear_all(self) -> int:
        """
        Remove all records and wipe the vector store completely.
        """
        try:
            total = self.count()
            self._vectorstore.delete_collection()
            # Recreate empty collection
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedder.langchain_embeddings,
                persist_directory=self.persist_directory,
            )
            logger.info("Cleared entire vector store (%d chunks removed)", total)
            return total
        except Exception as e:
            logger.error("Error clearing vector store: %s", str(e))
            return 0

    def count(self) -> int:
        """
        Total number of chunks in the collection.
        """
        try:
            return self._vectorstore._collection.count()
        except Exception:
            return 0

    def list_sources(self) -> List[str]:
        """
        List distinct filenames currently indexed.
        """
        try:
            records = self._vectorstore._collection.get(include=["metadatas"])
            metadatas = records.get("metadatas", [])
            distinct_sources: Set[str] = set()
            for meta in metadatas:
                if meta and "filename" in meta:
                    distinct_sources.add(meta["filename"])
            return sorted(list(distinct_sources))
        except Exception as e:
            logger.error("Failed to list sources: %s", str(e))
            return []
