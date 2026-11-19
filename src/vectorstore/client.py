"""
ChromaDB persistent vector store implementation.
"""
import os
import logging
from typing import List, Tuple, Dict, Any, Optional, Set
import chromadb
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from src.vectorstore.base import BaseVectorStore
from src.embeddings.embedder import Embedder

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    Production-grade ChromaDB vector store with disk persistence.
    Supports dual collections: text document chunks and multimodal CLIP image embeddings.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "rag_documents",
        embedder: Optional[Embedder] = None,
        image_collection_name: str = "rag_images",
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.image_collection_name = image_collection_name
        self.embedder = embedder or Embedder.get_instance()

        os.makedirs(self.persist_directory, exist_ok=True)

        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedder.langchain_embeddings,
            persist_directory=self.persist_directory,
        )

        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self._image_col = self._client.get_or_create_collection(
            name=self.image_collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(
            "Initialized ChromaVectorStore (text_col='%s', img_col='%s', dir='%s', text_chunks=%d, images=%d)",
            self.collection_name,
            self.image_collection_name,
            self.persist_directory,
            self.count(),
            self.count_images(),
        )

    @property
    def raw_vectorstore(self) -> Chroma:
        """
        Expose underlying LangChain Chroma instance.
        """
        self._ensure_collection_valid()
        return self._vectorstore

    def _ensure_collection_valid(self) -> None:
        """Reconnect to Chroma collections if deleted, stale, or recreated."""
        try:
            self._vectorstore._collection.count()
        except Exception:
            try:
                self._vectorstore = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedder.langchain_embeddings,
                    persist_directory=self.persist_directory,
                )
            except Exception as e:
                logger.warning("Could not re-initialize Chroma text collection: %s", str(e))

        try:
            self._image_col.count()
        except Exception:
            try:
                self._client = chromadb.PersistentClient(path=self.persist_directory)
                self._image_col = self._client.get_or_create_collection(
                    name=self.image_collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                logger.warning("Could not re-initialize Chroma image collection: %s", str(e))

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to Chroma with metadata. Automatically purges existing chunks
        for the same filename first to prevent duplicate entries.
        """
        if not documents:
            return []

        self._ensure_collection_valid()
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
        self._ensure_collection_valid()
        return self._vectorstore.similarity_search(query=query, k=k, filter=filter_dict)

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for top-k similar chunks with cosine distance scores.
        """
        self._ensure_collection_valid()
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
        self._ensure_collection_valid()
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

    def get_all_documents(self, filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Retrieve all document chunks from the collection as LangChain Document objects.
        """
        self._ensure_collection_valid()
        try:
            records = self._vectorstore._collection.get(
                where=filter_dict,
                include=["documents", "metadatas"],
            )
            raw_docs = records.get("documents", [])
            raw_metas = records.get("metadatas", [])
            return [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(raw_docs, raw_metas)
            ]
        except Exception as e:
            logger.error("Failed to retrieve all documents: %s", str(e))
            return []

    def add_image_embedding(
        self,
        image_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        document_text: str = "",
    ) -> None:
        """
        Store a CLIP image embedding along with metadata and optional caption/transcription text.
        """
        self._ensure_collection_valid()
        try:
            self._image_col.upsert(
                ids=[image_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[document_text] if document_text else ["visual_asset"],
            )
            logger.info("Indexed image embedding '%s' (filename='%s')", image_id, metadata.get("filename"))
        except Exception as e:
            logger.error("Failed to add image embedding '%s': %s", image_id, str(e))

    def search_images(
        self,
        query_embedding: List[float],
        k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the image collection using a CLIP cross-modal text or image embedding.
        Returns list of dicts with keys: 'id', 'metadata', 'distance', 'document'.
        """
        self._ensure_collection_valid()
        try:
            if self.count_images() == 0:
                return []

            where_filter = filter_dict if filter_dict else None
            results = self._image_col.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter,
                include=["metadatas", "distances", "documents"],
            )

            hits: List[Dict[str, Any]] = []
            if results and results.get("ids") and results["ids"][0]:
                ids = results["ids"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
                dists = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
                docs = results["documents"][0] if results.get("documents") else [""] * len(ids)

                for img_id, meta, dist, doc_str in zip(ids, metas, dists, docs):
                    hits.append({
                        "id": img_id,
                        "metadata": meta or {},
                        "distance": float(dist),
                        "document": doc_str,
                    })

            return hits
        except Exception as e:
            logger.warning("Visual image search failed: %s", str(e))
            return []

    def count_images(self, filename: Optional[str] = None) -> int:
        """
        Count the number of indexed image embeddings (optionally filtered by filename).
        """
        self._ensure_collection_valid()
        try:
            if filename:
                recs = self._image_col.get(where={"filename": filename})
                return len(recs.get("ids", []))
            return self._image_col.count()
        except Exception:
            return 0

    def delete_by_source(self, filename: str) -> int:
        """
        Remove all text chunks and image embeddings associated with a specific file.
        """
        self._ensure_collection_valid()
        deleted_count = 0
        try:
            records = self._vectorstore._collection.get(
                where={"filename": filename},
                include=["metadatas"],
            )
            ids_to_delete = records.get("ids", [])
            if ids_to_delete:
                self._vectorstore._collection.delete(ids=ids_to_delete)
                logger.info("Deleted %d text chunks for file '%s'", len(ids_to_delete), filename)
                deleted_count += len(ids_to_delete)
        except Exception as e:
            logger.error("Error deleting text chunks for '%s': %s", filename, str(e))

        try:
            img_records = self._image_col.get(where={"filename": filename})
            img_ids = img_records.get("ids", [])
            if img_ids:
                self._image_col.delete(ids=img_ids)
                logger.info("Deleted %d image embeddings for file '%s'", len(img_ids), filename)
        except Exception as e:
            logger.warning("Error deleting image embeddings for '%s': %s", filename, str(e))

        return deleted_count

    def clear_all(self) -> int:
        """
        Remove all records and wipe the vector store completely without breaking collection references.
        """
        self._ensure_collection_valid()
        deleted_count = 0
        try:
            records = self._vectorstore._collection.get()
            ids = records.get("ids", [])
            if ids:
                self._vectorstore._collection.delete(ids=ids)
            logger.info("Cleared text vector store (%d chunks removed)", len(ids))
            deleted_count += len(ids)
        except Exception as e:
            logger.error("Error clearing text vector store: %s", str(e))

        try:
            img_records = self._image_col.get()
            img_ids = img_records.get("ids", [])
            if img_ids:
                self._image_col.delete(ids=img_ids)
            logger.info("Cleared image vector store (%d images removed)", len(img_ids))
        except Exception as e:
            logger.warning("Error clearing image vector store: %s", str(e))

        return deleted_count

    def count(self) -> int:
        """
        Total number of chunks in the collection.
        """
        self._ensure_collection_valid()
        try:
            return self._vectorstore._collection.count()
        except Exception:
            return 0

    def list_sources(self) -> List[str]:
        """
        List distinct filenames currently indexed.
        """
        self._ensure_collection_valid()
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
