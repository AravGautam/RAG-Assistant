"""
Document chunking strategies with metadata enrichment.
"""
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """
    Splits documents into overlapping chunks with rich metadata.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def split_documents(
        self, documents: List[Document]
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Split a list of documents into chunks and enrich metadata.

        Args:
            documents: List of cleaned Document objects.

        Returns:
            Tuple of (chunked Document list, statistics dictionary).
        """
        if not documents:
            return [], {"total_chunks": 0, "total_characters": 0}

        raw_chunks = self._splitter.split_documents(documents)
        enriched_chunks: List[Document] = []

        total_chars = 0
        filename = documents[0].metadata.get("filename", "unknown")

        for idx, chunk in enumerate(raw_chunks):
            char_count = len(chunk.page_content)
            total_chars += char_count

            page_num = chunk.metadata.get("page_number", 1)
            chunk_id = f"{filename}_p{page_num}_c{idx + 1}"

            # Merge existing metadata with chunk tracking info
            chunk.metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": idx + 1,
                "total_chunks_in_doc": len(raw_chunks),
                "char_count": char_count,
            })
            enriched_chunks.append(chunk)

        stats = {
            "filename": filename,
            "total_chunks": len(enriched_chunks),
            "total_characters": total_chars,
            "avg_chunk_size": int(total_chars / len(enriched_chunks)) if enriched_chunks else 0,
        }

        return enriched_chunks, stats

