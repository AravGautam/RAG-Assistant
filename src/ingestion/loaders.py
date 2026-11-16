"""
Document loaders supporting PDF, Markdown, and plain text.
"""
import io
import os
from pathlib import Path
from typing import List, Optional
from pypdf import PdfReader
from langchain_core.documents import Document


class DocumentLoaderError(Exception):
    """Exception raised when document loading fails."""
    pass


class DocumentLoader:
    """
    Unified document loader for PDF, Markdown, and plain text files.
    """

    @classmethod
    def load_from_bytes(cls, file_bytes: bytes, filename: str) -> List[Document]:
        """
        Load documents from raw in-memory bytes based on file extension.

        Args:
            file_bytes: Raw bytes of the document.
            filename: Original filename with extension.

        Returns:
            List of LangChain Document objects with page content and metadata.
        """
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            return cls._load_pdf_from_bytes(file_bytes, filename)
        elif ext in [".md", ".markdown"]:
            return cls._load_text_from_bytes(file_bytes, filename, doc_type="markdown")
        elif ext in [".txt", ".text"]:
            return cls._load_text_from_bytes(file_bytes, filename, doc_type="text")
        else:
            raise DocumentLoaderError(f"Unsupported file format: '{ext}'. Allowed: .pdf, .md, .txt")

    @classmethod
    def load_from_path(cls, file_path: str) -> List[Document]:
        """
        Load documents from a filesystem path.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            List of LangChain Document objects.
        """
        path = Path(file_path)
        if not path.is_file():
            raise DocumentLoaderError(f"File not found: {file_path}")

        with open(path, "rb") as f:
            file_bytes = f.read()

        return cls.load_from_bytes(file_bytes, path.name)

    @classmethod
    def _load_pdf_from_bytes(cls, file_bytes: bytes, filename: str) -> List[Document]:
        """
        Extract text page-by-page from PDF bytes using pypdf.
        """
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            documents: List[Document] = []

            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": filename,
                                "filename": filename,
                                "page_number": page_idx + 1,
                                "total_pages": len(reader.pages),
                                "doc_type": "pdf",
                            },
                        )
                    )

            if not documents:
                raise DocumentLoaderError(
                    f"No extractable text found in '{filename}'. It may be a scanned image or protected PDF."
                )

            return documents

        except Exception as e:
            if isinstance(e, DocumentLoaderError):
                raise
            raise DocumentLoaderError(f"Failed to parse PDF '{filename}': {str(e)}") from e

    @classmethod
    def _load_text_from_bytes(
        cls, file_bytes: bytes, filename: str, doc_type: str
    ) -> List[Document]:
        """
        Load plain text or markdown from bytes.
        """
        try:
            text = file_bytes.decode("utf-8", errors="replace")
            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "filename": filename,
                        "page_number": 1,
                        "total_pages": 1,
                        "doc_type": doc_type,
                    },
                )
            ]
        except Exception as e:
            raise DocumentLoaderError(f"Failed to read text file '{filename}': {str(e)}") from e

