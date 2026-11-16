"""
Document management routes for uploading, listing, and deleting documents.
"""
import os
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Query, HTTPException, status

from src.api.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
)
from src.api.dependencies import get_pipeline
from src.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".text"}
RAW_DATA_DIR = Path("data/raw")


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
    description="Upload a PDF, Markdown, or TXT file to be chunked and indexed in ChromaDB.",
)
async def upload_document(
    file: UploadFile = File(...),
    clear_existing: bool = Query(
        False, description="If True, clear all existing documents before indexing this one."
    ),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> DocumentUploadResponse:
    """
    Ingest a document file, clean it, chunk it, and index it into the vector store.
    """
    filename = file.filename or "uploaded_document"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{filename}' is empty.",
            )

        # Save raw copy in data/raw/
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        raw_file_path = RAW_DATA_DIR / filename
        with open(raw_file_path, "wb") as f:
            f.write(content)

        # Index via pipeline
        stats = pipeline.index_document(
            file_bytes=content,
            filename=filename,
            clear_existing=clear_existing,
        )

        return DocumentUploadResponse(
            filename=filename,
            total_pages=stats.get("total_pages", 1),
            total_chunks=stats.get("total_chunks", 0),
            total_characters=stats.get("total_characters", 0),
            message=f"Successfully indexed {stats.get('total_chunks', 0)} chunks from '{filename}'.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process document '%s': %s", filename, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}",
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List indexed documents",
    description="Retrieve all document filenames and chunk statistics currently stored in ChromaDB.",
)
async def list_documents(
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> DocumentListResponse:
    """
    Get the list of active documents and total chunk count in the collection.
    """
    try:
        documents = pipeline.list_documents()
        total_chunks = pipeline.count()

        return DocumentListResponse(
            documents=documents,
            total_documents=len(documents),
            total_chunks=total_chunks,
        )
    except Exception as e:
        logger.error("Failed to list documents: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query indexed documents: {str(e)}",
        )


@router.delete(
    "/clear-all",
    summary="Clear all indexed documents",
    description="Wipe all vector records from the database.",
)
async def clear_all_documents(
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> dict:
    """
    Wipe all documents from the vector database.
    """
    try:
        deleted = pipeline.clear_all()
        return {"status": "success", "chunks_cleared": deleted, "message": "All documents cleared."}
    except Exception as e:
        logger.error("Failed to clear vector store: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear vector store: {str(e)}",
        )


@router.delete(
    "/{filename}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document",
    description="Remove all vector chunks and raw file copies of the specified document.",
)
async def delete_document(
    filename: str,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> DocumentDeleteResponse:
    """
    Remove all vector records for a specific document filename.
    """
    try:
        deleted_count = pipeline.delete_document(filename)

        raw_path = RAW_DATA_DIR / filename
        if raw_path.exists():
            try:
                os.remove(raw_path)
            except OSError:
                pass

        return DocumentDeleteResponse(
            filename=filename,
            chunks_deleted=deleted_count,
            message=f"Successfully deleted {deleted_count} chunks for '{filename}'.",
        )
    except Exception as e:
        logger.error("Failed to delete document '%s': %s", filename, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}",
        )
