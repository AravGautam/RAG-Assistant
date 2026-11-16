"""
Pydantic v2 request and response schemas.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request model for asking questions to the RAG system.
    """
    query: str = Field(
        ...,
        min_length=1,
        description="The question to ask about the indexed documents.",
        examples=["What is the main architecture described in the document?"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional override for the number of chunks to retrieve.",
    )
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filter (e.g. {'filename': 'report.pdf'}).",
    )


class Citation(BaseModel):
    """
    Metadata citation identifying the source passage used for generation.
    """
    source_id: int
    filename: str
    page: int
    chunk_index: int
    content: str
    score: float


class ChatResponse(BaseModel):
    """
    Response model containing generated answer and verified source citations.
    """
    answer: str
    citations: List[Citation]
    model: str
    is_grounded: bool


class DocumentUploadResponse(BaseModel):
    """
    Response model returned after indexing a document.
    """
    filename: str
    total_pages: int
    total_chunks: int
    total_characters: int
    message: str


class DocumentListResponse(BaseModel):
    """
    Response model listing all documents indexed in the vector store.
    """
    documents: List[str]
    total_documents: int
    total_chunks: int


class DocumentDeleteResponse(BaseModel):
    """
    Response model returned when deleting a document.
    """
    filename: str
    chunks_deleted: int
    message: str


class HealthResponse(BaseModel):
    """
    Health check response model.
    """
    status: str
    app_name: str
    version: str
    total_chunks: int
    distinct_documents: int

