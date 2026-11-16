"""
Chat routes for synchronous and streaming RAG question-answering.
"""
import json
import logging
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from src.api.schemas import ChatRequest, ChatResponse, Citation
from src.api.dependencies import get_pipeline
from src.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Query the RAG system",
    description="Ask a question against all indexed documents with verified citations.",
)
async def chat(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> ChatResponse:
    """
    Standard synchronous question-answering endpoint.
    """
    if not pipeline.llm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google Gemini API Key is not configured. Set GOOGLE_API_KEY in .env.",
        )

    try:
        result = await pipeline.query(
            question=request.query,
            top_k=request.top_k,
            filter_dict=request.filter,
        )

        citations_pydantic = [Citation(**cite) for cite in result["citations"]]

        return ChatResponse(
            answer=result["answer"],
            citations=citations_pydantic,
            model=result["model"],
            is_grounded=result["is_grounded"],
        )
    except Exception as e:
        logger.error("Chat generation failed: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}",
        )


@router.post(
    "/stream",
    summary="Stream RAG answer tokens",
    description="Streams answer tokens in real-time via Server-Sent Events (SSE).",
)
async def chat_stream(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> StreamingResponse:
    """
    Server-Sent Events (SSE) streaming endpoint for real-time token delivery.
    """
    if not pipeline.llm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google Gemini API Key is not configured. Set GOOGLE_API_KEY in .env.",
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            token_stream, citations = await pipeline.query_stream(
                question=request.query,
                top_k=request.top_k,
                filter_dict=request.filter,
            )

            # Stream text tokens
            async for token in token_stream:
                payload = json.dumps({"token": token})
                yield f"event: token\ndata: {payload}\n\n"

            # Stream citations at the end of the response
            citations_payload = json.dumps({"citations": citations})
            yield f"event: citations\ndata: {citations_payload}\n\n"

            # End of stream indicator
            yield "event: done\ndata: [DONE]\n\n"

        except Exception as e:
            logger.error("Stream generation failed: %s", str(e))
            error_payload = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )

