"""
FastAPI Main Application entry point with Lifespan, CORS, and Routes.
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import Settings, get_settings
from config.logging_config import setup_logging
from src.api.schemas import HealthResponse
from src.api.routes import chat_router, documents_router
from src.api.dependencies import get_pipeline
from src.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifecycle management (startup and shutdown).
    """
    settings = get_settings()
    setup_logging(debug=settings.debug)

    logger.info("Starting up %s (v%s)...", settings.app_name, settings.app_version)
    logger.info("Using Embedding Model: '%s' (%s)", settings.embedding_model_name, settings.embedding_device)
    logger.info("Using LLM Model: '%s'", settings.gemini_model)
    logger.info("ChromaDB Persist Directory: '%s'", settings.chroma_persist_directory)

    yield

    logger.info("Shutting down %s...", settings.app_name)


def create_app() -> FastAPI:
    """
    Factory function to initialize and configure the FastAPI application.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Modular, production-ready Retrieval-Augmented Generation (RAG) API "
            "powered by free local Hugging Face embeddings and Google Gemini."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Enable CORS for frontend clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API v1 Routers
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Application health check",
    )
    @app.get(
        "/api/v1/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Application health check",
    )
    async def health_check(
        pipeline: RAGPipeline = Depends(get_pipeline),
        current_settings: Settings = Depends(get_settings),
    ) -> HealthResponse:
        """
        Return the health status, version, and current index metrics.
        """
        distinct_docs = pipeline.list_documents()
        total_chunks = pipeline.count()

        return HealthResponse(
            status="ok",
            app_name=current_settings.app_name,
            version=current_settings.app_version,
            total_chunks=total_chunks,
            distinct_documents=len(distinct_docs),
        )

    @app.get(
        "/",
        tags=["System"],
        summary="Root information endpoint",
    )
    async def root(current_settings: Settings = Depends(get_settings)) -> JSONResponse:
        """
        API landing page with interactive documentation links.
        """
        return JSONResponse(
            content={
                "name": current_settings.app_name,
                "version": current_settings.app_version,
                "status": "online",
                "docs_url": "/docs",
                "redoc_url": "/redoc",
                "health_url": "/health",
            }
        )

    return app


# Application entry point for ASGI servers (Uvicorn)
app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

