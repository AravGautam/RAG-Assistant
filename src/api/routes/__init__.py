"""
Routes package exports.
"""
from src.api.routes.chat import router as chat_router
from src.api.routes.documents import router as documents_router

__all__ = ["chat_router", "documents_router"]

