"""
Health check endpoints
"""

from fastapi import APIRouter
from datetime import datetime

from core.config import settings
from core.logging import get_logger
from core.responses import success_response, error_response
from src.core.models import HealthResponse

logger = get_logger(__name__)
router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint - same as health check for compatibility."""
    return await health_check()


@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    health_response = HealthResponse(
        status="healthy",
        timestamp=datetime.now().timestamp(),
        version=settings.app_version,
        message="Arthur image recognition service is running",
    )

    return success_response(
        message=health_response.message,
        health=health_response.dict(),
    )
