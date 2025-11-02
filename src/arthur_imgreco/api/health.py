"""
Health check endpoints
"""

from fastapi import APIRouter, status
from pydantic import BaseModel
from datetime import datetime

from arthur_imgreco.core.config import settings
from arthur_imgreco.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime
    version: str
    message: str


class DetailedHealthResponse(BaseModel):
    """Detailed health check response model."""

    status: str
    timestamp: datetime
    version: str
    services: dict
    message: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        message="Arthur Image Recognition service is running",
    )


@router.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Root endpoint - same as health check for compatibility."""
    return await health_check()


@router.get("/api/health", response_model=DetailedHealthResponse)
async def detailed_health_check() -> DetailedHealthResponse:
    """Detailed health check with service status."""

    services = {}

    # TODO: Add actual service health checks
    services["database"] = "unknown"  # PostgreSQL connection
    services["vector_db"] = "unknown"  # Qdrant connection
    services["cache"] = "unknown"  # Redis connection
    services["ml_model"] = "unknown"  # CLIP model status

    overall_status = "healthy"  # TODO: Determine based on service checks

    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        services=services,
        message="Service health check completed",
    )
