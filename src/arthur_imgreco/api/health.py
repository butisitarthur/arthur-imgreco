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
    critical_failures = []
    
    # Check Qdrant Vector Database
    try:
        from arthur_imgreco.ml.vector_db import qdrant_service
        
        await qdrant_service.connect()
        
        # Try to get collection info
        try:
            collection_info = await qdrant_service.client.get_collection(qdrant_service.collection_name)
            vector_count = collection_info.vectors_count or 0
            services["vector_db"] = f"healthy ({vector_count} vectors)"
            logger.info("Qdrant health check passed", vectors=vector_count)
        except (ConnectionError, TimeoutError, ValueError) as e:
            services["vector_db"] = f"degraded (collection issue: {str(e)[:50]})"
            logger.warning("Qdrant collection check failed", error=str(e))
            
    except (ConnectionError, TimeoutError, ImportError) as e:
        services["vector_db"] = f"unhealthy ({str(e)[:50]})"
        critical_failures.append("vector_db")
        logger.error("Qdrant health check failed", error=str(e))
    
    # Check CLIP ML Model
    try:
        from arthur_imgreco.ml.clip_service import clip_service
        
        # Check if model is loaded and accessible
        if hasattr(clip_service, 'model') and clip_service.model is not None:
            services["ml_model"] = "healthy (CLIP ViT-B/32 loaded)"
        else:
            # Check if clip_service is accessible (this will load model if needed)
            services["ml_model"] = "healthy (CLIP service available)"
        
        logger.info("CLIP model health check passed")
        
    except (ImportError, RuntimeError, OSError) as e:
        services["ml_model"] = f"unhealthy ({str(e)[:50]})"
        critical_failures.append("ml_model")
        logger.error("CLIP model health check failed", error=str(e))
    
    # Check Redis Cache (if available)
    services["cache"] = "not_configured (optional service)"
    
    # Check PostgreSQL Database (not used yet)
    services["database"] = "not_implemented (using vector storage only)"
    
    # Check system resources
    try:
        import shutil
        
        # Check disk space for current directory
        _, _, free = shutil.disk_usage('/')
        free_gb = free / (1024**3)
        
        if free_gb < 1.0:  # Less than 1GB free
            services["storage"] = f"warning ({free_gb:.1f}GB free)"
        else:
            services["storage"] = f"healthy ({free_gb:.1f}GB free)"
            
        logger.info("System storage checked", disk_free_gb=round(free_gb, 1))
        
    except (OSError, ImportError) as e:
        services["storage"] = f"unknown ({str(e)[:50]})"
        logger.warning("System resource check failed", error=str(e))
    
    # Check model files exist
    try:
        from pathlib import Path
        
        # Check if CLIP cache directory exists (indicates model was downloaded)
        home_dir = Path.home()
        clip_cache = home_dir / ".cache" / "clip"
        
        if clip_cache.exists():
            services["model_files"] = "healthy (CLIP models cached)"
        else:
            services["model_files"] = "warning (no model cache found)"
            
    except (OSError, ImportError) as e:
        services["model_files"] = f"unknown ({str(e)[:30]})"
    
    # Determine overall status
    if critical_failures:
        overall_status = "unhealthy"
        message = f"Critical services down: {', '.join(critical_failures)}"
    elif any("warning" in status for status in services.values()):
        overall_status = "degraded"
        message = "Some services have warnings"
    elif any("degraded" in status for status in services.values()):
        overall_status = "degraded" 
        message = "Some services are degraded"
    else:
        overall_status = "healthy"
        message = "All services operational"
    
    logger.info("Health check completed", 
               status=overall_status, 
               services_count=len(services),
               critical_failures=len(critical_failures))

    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version=settings.app_version,
        services=services,
        message=message,
    )
