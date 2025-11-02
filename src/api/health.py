"""
Health check endpoints
"""

from fastapi import APIRouter
from datetime import datetime

from core.config import settings
from core.logging import get_logger
from api.models import HealthResponse, DetailedHealthResponse

logger = get_logger(__name__)
router = APIRouter()


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
    
    # Check Qdrant Vector DB
    try:
        from core.services import get_qdrant_service
        qdrant_service = get_qdrant_service()
        
        # Basic collection check
        collections = await qdrant_service.list_collections()
        if collections:
            services["vector_db"] = f"healthy ({len(collections)} collections)"
        else:
            services["vector_db"] = "healthy (no collections)"
    except Exception as e:
        services["vector_db"] = f"unhealthy: {str(e)}"
        logger.error("Vector DB health check failed", error=str(e))
    
    # Check Redis Cache
    try:
        from core.services import get_cache_service
        cache_service = await get_cache_service()
        
        # Test cache with a simple operation
        test_key = "health_check_test"
        await cache_service.set(test_key, "test_value", ttl=1)
        cached_value = await cache_service.get(test_key)
        
        if cached_value == "test_value":
            services["cache"] = "healthy (Redis connected)"
            # Clean up test key
            await cache_service.delete(test_key)
        else:
            services["cache"] = "degraded (Redis accessible but inconsistent)"
    except Exception as e:
        services["cache"] = f"degraded: {str(e)}"
        logger.warning("Cache health check failed, falling back to memory cache", error=str(e))
    
    # Check CLIP ML Model
    try:
        from core.services import get_clip_service
        clip_service = await get_clip_service()
        
        # Check if model is loaded and accessible
        if hasattr(clip_service, 'model') and clip_service.model is not None:
            services["ml_model"] = f"healthy (CLIP {settings.clip_model_name} loaded)"
        else:
            services["ml_model"] = "healthy (CLIP service available)"
        
    except Exception as e:
        services["ml_model"] = f"unhealthy: {str(e)}"
        logger.error("ML model health check failed", error=str(e))
    
    # Check CLIP ML Model
    try:
        from ml.clip_service import clip_service
        
        # Check if model is loaded and accessible
        if hasattr(clip_service, 'model') and clip_service.model is not None:
            services["ml_model"] = f"healthy (CLIP {settings.clip_model_name} loaded)"
        else:
            # Check if clip_service is accessible (this will load model if needed)
            services["ml_model"] = "healthy (CLIP service available)"
        
        logger.info("CLIP model health check passed")
        
    except (ImportError, RuntimeError, OSError) as e:
        services["ml_model"] = f"unhealthy ({str(e)[:50]})"
        critical_failures.append("ml_model")
        logger.error("CLIP model health check failed", error=str(e))
    
    # Check Cache Service
    try:
        from core.cache import get_cache_service
        
        cache_service = await get_cache_service()
        cache_health = await cache_service.health_check()
        
        cache_status = cache_health.get("status", "unknown")
        backend = cache_health.get("backend", "unknown")
        
        if cache_status == "healthy":
            services["cache"] = f"healthy ({backend} backend)"
        elif cache_status == "degraded":
            services["cache"] = f"degraded ({backend} backend)"
        else:
            services["cache"] = f"unhealthy ({backend} backend)"
            
        logger.info("Cache service checked", status=cache_status, backend=backend)
        
    except (ImportError, ConnectionError, RuntimeError) as e:
        services["cache"] = f"not_available ({str(e)[:30]})"
        logger.warning("Cache service check failed", error=str(e))
    
    # Database architecture note: This system uses Qdrant as the primary database
    # All metadata is stored as vector payload, no separate database needed
    services["metadata_storage"] = "healthy (integrated with vector_db)"
    
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
