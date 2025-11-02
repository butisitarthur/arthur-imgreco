"""
Centralized service management for Arthur Image Recognition 2.0

This module provides singleton access to core services to avoid
repeated initialization and ensure consistent service access patterns.
"""

from typing import Optional
from functools import lru_cache

from core.logging import get_logger

logger = get_logger(__name__)

# Global service instances
_clip_service: Optional['CLIPEmbeddingService'] = None
_qdrant_service: Optional['QdrantService'] = None
_cache_service: Optional['CacheService'] = None


@lru_cache()
async def get_clip_service():
    """Get or create CLIP embedding service instance."""
    global _clip_service
    
    if _clip_service is None:
        from ml.clip_service import CLIPEmbeddingService
        _clip_service = CLIPEmbeddingService()
        await _clip_service.load_model()
        logger.debug("CLIP service initialized")
    
    return _clip_service


@lru_cache()
def get_qdrant_service():
    """Get or create Qdrant vector database service instance."""
    global _qdrant_service
    
    if _qdrant_service is None:
        from ml.vector_db import QdrantService
        _qdrant_service = QdrantService()
        logger.debug("Qdrant service initialized")
    
    return _qdrant_service


@lru_cache()
async def get_cache_service():
    """Get or create cache service instance."""
    global _cache_service
    
    if _cache_service is None:
        from core.cache import CacheService
        _cache_service = CacheService()
        await _cache_service.connect()
        logger.debug("Cache service initialized")
    
    return _cache_service


def reset_services():
    """Reset all service instances (mainly for testing)."""
    global _clip_service, _qdrant_service, _cache_service
    
    _clip_service = None
    _qdrant_service = None
    _cache_service = None
    
    # Clear lru_cache
    get_clip_service.cache_clear()
    get_qdrant_service.cache_clear()
    get_cache_service.cache_clear()
    
    logger.debug("All services reset")