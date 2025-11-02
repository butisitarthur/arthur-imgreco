"""
Intelligent caching service for Arthur Image Recognition.

Provides smart caching for embeddings, search results, and system data
with automatic expiration and memory management.
"""

import json
import pickle
import hashlib
from typing import Any, Optional, List, Dict
from datetime import datetime, timedelta

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class CacheService:
    """
    Intelligent caching service with multiple backends and smart expiration.
    
    Falls back to in-memory caching if Redis is not available.
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.memory_cache: Dict[str, tuple] = {}  # (value, expiry_time)
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to Redis if available, otherwise use memory cache."""
        if not REDIS_AVAILABLE:
            logger.info("Redis not available, using in-memory cache")
            self.connected = True
            return True
            
        try:
            # Try to connect to Redis
            redis_url = getattr(settings, 'redis_url', 'redis://localhost:6379/0')
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info("Connected to Redis successfully", redis_url=redis_url)
            self.connected = True
            return True
            
        except Exception as e:
            logger.warning("Failed to connect to Redis, using memory cache", error=str(e))
            self.redis_client = None
            self.connected = True
            return False
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """Generate a consistent cache key from data."""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        
        hash_obj = hashlib.sha256(content.encode())
        return f"{prefix}:{hash_obj.hexdigest()[:16]}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.connected:
            await self.connect()
        
        try:
            if self.redis_client:
                # Try Redis first
                value = await self.redis_client.get(key)
                if value is not None:
                    return pickle.loads(value)
            else:
                # Use memory cache
                if key in self.memory_cache:
                    value, expiry = self.memory_cache[key]
                    if datetime.now() < expiry:
                        return value
                    else:
                        # Expired, remove it
                        del self.memory_cache[key]
                        
        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
        
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Set value in cache with TTL."""
        if not self.connected:
            await self.connect()
        
        try:
            if self.redis_client:
                # Use Redis
                serialized = pickle.dumps(value)
                await self.redis_client.setex(key, ttl_seconds, serialized)
            else:
                # Use memory cache
                expiry = datetime.now() + timedelta(seconds=ttl_seconds)
                self.memory_cache[key] = (value, expiry)
                
                # Clean up expired entries periodically
                await self._cleanup_memory_cache()
            
            return True
            
        except Exception as e:
            logger.warning("Cache set failed", key=key, error=str(e))
            return False
    
    async def _cleanup_memory_cache(self):
        """Remove expired entries from memory cache."""
        if len(self.memory_cache) > 1000:  # Only cleanup if cache is getting large
            now = datetime.now()
            expired_keys = [
                key for key, (_, expiry) in self.memory_cache.items()
                if now >= expiry
            ]
            
            for key in expired_keys:
                del self.memory_cache[key]
            
            if expired_keys:
                logger.info("Cleaned up expired cache entries", count=len(expired_keys))
    
    async def cache_embedding(self, text: str, embedding: List[float], ttl_hours: int = 24) -> str:
        """Cache a text embedding with smart key generation."""
        key = self._generate_key("embedding", text.lower().strip())
        await self.set(key, embedding, ttl_seconds=ttl_hours * 3600)
        return key
    
    async def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text query."""
        key = self._generate_key("embedding", text.lower().strip())
        return await self.get(key)
    
    async def cache_search_results(self, query: str, filters: Dict, results: List[Dict], ttl_minutes: int = 30) -> str:
        """Cache similarity search results."""
        cache_data = {"query": query, "filters": filters}
        key = self._generate_key("search", cache_data)
        await self.set(key, results, ttl_seconds=ttl_minutes * 60)
        return key
    
    async def get_cached_search_results(self, query: str, filters: Dict) -> Optional[List[Dict]]:
        """Get cached search results."""
        cache_data = {"query": query, "filters": filters}
        key = self._generate_key("search", cache_data)
        return await self.get(key)
    
    async def cache_system_stats(self, stats: Dict, ttl_minutes: int = 5) -> bool:
        """Cache system statistics with short TTL."""
        return await self.set("system:stats", stats, ttl_seconds=ttl_minutes * 60)
    
    async def get_cached_system_stats(self) -> Optional[Dict]:
        """Get cached system statistics."""
        return await self.get("system:stats")
    
    async def close(self):
        """Properly close Redis connections."""
        if self.redis_client:
            try:
                await self.redis_client.close()
            except Exception:
                pass  # Ignore cleanup errors
    
    async def health_check(self) -> Dict[str, str]:
        """Check cache service health."""
        if not self.connected:
            return {"status": "not_connected", "backend": "none"}
        
        try:
            # Test cache operations
            test_key = "health:test"
            test_value = {"timestamp": datetime.now().isoformat(), "test": True}
            
            success = await self.set(test_key, test_value, ttl_seconds=60)
            if success:
                retrieved = await self.get(test_key)
                if retrieved == test_value:
                    backend = "redis" if self.redis_client else "memory"
                    cache_size = len(self.memory_cache) if not self.redis_client else "redis_managed"
                    return {
                        "status": "healthy",
                        "backend": backend,
                        "cache_size": str(cache_size)
                    }
            
            return {"status": "degraded", "backend": "unknown"}
            
        except Exception as e:
            logger.error("Cache health check failed", error=str(e))
            return {"status": "unhealthy", "backend": "error", "error": str(e)[:50]}


# Global cache service instance
cache_service = CacheService()


async def get_cache_service() -> CacheService:
    """Get the global cache service instance."""
    if not cache_service.connected:
        await cache_service.connect()
    return cache_service