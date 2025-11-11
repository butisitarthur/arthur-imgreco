"""
System Management API endpoints - Index statistics, analytics, and system information.
"""

from datetime import datetime

from fastapi import APIRouter, status

from core.logging import get_logger
from core.config import settings
from core.responses import success_response, error_response
from core.models import IndexStatus, ArtistStatus, SystemStatus, EntryStatus, ImageStatus

logger = get_logger(__name__)
router = APIRouter()


# BASE PATH: /api/v1/status


@router.get("/system", summary="Get health status for system & services")
async def get_system_status():
    """Detailed health check with service status."""

    services = {}
    critical_failures = []
    indexing_threshold = None  # Store indexing threshold for SystemStatus

    # Check Qdrant Vector DB
    try:
        from core.services import get_qdrant_service

        qdrant_service = get_qdrant_service()

        # Use the built-in health check method
        is_healthy = await qdrant_service.health_check()
        if is_healthy:
            # Get additional collection info
            try:
                stats = await qdrant_service.get_collection_stats()
                if "error" not in stats:
                    vectors_count = stats.get("vectors_count")
                    points_count = stats.get("points_count", 0) or 0
                    total_count = vectors_count if vectors_count is not None else points_count
                    services["vector_db"] = f"healthy ({total_count} vectors)"
                else:
                    services["vector_db"] = "healthy (connected)"

                # Try to get indexing threshold from collection config
                try:
                    collection_info = await qdrant_service.client.get_collection(
                        qdrant_service.collection_name
                    )
                    if (
                        collection_info
                        and collection_info.config
                        and collection_info.config.optimizer_config
                    ):
                        indexing_threshold = (
                            collection_info.config.optimizer_config.indexing_threshold
                        )
                        logger.info(
                            "Retrieved indexing threshold from collection",
                            threshold=indexing_threshold,
                        )
                except Exception as threshold_error:
                    logger.warning(
                        "Could not retrieve indexing threshold", error=str(threshold_error)
                    )
            except Exception:
                services["vector_db"] = "healthy (connected)"
        else:
            services["vector_db"] = "unhealthy (connection failed)"
            critical_failures.append("vector_db")
    except Exception as e:
        services["vector_db"] = f"unhealthy: {str(e)}"
        critical_failures.append("vector_db")
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
        if hasattr(clip_service, "model") and clip_service.model is not None:
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
        if hasattr(clip_service, "model") and clip_service.model is not None:
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
        _, _, free = shutil.disk_usage("/")
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

        # Check HuggingFace cache where CLIP models are actually stored
        huggingface_cache = home_dir / ".cache" / "huggingface" / "hub"
        clip_model_cache = huggingface_cache / "models--openai--clip-vit-base-patch32"

        if clip_model_cache.exists():
            services["model_files"] = "healthy (CLIP models cached)"
        else:
            # Fallback to traditional clip cache location
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

    logger.info(
        "Health check completed",
        status=overall_status,
        services_count=len(services),
        critical_failures=len(critical_failures),
    )

    detailed_health_response = SystemStatus(
        status=overall_status,
        timestamp=datetime.now().timestamp(),
        version=settings.app_version,
        services=services,
        message=message,
        indexing_threshold=indexing_threshold,
    )

    if overall_status == "healthy":
        return success_response(
            message=message,
            status=detailed_health_response.dict(),
        )
    else:
        return success_response(
            message=message,
            status=detailed_health_response.dict(),
        )


@router.get("/index", summary="Get index status")
async def get_index_status(no_cache: bool = False):
    """
    Get comprehensive statistics about the image index.

    Returns counts, sizes, last update times, and model information.
    Uses 5-minute caching to reduce database load.

    Args:
        no_cache: If True, bypass cache and fetch fresh data
    """
    from ml.vector_db import qdrant_service

    # Try to get cached stats first (unless no_cache is True)
    if not no_cache:
        try:
            from core.cache import get_cache_service

            cache_service = await get_cache_service()
            cached_stats = await cache_service.get_cached_system_stats()

            if cached_stats:
                logger.info("Returning cached index status")
                # Add cached timestamp to the cached result
                cached_stats["cached"] = True
                return success_response(
                    message="Index status retrieved (cached)", status=cached_stats
                )
        except Exception:
            pass

    if no_cache:
        logger.info("Computing fresh index status (cache bypassed)")
    else:
        logger.info("Computing fresh index status")

    try:
        # Use the QdrantService method to get stats
        stats = await qdrant_service.get_collection_stats()

        # Initialize default values
        indexed_vectors_count = 0
        points_count = 0
        indexing_status = "unknown"

        if "error" in stats:
            logger.warning("Error getting collection stats", error=stats["error"])
            total_images = 0
            artist_ids = set()
            index_size_mb = 0.0
            indexing_status = "error"
        else:
            # Use indexed_vectors_count for proper vector search capability
            # and points_count for total stored records
            indexed_vectors_count = stats.get("indexed_vectors_count", 0) or 0
            points_count = stats.get("points_count", 0) or 0

            # For similarity search to work, we need indexed vectors
            # If indexed_vectors_count is 0 but points_count > 0, indexing hasn't started
            total_images = indexed_vectors_count if indexed_vectors_count > 0 else points_count

            # Determine indexing status
            if indexed_vectors_count == 0 and points_count > 0:
                logger.warning(
                    "Vectors stored but not indexed yet",
                    points_stored=points_count,
                    indexed_vectors=indexed_vectors_count,
                    indexing_note="Similarity search may be slow until indexing completes",
                )
                indexing_status = "pending"
            elif indexed_vectors_count > 0 and indexed_vectors_count < points_count:
                indexing_status = "building"
            elif indexed_vectors_count == points_count and points_count > 0:
                indexing_status = "ready"
            else:
                indexing_status = "empty"

            # Calculate artist & entry counts by scrolling through all points
            artist_ids = set()
            entry_ids = set()

            # Get index size - try multiple size fields from Qdrant
            index_size_bytes = stats.get("disk_data_size", 0) or stats.get("ram_data_size", 0) or 0

            # If no size info available from Qdrant, estimate based on vector count
            if index_size_bytes == 0 and total_images > 0:
                # Rough estimate: vectors + metadata
                # Each vector: dimension * 4 bytes (float32) + ~100 bytes metadata
                vector_size = settings.embedding_dimension * 4
                metadata_size = 100  # Approximate metadata size per vector
                estimated_size = total_images * (vector_size + metadata_size)
                index_size_bytes = estimated_size
                logger.debug(
                    "Using estimated index size",
                    estimated_bytes=index_size_bytes,
                    vector_size=vector_size,
                    total_vectors=total_images,
                )

            logger.debug(
                "Index size calculation",
                disk_data_size=stats.get("disk_data_size"),
                ram_data_size=stats.get("ram_data_size"),
                calculated_bytes=index_size_bytes,
                estimated=index_size_bytes > 0 and stats.get("disk_data_size", 0) == 0,
            )

            if total_images > 0:
                try:
                    # Scroll through all points to get unique artists
                    scroll_result = await qdrant_service.client.scroll(
                        collection_name=qdrant_service.collection_name,
                        limit=10000,  # Get all points in batches
                        with_payload=True,
                        with_vectors=False,  # Don't need vectors, just payload
                    )

                    for point in scroll_result[0]:  # scroll_result is (points, next_page_offset)
                        if point.payload:
                            # Extract artist_id from payload
                            if "artist_id" in point.payload and point.payload["artist_id"]:
                                artist_ids.add(point.payload["artist_id"])

                            # Extract entry_id from payload
                            if "entry_id" in point.payload and point.payload["entry_id"]:
                                entry_ids.add(point.payload["entry_id"])

                    logger.debug(
                        "Scrolled collection for artist count",
                        total_points=len(scroll_result[0]),
                        unique_artists=len(artist_ids),
                        unique_entries=len(entry_ids),
                    )
                except Exception as scroll_error:
                    logger.warning(
                        "Failed to scroll collection for artist count", error=str(scroll_error)
                    )

            # Convert bytes to MB
            index_size_mb = index_size_bytes / (1024 * 1024) if index_size_bytes > 0 else 0.0

            logger.info(
                "Index status retrieved",
                total_artists=len(artist_ids),
                total_entries=len(entry_ids),
                total_images=total_images,
                index_size_mb=round(index_size_mb, 2),
                collection_status=stats.get("status", "unknown"),
            )

        index_status = IndexStatus(
            total_artists=len(artist_ids),
            total_entries=len(entry_ids),
            total_images=total_images,
            index_size_mb=round(index_size_mb, 2),
            last_updated=datetime.now().timestamp(),
            vector_dimension=settings.embedding_dimension,
            similarity_model=f"CLIP-{settings.clip_model_name}",
            cached_time=datetime.now().timestamp(),  # Timestamp for when retrieving cached result
            cached=False,
            indexed_vectors=indexed_vectors_count,
            stored_points=points_count,
            indexing_status=indexing_status,
        )

        # Cache the computed stats for 5 minutes
        try:
            from core.cache import get_cache_service

            cache_service = await get_cache_service()
            await cache_service.cache_system_stats(index_status.dict(), ttl_minutes=5)
        except Exception:
            # If caching fails, that's ok
            pass

        return success_response(message="Index status retrieved", status=index_status.dict())

    except Exception as e:
        logger.error("Failed to get index status", error=str(e))

        # Return default stats on error
        default_status = IndexStatus(
            total_images=0,
            total_artists=0,
            total_entries=0,  # Add missing required field
            index_size_mb=0.0,
            last_updated=datetime.now().timestamp(),
            vector_dimension=settings.embedding_dimension,
            similarity_model=f"CLIP-{settings.clip_model_name}",
            cached=False,  # Add missing required field
            indexed_vectors=0,
            stored_points=0,
            indexing_status="error",
        )
        return error_response(
            message="Failed to retrieve index status",
            details=str(e),
            status=default_status.dict(),
        )


@router.get("/artist/{artist_id}", summary="Get artist status")
async def get_artist_status(artist_id: str):
    """
    Get detailed analytics for a specific artist.

    Returns image counts, upload patterns, and similarity insights.
    """
    logger.info("Getting artist analytics", artist_id=artist_id)

    try:
        from ml.vector_db import qdrant_service

        await qdrant_service.connect()

        # Query vector database for artist's images
        scroll_result = await qdrant_service.client.scroll(
            collection_name=qdrant_service.collection_name,
            scroll_filter={"must": [{"key": "artist_id", "match": {"value": artist_id}}]},
            limit=1000,  # Get up to 1000 images for this artist
            with_payload=True,
            with_vectors=False,
        )

        artist_points = scroll_result[0]
        total_images = len(artist_points)

        if total_images == 0:
            logger.info("No images found for artist", artist_id=artist_id)
            empty_analytics = ArtistStatus(
                artist_id=artist_id,
                total_images=0,
                latest_upload=None,
                avg_similarity_score=0.0,
                image_distribution={},
                vector_size_bytes=0,
            )
            return success_response(
                message=f"No images found for artist {artist_id}", report=empty_analytics.dict()
            )

        # Calculate statistics and patterns
        upload_times = []
        entry_distribution = {}

        for point in artist_points:
            if point.payload:
                # Track upload times if available
                if "upload_timestamp" in point.payload:
                    try:
                        timestamp = float(point.payload["upload_timestamp"])
                        upload_times.append(timestamp)
                    except (ValueError, TypeError, OSError):
                        pass

                # Track entry distribution
                entry_id = point.payload.get("entry_id", "unknown")
                entry_distribution[entry_id] = entry_distribution.get(entry_id, 0) + 1

        # Find latest upload
        latest_upload = max(upload_times) if upload_times else None

        # Calculate vector size for this artist
        # Each vector: dimension * 4 bytes (float32) + ~100 bytes metadata
        vector_size = settings.embedding_dimension * 4
        metadata_size = 100  # Approximate metadata size per vector
        artist_size_bytes = total_images * (vector_size + metadata_size)

        # Calculate average similarity score (simplified - compare first few images)
        avg_similarity = 0.0
        if total_images > 1:
            # Take a sample of images to calculate average internal similarity
            sample_size = min(5, total_images)
            sample_points = artist_points[:sample_size]

            # Get vectors for similarity calculation
            vector_ids = [point.id for point in sample_points]
            if len(vector_ids) > 1:
                try:
                    # Get one vector and search for similar ones
                    first_vector = await qdrant_service.client.retrieve(
                        collection_name=qdrant_service.collection_name,
                        ids=[vector_ids[0]],
                        with_vectors=True,
                    )

                    if first_vector and first_vector[0].vector:
                        search_result = await qdrant_service.client.search(
                            collection_name=qdrant_service.collection_name,
                            query_vector=first_vector[0].vector,
                            query_filter={
                                "must": [{"key": "artist_id", "match": {"value": artist_id}}]
                            },
                            limit=sample_size,
                        )

                        if search_result:
                            scores = [hit.score for hit in search_result[1:]]  # Exclude self-match
                            avg_similarity = sum(scores) / len(scores) if scores else 0.0
                except (RuntimeError, ValueError, TypeError):
                    # If similarity calculation fails, use default
                    avg_similarity = 0.0

        logger.info(
            "Artist analytics calculated",
            artist_id=artist_id,
            total_images=total_images,
            vector_size_bytes=round(artist_size_bytes, 0),
            avg_similarity=round(avg_similarity, 3),
            entries=len(entry_distribution),
        )

        # raise HTTPException(status_code=500, detail="This is a test")

        analytics = ArtistStatus(
            artist_id=artist_id,
            total_images=total_images,
            latest_upload=latest_upload,
            avg_similarity_score=round(avg_similarity, 3),
            vector_size_bytes=artist_size_bytes,
            image_distribution=entry_distribution,
        )

        return success_response(
            message=f"Analytics retrieved for artist {artist_id}", report=analytics.dict()
        )

    except Exception as e:
        logger.error("Failed to get analytics for artist", artist_id=artist_id, error=str(e))
        return error_response(
            message="Failed to retrieve artist analytics", details=str(e), artist_id=artist_id
        )


@router.get("/entry/{entry_id}", summary="Get entry status")
async def get_entry_status(entry_id: str):
    """
    Get basic analytics for a specific entry.

    Returns latest upload time and view count.
    """
    logger.info("Getting entry analytics", entry_id=entry_id)

    try:
        from ml.vector_db import qdrant_service

        await qdrant_service.connect()

        # Query vector database for entry's images
        scroll_result = await qdrant_service.client.scroll(
            collection_name=qdrant_service.collection_name,
            scroll_filter={"must": [{"key": "entry_id", "match": {"value": entry_id}}]},
            limit=1000,  # Get up to 1000 images for this entry
            with_payload=True,
            with_vectors=False,
        )

        entry_points = scroll_result[0]

        if len(entry_points) == 0:
            logger.info("No images found for entry", entry_id=entry_id)
            empty_entry = EntryStatus(
                entry_id=entry_id,
                latest_upload=None,
                view_count=0,
                total_images=0,
            )
            return success_response(
                message=f"Analytics retrieved for entry {entry_id}",
                entry=empty_entry.dict(),
            )

        # Find latest upload timestamp
        upload_times = []
        view_count = 0

        for point in entry_points:
            if point.payload:
                # Track upload times if available
                if "upload_timestamp" in point.payload:
                    try:
                        timestamp = float(point.payload["upload_timestamp"])
                        upload_times.append(timestamp)
                    except (ValueError, TypeError, OSError):
                        pass

                # Track view count if available (or just count images as a proxy)
                if "view_count" in point.payload:
                    try:
                        view_count += int(point.payload.get("view_count", 0))
                    except (ValueError, TypeError):
                        pass

        # If no view_count in payload, use image count as view_count
        if view_count == 0:
            view_count = len(entry_points)

        # Find latest upload
        latest_upload = max(upload_times) if upload_times else None

        logger.info(
            "Entry analytics calculated",
            entry_id=entry_id,
            latest_upload=latest_upload,
            view_count=view_count,
        )

        entry_status = EntryStatus(
            entry_id=entry_id,
            latest_upload=latest_upload,
            view_count=view_count,
            total_images=len(entry_points),
        )

        return success_response(
            message=f"Analytics retrieved for entry {entry_id}",
            report=entry_status.dict(),
        )

    except Exception as e:
        logger.error("Failed to get entry analytics for entry", entry_id=entry_id, error=str(e))
        return error_response(
            message="Failed to retrieve entry analytics", details=str(e), entry_id=entry_id
        )


@router.get("/image/{view_id}", summary="Get image status")
async def get_image_status(view_id: str):
    """
    Get information about a specific image in the index.

    Returns metadata, embedding info, and similarity statistics.
    Can look up by either vector_id (UUID) or view_id.
    """
    from ml.vector_db import qdrant_service

    logger.info("Getting image info", image_id=view_id)

    try:
        # Connect to Qdrant
        await qdrant_service.connect()

        # Try metadata lookup on view_id
        try:
            # Search for vectors with matching view_id in metadata
            search_results = await qdrant_service.client.scroll(
                collection_name=qdrant_service.collection_name,
                scroll_filter={"must": [{"key": "view_id", "match": {"value": view_id}}]},
                limit=10,  # Allow for multiple matches
                with_payload=True,
                with_vectors=False,
            )

            if not search_results[0]:  # No points found
                logger.warning("Image not found", image_id=view_id)
                return error_response(
                    message="Image not found",
                    details=f"No image found with ID: {view_id}",
                    status_code=status.HTTP_404_NOT_FOUND,
                )

            # Return information about all matching vectors
            results = []
            for point in search_results[0]:
                metadata = point.payload or {}
                results.append(
                    {
                        "vector_id": str(point.id),
                        "image_id": metadata.get("entry_id"),
                        "artist_id": metadata.get("artist_id"),
                        "view_id": metadata.get("view_id"),
                        "image_url": metadata.get("image_url"),
                        "upload_timestamp": metadata.get("upload_timestamp"),
                        "embedding_model": metadata.get("embedding_model"),
                        "metadata": metadata,
                    }
                )

            logger.info("Images found by metadata search", view_id=view_id, count=len(results))

            if len(results) == 1:
                # Single result
                result = results[0]
                image_status = ImageStatus(
                    vector_id=result["vector_id"],
                    entry_id=result["image_id"],
                    artist_id=result["artist_id"],
                    view_id=result["view_id"],
                    image_url=result["image_url"],
                    upload_timestamp=result["upload_timestamp"],
                    embedding_model=result["embedding_model"],
                    lookup_method="metadata_search",
                    metadata=result["metadata"],
                )

                return success_response(
                    message=f"Image found by metadata search: {view_id}",
                    image=image_status.dict(),
                )
            else:
                # Multiple results - convert each to ImageStatus
                image_statuses = []
                for result in results:
                    image_status = ImageStatus(
                        vector_id=result["vector_id"],
                        entry_id=result["image_id"],
                        artist_id=result["artist_id"],
                        view_id=result["view_id"],
                        image_url=result["image_url"],
                        upload_timestamp=result["upload_timestamp"],
                        embedding_model=result["embedding_model"],
                        lookup_method="metadata_search",
                        metadata=result["metadata"],
                    )
                    image_statuses.append(image_status.dict())

                return success_response(
                    message=f"Found {len(results)} vectors for image_id: {view_id}",
                    images={
                        "count": len(results),
                        "results": image_statuses,
                    },
                )

        except Exception as view_id_error:
            logger.error("Failed to search by metadata", image_id=view_id, error=str(view_id_error))

        # Try direct lookup by vector_id
        try:
            points = await qdrant_service.client.retrieve(
                collection_name=qdrant_service.collection_name,
                ids=[view_id],
                with_payload=True,
                with_vectors=False,  # Don't return the actual embedding vector
            )

            if points:
                point = points[0]

                # Extract metadata
                metadata = point.payload or {}

                logger.info("Image found by vector_id", vector_id=view_id, metadata=metadata)

                image_status = ImageStatus(
                    vector_id=str(point.id),
                    entry_id=metadata.get("entry_id"),
                    artist_id=metadata.get("artist_id"),
                    view_id=metadata.get("view_id"),
                    image_url=metadata.get("image_url"),
                    upload_timestamp=metadata.get("upload_timestamp"),
                    embedding_model=metadata.get("embedding_model"),
                    lookup_method="vector_id",
                    metadata=metadata,
                )

                return success_response(
                    message=f"Image found by vector_id: {view_id}",
                    image=image_status.dict(),
                )

        except Exception as vector_lookup_error:
            logger.debug(
                "Failed to find by vector_id, trying metadata search",
                image_id=view_id,
                error=str(vector_lookup_error),
            )
            return error_response(
                message="Could not find image",
                details=f"view_id lookup: {view_id_error}\nvector_id lookup: {vector_lookup_error}",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    except Exception as e:
        logger.error("Failed to get image info", image_id=view_id, error=str(e), exc_info=True)
        return error_response(
            message="Failed to get image info",
            details=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.get("/model", summary="Get model info")
async def get_model_info():
    """
    Get information about the AI models and their capabilities.

    Returns model versions, performance metrics, and supported operations.
    """
    logger.info("Getting model information")

    try:
        from ml.clip_service import clip_service
        from ml.vector_db import qdrant_service

        # Query CLIP service for model details
        model_info = {
            "clip_model": settings.clip_model_name,
            "embedding_dimension": settings.embedding_dimension,
            "supported_formats": ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"],
            "model_device": "Unknown",
            "model_precision": "float32",
            "preprocessing": {
                "resize": "224x224",
                "normalize": "ImageNet stats",
                "center_crop": True,
            },
            "vector_database": {
                "type": "Qdrant",
                "distance_metric": "Cosine",
                "index_type": "HNSW",
            },
        }

        # Get actual device info from CLIP service if available
        try:
            if hasattr(clip_service, "device"):
                model_info["model_device"] = str(clip_service.device)
            elif hasattr(clip_service, "get_device"):
                model_info["model_device"] = str(clip_service.get_device())

            # Get model name if available
            if hasattr(clip_service, "model_name"):
                model_info["clip_model"] = clip_service.model_name
            elif hasattr(clip_service, "get_model_name"):
                model_info["clip_model"] = clip_service.get_model_name()

        except (AttributeError, RuntimeError):
            # If we can't get device/model info, use defaults
            pass

        # Get vector database info
        try:
            await qdrant_service.connect()
            collection_info = await qdrant_service.client.get_collection(
                qdrant_service.collection_name
            )

            if collection_info and collection_info.config:
                config = collection_info.config
                model_info["vector_database"]["collection_name"] = qdrant_service.collection_name
                model_info["vector_database"]["vector_size"] = (
                    config.params.vectors.size
                    if config.params.vectors
                    else settings.embedding_dimension
                )
                model_info["vector_database"]["distance"] = (
                    config.params.vectors.distance.name
                    if config.params.vectors and config.params.vectors.distance
                    else "Cosine"
                )

                # Get performance info
                if hasattr(config.params, "hnsw_config") and config.params.hnsw_config:
                    hnsw = config.params.hnsw_config
                    model_info["vector_database"]["hnsw_config"] = {
                        "m": hnsw.m,
                        "ef_construct": hnsw.ef_construct,
                        "full_scan_threshold": hnsw.full_scan_threshold,
                    }

        except Exception:
            # If we can't get DB info, use defaults
            pass

        # Add performance characteristics
        model_info["performance"] = {
            "embedding_generation": "~100-300ms per image",
            "similarity_search": "~10-50ms for 1000 vectors",
            "batch_processing": "Supported",
            "concurrent_requests": "Yes",
            "gpu_acceleration": (
                "Apple Silicon (MPS)"
                if "mps" in model_info["model_device"].lower()
                else "Auto-detected"
            ),
        }

        # Add capabilities
        model_info["capabilities"] = {
            "text_queries": True,
            "image_queries": True,
            "batch_processing": True,
            "similarity_threshold": True,
            "hierarchical_ids": True,
            "metadata_filtering": True,
        }

        logger.info(
            "Model information gathered successfully",
            clip_model=model_info["clip_model"],
            device=model_info["model_device"],
            dimension=model_info["embedding_dimension"],
        )

        return success_response(message="Model information retrieved", model=model_info)

    except Exception as e:
        logger.error("Failed to get model info", error=str(e))
        return error_response(message="Failed to retrieve model information", details=str(e))


@router.get("/latest-entries", summary="Get latest entries")
async def get_latest_entries(limit: int = 10, after_timestamp: float = None):
    """
    Get the most recent entries added to the vector database.

    Returns the last N entries based on upload timestamp, with cursor-based pagination.

    Args:
        limit: Number of entries to return (default: 10, max: 500)
        after_timestamp: Return entries before this timestamp (for pagination)

    Note: Use after_timestamp for efficient pagination. The response includes the timestamp
    of the last entry for use in the next request.
    """
    # Enforce max limit to prevent excessive queries
    limit = min(limit, 500)

    logger.info(
        "Getting latest entries",
        limit=limit,
        after_timestamp=after_timestamp,
    )

    try:
        from ml.vector_db import qdrant_service

        await qdrant_service.connect()

        # Build filter for efficient querying
        scroll_filter = None
        if after_timestamp is not None:
            # Only get entries with timestamp less than after_timestamp
            scroll_filter = {
                "must": [{"key": "upload_timestamp", "range": {"lt": after_timestamp}}]
            }

        # Fetch more than requested to account for deduplication
        fetch_limit = limit * 2

        # Scroll through points with filtering and descending order
        from qdrant_client.models import OrderBy, Direction

        scroll_result = await qdrant_service.client.scroll(
            collection_name=qdrant_service.collection_name,
            scroll_filter=scroll_filter,
            limit=fetch_limit,
            with_payload=True,
            with_vectors=False,
            order_by=OrderBy(key="upload_timestamp", direction=Direction.DESC),
        )

        points = scroll_result[0]

        if not points:
            logger.info("No entries found in database")
            return success_response(
                message="No entries found",
                entries=[],
                has_more=False,
                next_cursor=None,
            )

        # Extract entries with timestamps
        entries = []
        seen_entry_ids = set()

        for point in points:
            if point.payload:
                entry_id = point.payload.get("entry_id")

                # Deduplicate by entry_id (multiple views per entry)
                if entry_id and entry_id in seen_entry_ids:
                    continue
                if entry_id:
                    seen_entry_ids.add(entry_id)

                entry = {
                    "vector_id": str(point.id),
                    "artist_id": point.payload.get("artist_id"),
                    "entry_id": entry_id,
                    "view_id": point.payload.get("view_id"),
                    "image_url": point.payload.get("image_url"),
                    "upload_timestamp": point.payload.get("upload_timestamp"),
                    "embedding_model": point.payload.get("embedding_model"),
                }
                entries.append(entry)

        # Entries are already sorted by Qdrant in descending order
        # Just separate those with/without timestamps for consistency
        entries_with_timestamp = [e for e in entries if e.get("upload_timestamp")]
        entries_without_timestamp = [e for e in entries if not e.get("upload_timestamp")]

        sorted_entries = entries_with_timestamp + entries_without_timestamp

        # Get the requested number of entries
        latest_entries = sorted_entries[:limit]
        has_more = len(sorted_entries) > limit

        # Get cursor for next page (timestamp of last entry)
        next_cursor = None
        if latest_entries and has_more:
            last_entry = latest_entries[-1]
            next_cursor = last_entry.get("upload_timestamp")

        logger.info(
            "Latest entries retrieved",
            returned_count=len(latest_entries),
            has_more=has_more,
            next_cursor=next_cursor,
        )

        return success_response(
            message=f"Retrieved {len(latest_entries)} latest entries",
            entries=latest_entries,
            has_more=has_more,
            next_cursor=next_cursor,
            pagination_hint="Use next_cursor as after_timestamp parameter for next page",
        )

    except Exception as e:
        logger.error("Failed to get latest entries", error=str(e))
        return error_response(
            message="Failed to retrieve latest entries",
            details=str(e),
        )


@router.delete("/clear-cache", summary="Clear status cache")
async def clear_status_cache():
    """
    Clear the cached statistics to force fresh calculation on next request.

    Useful when the database has been updated and you want immediate stats refresh.
    """
    logger.info("Clearing stats cache")

    try:
        from core.cache import get_cache_service

        cache_service = await get_cache_service()

        # Clear the cached stats
        await cache_service.set("system:stats", None, ttl_seconds=1)

        logger.info("Stats cache cleared successfully")
        return success_response(
            message="Stats cache cleared successfully",
            status="success",
            note="Next stats request will fetch fresh data",
        )

    except Exception as e:
        logger.error("Failed to clear stats cache", error=str(e))
        return error_response(message="Failed to clear cache", details=str(e), status="error")
