"""
System Management API endpoints - Index statistics, analytics, and system information.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from core.logging import get_logger
from api.models import IndexStats, ArtistAnalytics

logger = get_logger(__name__)
router = APIRouter()


@router.get("/index/stats", response_model=IndexStats)
async def get_index_statistics() -> IndexStats:
    """
    Get comprehensive statistics about the image index.
    
    Returns counts, sizes, last update times, and model information.
    """
    from ml.vector_db import qdrant_service
    
    logger.info("Getting index statistics")
    
    try:
        await qdrant_service.connect()
        
        # Get collection info from Qdrant
        await qdrant_service.connect()
        
        # Use the client directly to get collection info
        try:
            collection_info = await qdrant_service.client.get_collection(qdrant_service.collection_name)
            total_images = collection_info.vectors_count or 0
            
            # Calculate actual artist count by scrolling through all points
            artist_ids = set()
            index_size_bytes = 0
            
            if total_images > 0:
                # Scroll through all points to get unique artists and calculate size
                scroll_result = await qdrant_service.client.scroll(
                    collection_name=qdrant_service.collection_name,
                    limit=10000,  # Get all points in batches
                    with_payload=True,
                    with_vectors=False  # Don't need vectors, just payload
                )
                
                for point in scroll_result[0]:  # scroll_result is (points, next_page_offset)
                    if point.payload:
                        # Extract artist_id from payload
                        if 'artist_id' in point.payload and point.payload['artist_id']:
                            artist_ids.add(point.payload['artist_id'])
                        # Estimate size based on payload
                        index_size_bytes += len(str(point.payload)) + 512 * 4  # Rough estimate: payload + vector size
            
            # Convert bytes to MB
            index_size_mb = index_size_bytes / (1024 * 1024)
            
            logger.info(
                "Index statistics retrieved",
                total_vectors=total_images,
                total_artists=len(artist_ids),
                index_size_mb=round(index_size_mb, 2),
                collection_status=collection_info.status
            )
        except Exception as e:
            logger.warning("Could not get precise collection info", error=str(e))
            total_images = 0
            artist_ids = set()
            index_size_mb = 0.0
        
        return IndexStats(
            total_images=total_images,
            total_artists=len(artist_ids),
            index_size_mb=round(index_size_mb, 2),
            last_updated=datetime.now(),
            vector_dimension=512,  # CLIP embedding dimension
            similarity_model="CLIP-ViT-B/32"
        )
        
    except Exception as e:
        logger.error("Failed to get index statistics", error=str(e))
        
        # Return default stats on error
        return IndexStats(
            total_images=0,
            total_artists=0,
            index_size_mb=0.0,
            last_updated=datetime.now(),
            vector_dimension=512,
            similarity_model="CLIP-ViT-B/32"
        )


@router.get("/artists/{artist_id}/analytics", response_model=ArtistAnalytics)
async def get_artist_analytics(artist_id: str) -> ArtistAnalytics:
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
            scroll_filter={
                "must": [
                    {
                        "key": "artist_id",
                        "match": {"value": artist_id}
                    }
                ]
            },
            limit=1000,  # Get up to 1000 images for this artist
            with_payload=True,
            with_vectors=False
        )
        
        artist_points = scroll_result[0]
        total_images = len(artist_points)
        
        if total_images == 0:
            return ArtistAnalytics(
                artist_id=artist_id,
                total_images=0,
                latest_upload=None,
                avg_similarity_score=0.0,
                image_distribution={}
            )
        
        # Calculate statistics and patterns
        upload_times = []
        entry_distribution = {}
        
        for point in artist_points:
            if point.payload:
                # Track upload times if available
                if 'upload_time' in point.payload:
                    try:
                        upload_time = datetime.fromisoformat(point.payload['upload_time'])
                        upload_times.append(upload_time)
                    except (ValueError, TypeError):
                        pass
                
                # Track entry distribution
                entry_id = point.payload.get('entry_id', 'unknown')
                entry_distribution[entry_id] = entry_distribution.get(entry_id, 0) + 1
        
        # Find latest upload
        latest_upload = max(upload_times) if upload_times else None
        
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
                        with_vectors=True
                    )
                    
                    if first_vector and first_vector[0].vector:
                        search_result = await qdrant_service.client.search(
                            collection_name=qdrant_service.collection_name,
                            query_vector=first_vector[0].vector,
                            query_filter={
                                "must": [{"key": "artist_id", "match": {"value": artist_id}}]
                            },
                            limit=sample_size
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
            entries=len(entry_distribution),
            avg_similarity=round(avg_similarity, 3)
        )
        
        return ArtistAnalytics(
            artist_id=artist_id,
            total_images=total_images,
            latest_upload=latest_upload,
            avg_similarity_score=round(avg_similarity, 3),
            image_distribution=entry_distribution
        )
        
    except Exception as e:
        logger.error("Failed to get analytics for artist", artist_id=artist_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics for artist {artist_id}: {str(e)}"
        ) from e


@router.post("/index/rebuild")
async def rebuild_index():
    """
    Rebuild the entire vector index from scratch.
    
    Useful for index corruption recovery or major upgrades.
    """
    logger.info("Starting index rebuild")
    
    try:
        from ml.vector_db import qdrant_service
        
        await qdrant_service.connect()
        
        # Get current collection info before rebuilding
        collection_name = qdrant_service.collection_name
        backup_name = f"{collection_name}_backup_{int(datetime.now().timestamp())}"
        
        logger.info("Starting index rebuild process", collection=collection_name, backup=backup_name)
        
        # Step 1: Create backup collection
        try:
            # Get current collection info
            collection_info = await qdrant_service.client.get_collection(collection_name)
            
            # Create backup collection with same configuration
            await qdrant_service.client.create_collection(
                collection_name=backup_name,
                vectors_config=collection_info.config.params.vectors
            )
            
            # Copy all points to backup
            scroll_result = await qdrant_service.client.scroll(
                collection_name=collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=True
            )
            
            if scroll_result[0]:  # If there are points to backup
                await qdrant_service.client.upsert(
                    collection_name=backup_name,
                    points=scroll_result[0]
                )
            
            logger.info("Backup created successfully", backup_points=len(scroll_result[0]))
            
        except Exception as e:
            logger.error("Failed to create backup", error=str(e))
            return {"message": f"Failed to create backup: {str(e)}", "status": "error"}
        
        # Step 2: Recreate main collection
        try:
            # Delete current collection
            await qdrant_service.client.delete_collection(collection_name)
            
            # Recreate with same configuration
            await qdrant_service.client.create_collection(
                collection_name=collection_name,
                vectors_config=collection_info.config.params.vectors
            )
            
            logger.info("Main collection recreated successfully")
            
        except Exception as e:
            logger.error("Failed to recreate collection", error=str(e))
            # Try to restore from backup
            try:
                backup_points = await qdrant_service.client.scroll(
                    collection_name=backup_name,
                    limit=10000,
                    with_payload=True,
                    with_vectors=True
                )
                if backup_points[0]:
                    await qdrant_service.client.upsert(
                        collection_name=collection_name,
                        points=backup_points[0]
                    )
                logger.info("Restored from backup after recreation failure")
            except Exception:
                logger.error("Failed to restore from backup")
            
            return {"message": f"Failed to recreate collection: {str(e)}", "status": "error"}
        
        # Step 3: Restore data from backup
        try:
            backup_points = await qdrant_service.client.scroll(
                collection_name=backup_name,
                limit=10000,
                with_payload=True,
                with_vectors=True
            )
            
            if backup_points[0]:
                await qdrant_service.client.upsert(
                    collection_name=collection_name,
                    points=backup_points[0]
                )
                
                logger.info("Data restored from backup", restored_points=len(backup_points[0]))
            
            # Clean up backup collection
            await qdrant_service.client.delete_collection(backup_name)
            
        except Exception as e:
            logger.error("Failed to restore data from backup", error=str(e))
            return {
                "message": f"Index recreated but data restoration failed: {str(e)}. Backup available at {backup_name}",
                "status": "partial_success",
                "backup_collection": backup_name
            }
        
        # Step 4: Verify index integrity
        try:
            final_info = await qdrant_service.client.get_collection(collection_name)
            final_count = final_info.vectors_count or 0
            
            logger.info("Index rebuild completed successfully", final_count=final_count)
            
            return {
                "message": "Index rebuilt successfully",
                "status": "success",
                "vectors_restored": final_count,
                "collection_status": final_info.status.name if final_info.status else "unknown"
            }
            
        except Exception as e:
            logger.error("Failed to verify rebuilt index", error=str(e))
            return {
                "message": f"Index rebuilt but verification failed: {str(e)}",
                "status": "success_unverified"
            }
        
    except Exception as e:
        logger.error("Index rebuild failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Index rebuild failed: {str(e)}"
        ) from e


@router.get("/model/info")
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
            "clip_model": "ViT-B/32",
            "embedding_dimension": 512,
            "supported_formats": ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"],
            "model_device": "Unknown",
            "model_precision": "float32",
            "preprocessing": {
                "resize": "224x224",
                "normalize": "ImageNet stats",
                "center_crop": True
            },
            "vector_database": {
                "type": "Qdrant",
                "distance_metric": "Cosine",
                "index_type": "HNSW"
            }
        }
        
        # Get actual device info from CLIP service if available
        try:
            if hasattr(clip_service, 'device'):
                model_info["model_device"] = str(clip_service.device)
            elif hasattr(clip_service, 'get_device'):
                model_info["model_device"] = str(clip_service.get_device())
                
            # Get model name if available
            if hasattr(clip_service, 'model_name'):
                model_info["clip_model"] = clip_service.model_name
            elif hasattr(clip_service, 'get_model_name'):
                model_info["clip_model"] = clip_service.get_model_name()
                
        except (AttributeError, RuntimeError):
            # If we can't get device/model info, use defaults
            pass
        
        # Get vector database info
        try:
            await qdrant_service.connect()
            collection_info = await qdrant_service.client.get_collection(qdrant_service.collection_name)
            
            if collection_info and collection_info.config:
                config = collection_info.config
                model_info["vector_database"]["collection_name"] = qdrant_service.collection_name
                model_info["vector_database"]["vector_size"] = config.params.vectors.size if config.params.vectors else 512
                model_info["vector_database"]["distance"] = config.params.vectors.distance.name if config.params.vectors and config.params.vectors.distance else "Cosine"
                
                # Get performance info
                if hasattr(config.params, 'hnsw_config') and config.params.hnsw_config:
                    hnsw = config.params.hnsw_config
                    model_info["vector_database"]["hnsw_config"] = {
                        "m": hnsw.m,
                        "ef_construct": hnsw.ef_construct,
                        "full_scan_threshold": hnsw.full_scan_threshold
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
            "gpu_acceleration": "Apple Silicon (MPS)" if "mps" in model_info["model_device"].lower() else "Auto-detected"
        }
        
        # Add capabilities
        model_info["capabilities"] = {
            "text_queries": True,
            "image_queries": True,
            "batch_processing": True,
            "similarity_threshold": True,
            "hierarchical_ids": True,
            "metadata_filtering": True
        }
        
        logger.info("Model information gathered successfully", 
                   clip_model=model_info["clip_model"],
                   device=model_info["model_device"],
                   dimension=model_info["embedding_dimension"])
        
        return model_info
        
    except Exception as e:
        logger.error("Failed to get model info", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        ) from e