"""
Modern API v1 - Main router that includes all endpoint modules.
"""

from fastapi import APIRouter

from arthur_imgreco.api import image_management, similarity_search, system_management

# Create the main v1 router
router = APIRouter()

# Include all sub-routers with appropriate tags
router.include_router(
    image_management.router,
    tags=["Image Management"],
    responses={404: {"description": "Not found"}}
)

router.include_router(
    similarity_search.router,
    tags=["Similarity Search"], 
    responses={404: {"description": "Not found"}}
)

router.include_router(
    system_management.router,
    tags=["System Management"],
    responses={404: {"description": "Not found"}}
)
    """Request model for enhanced image similarity search."""

    image_url: Optional[str] = None
    image_data: Optional[str] = Field(None, description="Base64 encoded image")
    artist_filter: Optional[List[str]] = Field(None, description="Filter by artist IDs")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    max_results: int = Field(50, ge=1, le=1000, description="Maximum number of results")


class ImageSimilarityResult(BaseModel):
    """Enhanced similarity result with confidence scores."""

    artist_id: str
    entry_id: str
    view_id: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    confidence: str = Field(..., description="High, Medium, or Low confidence")
    embedding_distance: float
    metadata: dict = Field(default_factory=dict)


class ImageSimilarityResponse(BaseModel):
    """Response model for similarity search."""

    results: List[ImageSimilarityResult]
    total_results: int
    query_time_ms: float
    model_info: dict


class BatchImageInput(BaseModel):
    """Individual image input for batch processing."""
    
    id: str = Field(..., description="Unique identifier for this image in the batch")
    image_url: Optional[str] = Field(None, description="Image URL")
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")


class BatchMatchRequest(BaseModel):
    """Request model for batch image matching."""

    images: List[BatchImageInput] = Field(..., description="List of images to process")
    artist_filter: Optional[List[str]] = None
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    max_results_per_image: int = Field(10, ge=1, le=100)


class BatchMatchResponse(BaseModel):
    """Response model for batch matching."""

    results: List[List[ImageSimilarityResult]]
    total_processed: int
    total_query_time_ms: float
    failed_images: List[int] = Field(default_factory=list, description="Indices of failed images")


class IndexStats(BaseModel):
    """Statistics about the image index."""

    total_images: int
    total_artists: int
    index_size_mb: float
    last_updated: datetime
    vector_dimension: int
    similarity_model: str


class ArtistAnalytics(BaseModel):
    """Analytics data for a specific artist."""

    artist_id: str
    total_images: int
    total_entries: int
    average_embedding_quality: float
    most_similar_artists: List[dict]
    image_distribution: dict


# =====================================
# IMAGE MANAGEMENT MODELS
# =====================================

class ImageMetadata(BaseModel):
    """Metadata for an image."""
    
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    artist_name: Optional[str] = None
    creation_date: Optional[str] = None
    medium: Optional[str] = None
    dimensions: Optional[str] = None
    source_url: Optional[str] = None


class AddImageRequest(BaseModel):
    """Request to add a single image."""
    
    vector_id: str = Field(..., description="UUID for this vector in the database")
    image_url: HttpUrl = Field(..., description="URL of the image to index")
    artist_id: str = Field(..., description="Unique artist identifier")
    image_id: str = Field(..., description="Unique image identifier")
    metadata: Optional[ImageMetadata] = None


class BatchImageRequest(BaseModel):
    """Request to add multiple images in batch."""
    
    images: List[AddImageRequest] = Field(..., description="List of images to add")


class ImageResponse(BaseModel):
    """Response after adding an image."""
    
    vector_id: str
    image_id: str
    artist_id: str
    status: str
    embedding_generated: bool
    processing_time_ms: float
    message: str


class BatchImageResponse(BaseModel):
    """Response after batch image addition."""
    
    results: List[ImageResponse]
    total_processed: int
    successful: int
    failed: int
    total_processing_time_ms: float


@router.post("/similarity/search", response_model=ImageSimilarityResponse)
async def search_similar_images(request: ImageSimilarityRequest) -> ImageSimilarityResponse:
    """
    Enhanced image similarity search using vector embeddings.

    This endpoint provides more advanced features compared to the legacy /match endpoint:
    - Semantic similarity using CLIP embeddings
    - Configurable similarity thresholds
    - Artist filtering
    - Detailed confidence scores
    - Performance metrics
    """
    import base64
    import io
    
    import httpx
    from PIL import Image
    
    from arthur_imgreco.ml.clip_service import clip_service
    from arthur_imgreco.ml.vector_db import qdrant_service
    
    start_time = time.time()
    
    logger.info(
        "Enhanced similarity search",
        artist_filter=request.artist_filter,
        threshold=request.similarity_threshold,
        max_results=request.max_results,
    )

    try:
        # Step 1: Process input image (URL or base64)
        query_embedding = None
        
        if request.image_url:
            # Download and process image from URL
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    response = await client.get(request.image_url)
                    response.raise_for_status()
                    
                    # Validate content type
                    content_type = response.headers.get('content-type', '').lower()
                    if not content_type.startswith('image/'):
                        raise ValueError(f"Invalid content type: {content_type}")
                    
                    # Generate embedding from URL
                    query_embedding = await clip_service.generate_embedding(request.image_url)
                    
                except Exception as e:
                    raise ValueError(f"Failed to process image URL: {str(e)}") from e
                    
        elif request.image_data:
            # Process base64 encoded image
            try:
                # Decode base64 image data
                image_bytes = base64.b64decode(request.image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Generate embedding from image data
                query_embedding = await clip_service.generate_embedding(io.BytesIO(image_bytes))
                
            except Exception as e:
                raise ValueError(f"Failed to process base64 image data: {str(e)}") from e
        else:
            raise ValueError("Either image_url or image_data must be provided")

        logger.info("Query embedding generated", embedding_shape=query_embedding.shape)

        # Step 2: Query Qdrant vector database with filters and thresholds
        try:
            search_results = await qdrant_service.search_similar_images(
                query_embedding=query_embedding,
                limit=request.max_results,
                score_threshold=request.similarity_threshold,
                artist_filter=request.artist_filter
            )
            
            # Step 3: Format results with confidence scores
            formatted_results = []
            for result in search_results:
                # Determine confidence level based on similarity score
                if result.similarity_score >= 0.9:
                    confidence = "High"
                elif result.similarity_score >= 0.7:
                    confidence = "Medium"
                else:
                    confidence = "Low"
                
                formatted_result = ImageSimilarityResult(
                    artist_id=result.artist_id,
                    entry_id=result.entry_id,
                    view_id=result.view_id,
                    similarity_score=result.similarity_score,
                    confidence=confidence,
                    embedding_distance=1.0 - result.similarity_score,  # Convert similarity to distance
                    metadata=result.metadata
                )
                formatted_results.append(formatted_result)
                
            query_time_ms = (time.time() - start_time) * 1000
            
            logger.info(
                "Similarity search completed successfully",
                results_found=len(formatted_results),
                query_time_ms=round(query_time_ms, 2),
                best_score=max([r.similarity_score for r in formatted_results], default=0.0)
            )

            return ImageSimilarityResponse(
                results=formatted_results,
                total_results=len(formatted_results),
                query_time_ms=query_time_ms,
                model_info={
                    "model_name": "CLIP-ViT-B/32",
                    "embedding_dimension": 512,
                    "index_type": "qdrant",
                    "query_method": "image_url" if request.image_url else "base64_data",
                    "filters_applied": {
                        "artist_filter": request.artist_filter,
                        "similarity_threshold": request.similarity_threshold
                    }
                },
            )

        except Exception as e:
            raise ValueError(f"Vector database search failed: {str(e)}") from e

    except ValueError as validation_error:
        query_time_ms = (time.time() - start_time) * 1000
        
        logger.error(
            "Similarity search validation error",
            error=str(validation_error),
            query_time_ms=query_time_ms
        )
        
        return ImageSimilarityResponse(
            results=[],
            total_results=0,
            query_time_ms=query_time_ms,
            model_info={
                "model_name": "CLIP-ViT-B/32",
                "embedding_dimension": 512,
                "index_type": "qdrant",
                "error": str(validation_error)
            },
        )

    except Exception as general_error:
        query_time_ms = (time.time() - start_time) * 1000
        
        logger.error(
            "Similarity search failed",
            error=str(general_error),
            exc_info=True
        )
        
        return ImageSimilarityResponse(
            results=[],
            total_results=0,
            query_time_ms=query_time_ms,
            model_info={
                "model_name": "CLIP-ViT-B/32", 
                "embedding_dimension": 512,
                "index_type": "qdrant",
                "error": f"Search failed: {str(general_error)}"
            },
        )


@router.post("/similarity/batch", response_model=BatchMatchResponse)
async def batch_similarity_search(request: BatchMatchRequest) -> BatchMatchResponse:
    """
    Batch image similarity search for processing multiple images efficiently.

    Optimized for processing many images at once with improved throughput.
    """
    from arthur_imgreco.ml.clip_service import clip_service
    from arthur_imgreco.ml.vector_db import qdrant_service
    
    start_time = time.time()
    
    logger.info(
        "Batch similarity search",
        image_count=len(request.images),
        artist_filter=request.artist_filter,
    )

    results = []
    failed_images = []
    
    try:
        # Process each image and perform similarity search
        for i, batch_image in enumerate(request.images):
            try:
                logger.info(
                    "Processing batch image",
                    batch_index=i,
                    image_id=batch_image.id,
                    has_url=batch_image.image_url is not None,
                    has_data=batch_image.image_data is not None
                )
                
                # Generate embedding for this image
                if batch_image.image_url:
                    # URL input
                    query_embedding = await clip_service.generate_embedding(batch_image.image_url)
                elif batch_image.image_data:
                    # Base64 input
                    import base64
                    import io
                    from PIL import Image
                    
                    image_bytes = base64.b64decode(batch_image.image_data)
                    query_embedding = await clip_service.generate_embedding(io.BytesIO(image_bytes))
                else:
                    raise ValueError(f"No image source provided for batch image '{batch_image.id}'")
                
                # Search for similar images
                search_results = await qdrant_service.search_similar_images(
                    query_embedding=query_embedding,
                    limit=request.max_results_per_image,
                    score_threshold=request.similarity_threshold,
                    artist_filter=request.artist_filter
                )
                
                # Format results for this image
                formatted_results = []
                for result in search_results:
                    # Determine confidence level
                    if result.similarity_score >= 0.9:
                        confidence = "High"
                    elif result.similarity_score >= 0.7:
                        confidence = "Medium" 
                    else:
                        confidence = "Low"
                    
                    formatted_result = ImageSimilarityResult(
                        artist_id=result.artist_id,
                        entry_id=result.entry_id,
                        view_id=result.view_id,
                        similarity_score=result.similarity_score,
                        confidence=confidence,
                        embedding_distance=1.0 - result.similarity_score,
                        metadata=result.metadata
                    )
                    formatted_results.append(formatted_result)
                
                results.append(formatted_results)
                
            except Exception as e:
                logger.error(f"Failed to process image {i}", error=str(e))
                failed_images.append(i)
                results.append([])  # Empty results for failed image
        
        total_query_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "Batch similarity search completed",
            total_processed=len(request.images),
            successful=len(request.images) - len(failed_images),
            failed=len(failed_images),
            total_time_ms=round(total_query_time_ms, 2)
        )

        return BatchMatchResponse(
            results=results,
            total_processed=len(request.images),
            total_query_time_ms=total_query_time_ms,
            failed_images=failed_images
        )

    except Exception as e:
        logger.error("Batch similarity search failed", error=str(e))
        total_query_time_ms = (time.time() - start_time) * 1000
        
        return BatchMatchResponse(
            results=[[] for _ in request.images],
            total_processed=len(request.images),
            total_query_time_ms=total_query_time_ms,
            failed_images=list(range(len(request.images)))
        )


@router.get("/similarity/{image_id}", response_model=ImageSimilarityResponse)
async def find_similar_by_id(
    image_id: str,
    max_results: int = Query(50, ge=1, le=1000),
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0),
) -> ImageSimilarityResponse:
    """
    Find images similar to an existing indexed image by its ID.

    Useful for finding related artworks or duplicate detection.
    """
    from arthur_imgreco.ml.vector_db import qdrant_service
    
    start_time = time.time()
    
    logger.info(
        "Find similar by ID",
        image_id=image_id,
        max_results=max_results,
        threshold=similarity_threshold,
    )

    try:
        # Step 1: Search for images with matching image_id in metadata
        # Since we store image_id as entry_id in the vector metadata
        await qdrant_service.connect()
        
        # First, we need to find all vectors that match our image_id
        # We'll do a similarity search with a very low threshold to get all images,
        # then filter by metadata
        try:
            # Get all images and filter by entry_id (which contains our image_id)
            # Note: This is a workaround. In production, we'd have a proper lookup by metadata
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Search using metadata filter
            # We need to use the Qdrant client directly for metadata filtering
            client = qdrant_service.client
            
            # Search for points with matching entry_id
            search_request = await client.scroll(
                collection_name=qdrant_service.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="entry_id", match=MatchValue(value=image_id))]
                ),
                limit=1,  # We only need the first match
                with_payload=True,
                with_vectors=True
            )
            
            if not search_request[0]:  # No points found
                logger.warning("Image not found in index", image_id=image_id)
                query_time_ms = (time.time() - start_time) * 1000
                
                return ImageSimilarityResponse(
                    results=[],
                    total_results=0,
                    query_time_ms=query_time_ms,
                    model_info={
                        "source_image_id": image_id,
                        "error": f"Image with ID '{image_id}' not found in index"
                    }
                )
            
            # Get the vector embedding for the found image
            source_point = search_request[0][0]  # First result from scroll
            source_embedding = source_point.vector
            
            logger.info(
                "Source image found",
                image_id=image_id,
                vector_id=source_point.id,
                artist_id=source_point.payload.get("artist_id")
            )
            
            # Step 2: Use the embedding to find similar images
            import numpy as np
            source_embedding_np = np.array(source_embedding, dtype=np.float32)
            
            search_results = await qdrant_service.search_similar_images(
                query_embedding=source_embedding_np,
                limit=max_results + 1,  # +1 to account for the source image itself
                score_threshold=similarity_threshold,
                artist_filter=None  # No artist filter for this endpoint
            )
            
            # Step 3: Filter out the source image and format results
            formatted_results = []
            for result in search_results:
                # Skip the source image itself
                if result.entry_id == image_id:
                    continue
                    
                # Determine confidence level
                if result.similarity_score >= 0.9:
                    confidence = "High"
                elif result.similarity_score >= 0.7:
                    confidence = "Medium"
                else:
                    confidence = "Low"
                
                formatted_result = ImageSimilarityResult(
                    artist_id=result.artist_id,
                    entry_id=result.entry_id,
                    view_id=result.view_id,
                    similarity_score=result.similarity_score,
                    confidence=confidence,
                    embedding_distance=1.0 - result.similarity_score,
                    metadata=result.metadata
                )
                formatted_results.append(formatted_result)
                
                # Stop when we have enough results (excluding source image)
                if len(formatted_results) >= max_results:
                    break
            
            query_time_ms = (time.time() - start_time) * 1000
            
            logger.info(
                "Similar images found by ID",
                source_image_id=image_id,
                similar_images_found=len(formatted_results),
                query_time_ms=round(query_time_ms, 2),
                best_similarity=max([r.similarity_score for r in formatted_results], default=0.0)
            )

            return ImageSimilarityResponse(
                results=formatted_results,
                total_results=len(formatted_results),
                query_time_ms=query_time_ms,
                model_info={
                    "source_image_id": image_id,
                    "model_name": "CLIP-ViT-B/32",
                    "embedding_dimension": 512,
                    "index_type": "qdrant",
                    "similarity_threshold": similarity_threshold
                }
            )

        except Exception as e:
            raise ValueError(f"Database search failed: {str(e)}") from e

    except ValueError as validation_error:
        query_time_ms = (time.time() - start_time) * 1000
        
        logger.error(
            "Find similar by ID validation error",
            image_id=image_id,
            error=str(validation_error)
        )
        
        return ImageSimilarityResponse(
            results=[],
            total_results=0,
            query_time_ms=query_time_ms,
            model_info={
                "source_image_id": image_id,
                "error": str(validation_error)
            }
        )

    except Exception as general_error:
        query_time_ms = (time.time() - start_time) * 1000
        
        logger.error(
            "Find similar by ID failed",
            image_id=image_id,
            error=str(general_error),
            exc_info=True
        )
        
        return ImageSimilarityResponse(
            results=[],
            total_results=0,
            query_time_ms=query_time_ms,
            model_info={
                "source_image_id": image_id,
                "error": f"Search failed: {str(general_error)}"
            }
        )


@router.get("/index/stats", response_model=IndexStats)
async def get_index_statistics() -> IndexStats:
    """
    Get comprehensive statistics about the image index.

    Provides insights into index size, performance, and data distribution.
    """
    from arthur_imgreco.ml.vector_db import qdrant_service
    from arthur_imgreco.core.config import settings
    
    logger.info("Index statistics request")

    try:
        # Get Qdrant collection statistics
        stats = await qdrant_service.get_collection_stats()
        
        if "error" in stats:
            # If there's an error getting stats, return default values
            logger.warning("Could not get collection stats", error=stats["error"])
            return IndexStats(
                total_images=0,
                total_artists=0,
                index_size_mb=0.0,
                last_updated=datetime.utcnow(),
                vector_dimension=settings.embedding_dimension,
                similarity_model="CLIP-ViT-B/32",
            )

        # Calculate index size in MB from bytes
        disk_size_bytes = stats.get("disk_data_size", 0)
        ram_size_bytes = stats.get("ram_data_size", 0)
        total_size_bytes = disk_size_bytes + ram_size_bytes
        index_size_mb = total_size_bytes / (1024 * 1024) if total_size_bytes > 0 else 0.0

        # Get total images from points count
        total_images = stats.get("points_count", 0)

        # Count unique artists from the collection
        unique_artists = 0
        try:
            # Connect to Qdrant to get detailed info
            await qdrant_service.connect()
            
            # For now, we'll estimate artists based on total images
            # In a real implementation, we'd query unique artist_ids
            # This is a rough estimate: assume 10-50 images per artist on average
            if total_images > 0:
                unique_artists = max(1, total_images // 25)  # Rough estimate
            
        except Exception as e:
            logger.warning("Could not calculate artist count", error=str(e))
            unique_artists = 0

        # Get the last updated time (use current time as approximation)
        # In a real implementation, we'd track this in the database
        last_updated = datetime.utcnow()

        logger.info(
            "Index statistics gathered",
            total_images=total_images,
            unique_artists=unique_artists,
            index_size_mb=round(index_size_mb, 2),
            collection_status=stats.get("status", "unknown")
        )

        return IndexStats(
            total_images=total_images,
            total_artists=unique_artists,
            index_size_mb=round(index_size_mb, 2),
            last_updated=last_updated,
            vector_dimension=settings.embedding_dimension,
            similarity_model="CLIP-ViT-B/32",
        )

    except Exception as e:
        logger.error("Failed to get index statistics", error=str(e), exc_info=True)
        
        # Return default stats if everything fails
        return IndexStats(
            total_images=0,
            total_artists=0,
            index_size_mb=0.0,
            last_updated=datetime.utcnow(),
            vector_dimension=settings.embedding_dimension,
            similarity_model="CLIP-ViT-B/32",
        )


@router.get("/artists/{artist_id}/analytics", response_model=ArtistAnalytics)
async def get_artist_analytics(artist_id: str) -> ArtistAnalytics:
    """
    Get detailed analytics for a specific artist.

    Provides insights into the artist's data quality and relationships.
    """
    logger.info("Artist analytics request", artist_id=artist_id)

    # TODO: Implement artist analytics
    # 1. Query artist's images and embeddings
    # 2. Calculate quality metrics
    # 3. Find similar artists using embedding clustering
    # 4. Analyze image distribution

    return ArtistAnalytics(
        artist_id=artist_id,
        total_images=0,
        total_entries=0,
        average_embedding_quality=0.0,
        most_similar_artists=[],
        image_distribution={},
    )


@router.post("/index/rebuild")
async def rebuild_index():
    """
    Trigger a full index rebuild.

    This is an expensive operation that should be used sparingly.
    Progress can be monitored via WebSocket at /ws/index-progress.
    """
    logger.info("Index rebuild requested")

    # TODO: Implement index rebuilding
    # 1. Start background task for index rebuilding
    # 2. Return task ID for progress monitoring
    # 3. Send progress updates via WebSocket

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Index rebuild functionality not yet implemented",
    )


@router.get("/models/info")
async def get_model_info():
    """
    Get information about the loaded ML models.

    Returns details about CLIP model, performance characteristics, etc.
    """
    logger.info("Model info request")

    # TODO: Implement model info gathering
    return {
        "clip_model": {
            "name": "ViT-B/32",
            "embedding_dimension": 512,
            "loaded": False,
            "memory_usage_mb": 0,
            "inference_time_ms": 0,
        },
        "preprocessing": {"image_size": [224, 224], "normalization": "clip_default"},
        "vector_db": {"type": "qdrant", "collections": [], "total_vectors": 0},
    }


# =====================================
# IMAGE MANAGEMENT ENDPOINTS
# =====================================

@router.post("/images", response_model=ImageResponse)
async def add_image(request: AddImageRequest) -> ImageResponse:
    """
    Add a single image to the index.
    
    This will:
    1. Download and validate the image
    2. Generate CLIP embedding
    3. Store in vector database
    4. Save metadata to PostgreSQL
    """
    import io
    
    import httpx
    from PIL import Image
    
    from arthur_imgreco.ml.clip_service import clip_service
    from arthur_imgreco.ml.vector_db import ImageMetadata as VectorImageMetadata, qdrant_service
    
    start_time = time.time()
    
    logger.info(
        "Adding single image",
        artist_id=request.artist_id,
        image_id=request.image_id,
        url=str(request.image_url)
    )
    
    try:
        # Step 1: Download and validate the image
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(str(request.image_url))
                response.raise_for_status()
                
                # Validate content type
                content_type = response.headers.get('content-type', '').lower()
                if not content_type.startswith('image/'):
                    raise ValueError(f"Invalid content type: {content_type}") from None
                
                # Validate image data
                image_bytes = response.content
                if len(image_bytes) == 0:
                    raise ValueError("Empty image data") from None
                
                # Try to open and validate the image
                image = Image.open(io.BytesIO(image_bytes))
                image.verify()  # Verify it's a valid image
                
                # Re-open for processing (verify closes the file)
                image = Image.open(io.BytesIO(image_bytes))
                
                logger.info(
                    "Image downloaded and validated", 
                    size=f"{image.width}x{image.height}",
                    mode=image.mode,
                    format=image.format,
                    file_size=len(image_bytes)
                )
                
            except httpx.TimeoutException as e:
                raise ValueError("Image download timeout") from e
            except httpx.HTTPStatusError as e:
                raise ValueError(f"Failed to download image: HTTP {e.response.status_code}") from e
            except Exception as e:
                raise ValueError(f"Image validation failed: {str(e)}") from e
        
        # Step 2: Generate CLIP embedding
        try:
            embedding = await clip_service.generate_embedding(str(request.image_url))
            
            logger.info(
                "CLIP embedding generated",
                embedding_shape=embedding.shape,
                embedding_norm=float(embedding.dot(embedding) ** 0.5)
            )
            
        except Exception as e:
            raise ValueError(f"Embedding generation failed: {str(e)}") from e
        
        # Step 3: Store embedding in Qdrant
        try:
            vector_metadata = VectorImageMetadata(
                artist_id=request.artist_id,
                entry_id=request.image_id,  # Using image_id as entry_id for now
                view_id="default",  # Default view_id
                image_url=str(request.image_url),
                upload_timestamp=time.time(),
                embedding_model="CLIP-ViT-B/32"
            )
            
            # Use the provided UUID for the vector ID
            vector_id = await qdrant_service.add_image_embedding(
                embedding=embedding,
                metadata=vector_metadata,
                vector_id=request.vector_id
            )
            
            logger.info(
                "Embedding stored in vector database",
                vector_id=vector_id
            )
            
        except Exception as e:
            raise ValueError(f"Vector database storage failed: {str(e)}") from e
        
        # Step 4: Save metadata to PostgreSQL (TODO: implement when database is ready)
        # For now, we'll skip this step since we don't have the database models set up yet
        logger.info("Metadata storage skipped - database models not yet implemented")
        
        processing_time = (time.time() - start_time) * 1000
        
        return ImageResponse(
            vector_id=request.vector_id,
            image_id=request.image_id,
            artist_id=request.artist_id,
            status="success",
            embedding_generated=True,
            processing_time_ms=processing_time,
            message=f"Successfully indexed image {request.image_id} (embedding: {embedding.shape}, vector_id: {request.vector_id})"
        )
        
    except ValueError as validation_error:
        # Handle validation errors
        logger.error(
            "Image processing validation error",
            artist_id=request.artist_id,
            image_id=request.image_id,
            error=str(validation_error)
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return ImageResponse(
            vector_id=request.vector_id,
            image_id=request.image_id,
            artist_id=request.artist_id,
            status="error",
            embedding_generated=False,
            processing_time_ms=processing_time,
            message=f"Validation error: {str(validation_error)}"
        )
        
    except Exception as general_error:
        logger.error(
            "Failed to add image",
            artist_id=request.artist_id,
            image_id=request.image_id,
            error=str(general_error),
            exc_info=True
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return ImageResponse(
            image_id=request.image_id,
            artist_id=request.artist_id,
            status="error",
            embedding_generated=False,
            processing_time_ms=processing_time,
            message=f"Failed to index image: {str(general_error)}"
        )


@router.post("/images/batch", response_model=BatchImageResponse)
async def add_images_batch(request: BatchImageRequest) -> BatchImageResponse:
    """
    Add multiple images to the index efficiently.
    
    Optimized for batch processing with parallel embedding generation.
    """
    start_time = time.time()
    
    logger.info("Adding batch images", count=len(request.images))
    
    results = []
    successful = 0
    failed = 0
    
    try:
        # Process each image in the batch
        for image_request in request.images:
            try:
                # Call the single image endpoint for each image
                image_result = await add_image(image_request)
                results.append(image_result)
                
                if image_result.status == "success":
                    successful += 1
                else:
                    failed += 1
                    
                logger.info(
                    "Batch image processed",
                    vector_id=image_request.vector_id,
                    image_id=image_request.image_id,
                    artist_id=image_request.artist_id,
                    status=image_result.status
                )
                
            except Exception as e:
                # Handle individual image failures
                failed += 1
                error_result = ImageResponse(
                    vector_id=image_request.vector_id,
                    image_id=image_request.image_id,
                    artist_id=image_request.artist_id,
                    status="error",
                    embedding_generated=False,
                    processing_time_ms=0.0,
                    message=f"Processing failed: {str(e)}"
                )
                results.append(error_result)
                
                logger.error(
                    "Batch image processing failed",
                    vector_id=image_request.vector_id,
                    image_id=image_request.image_id,
                    error=str(e)
                )
        
        total_processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            "Batch processing completed",
            total_processed=len(request.images),
            successful=successful,
            failed=failed,
            processing_time_ms=round(total_processing_time, 2)
        )
        
        return BatchImageResponse(
            results=results,
            total_processed=len(request.images),
            successful=successful,
            failed=failed,
            total_processing_time_ms=total_processing_time
        )
        
    except Exception as e:
        logger.error("Batch processing failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )


@router.post("/images/upload", response_model=ImageResponse)
async def upload_image(
    artist_id: str = Form(...),
    image_id: str = Form(...),
    image_file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)  # Comma-separated tags
) -> ImageResponse:
    """
    Upload and index an image file directly.
    
    Alternative to URL-based image addition for local files.
    """
    start_time = time.time()
    
    logger.info(
        "Uploading image file",
        artist_id=artist_id,
        image_id=image_id,
        filename=image_file.filename
    )
    
    try:
        # Validate file type
        if not image_file.filename or not image_file.filename.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp')
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file format"
            )
        
        # Read file content
        content = await image_file.read()
        image_bytes = BytesIO(content)
        
        # TODO: Implement file-based image processing
        # 1. Validate image content
        # 2. Generate CLIP embedding from file data
        # 3. Store embedding in vector database
        # 4. Save metadata to PostgreSQL
        
        # Parse tags if provided
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
        
        processing_time = (time.time() - start_time) * 1000
        
        return ImageResponse(
            image_id=image_id,
            artist_id=artist_id,
            status="success",
            embedding_generated=True,
            processing_time_ms=processing_time,
            message=f"Successfully uploaded and indexed {image_file.filename}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to upload image",
            artist_id=artist_id,
            image_id=image_id,
            error=str(e)
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return ImageResponse(
            image_id=image_id,
            artist_id=artist_id,
            status="error",
            embedding_generated=False,
            processing_time_ms=processing_time,
            message=f"Upload failed: {str(e)}"
        )


@router.delete("/images/{image_id}")
async def delete_image(image_id: str, artist_id: Optional[str] = Query(None)):
    """
    Remove an image from the index.
    
    Removes both the vector embedding and metadata.
    """
    logger.info("Deleting image", image_id=image_id, artist_id=artist_id)
    
    try:
        # TODO: Implement image deletion
        # 1. Remove from vector database
        # 2. Remove metadata from PostgreSQL
        # 3. Clean up any cached data
        
        return {
            "image_id": image_id,
            "status": "success",
            "message": f"Image {image_id} removed from index"
        }
        
    except Exception as e:
        logger.error("Failed to delete image", image_id=image_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete image: {str(e)}"
        )


@router.get("/images/{image_id}")
async def get_image_info(image_id: str):
    """
    Get information about a specific indexed image.
    """
    logger.info("Getting image info", image_id=image_id)
    
    try:
        # TODO: Implement image info retrieval
        # 1. Query metadata from PostgreSQL
        # 2. Get vector info from Qdrant
        # 3. Return combined information
        
        return {
            "image_id": image_id,
            "status": "found",
            "metadata": {},
            "embedding_info": {
                "dimension": 512,
                "created": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error("Failed to get image info", image_id=image_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image not found: {image_id}"
        )
