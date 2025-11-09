"""
Similarity Search API endpoints - Finding similar images using CLIP embeddings.
"""

import time
from fastapi import APIRouter, Query

from core.logging import get_logger
from core.config import settings
from core.responses import success_response, error_response
from src.core.models import (
    ImageSimilarityRequest,
    ImageSimilarityResponse,
    ImageSimilarityResult,
    BatchMatchRequest,
    BatchMatchResponse,
)

logger = get_logger(__name__)
router = APIRouter()


# BASE PATH: /api/v1/match


@router.post("/")
async def similarity_search(request: ImageSimilarityRequest):
    """
    Find images similar to a provided image (URL or base64).

    Uses CLIP embeddings for semantic similarity matching.
    Supports both image URLs and base64-encoded image data.
    """
    from ml.clip_service import clip_service
    from ml.vector_db import qdrant_service
    from core.cache import get_cache_service

    start_time = time.time()
    cache_service = await get_cache_service()

    logger.info(
        "Similarity search",
        has_url=request.image_url is not None,
        has_data=request.image_data is not None,
        max_results=request.max_results,
        threshold=request.similarity_threshold,
        artist_filter=request.artist_filter,
    )

    try:
        # Step 1: Generate embedding for query image
        if request.image_url:
            # URL-based input
            query_embedding = await clip_service.generate_embedding(str(request.image_url))
            query_method = "image_url"

            logger.info(
                "Query embedding generated from URL",
                url=str(request.image_url),
                embedding_shape=query_embedding.shape,
            )

        elif request.image_data:
            # Base64-encoded image data
            import base64
            import io
            from PIL import Image

            try:
                image_bytes = base64.b64decode(request.image_data)
                query_embedding = await clip_service.generate_embedding(io.BytesIO(image_bytes))
                query_method = "base64_data"

                logger.info(
                    "Query embedding generated from base64",
                    data_size=len(image_bytes),
                    embedding_shape=query_embedding.shape,
                )

            except Exception as e:
                raise ValueError(f"Invalid base64 image data: {str(e)}") from e
        else:
            raise ValueError("Either image_url or image_data must be provided")

        # Step 2: Search for similar images in vector database
        try:
            await qdrant_service.connect()

            search_results = await qdrant_service.search_similar_images(
                query_embedding=query_embedding,
                limit=request.max_results,
                score_threshold=request.similarity_threshold,
                artist_filter=request.artist_filter,
            )

            logger.info(
                "Vector search completed",
                results_found=len(search_results),
                best_score=max([r.similarity_score for r in search_results], default=0.0),
            )

        except Exception as e:
            raise ValueError(f"Vector database search failed: {str(e)}") from e

        # Step 3: Format results
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
                embedding_distance=1.0 - result.similarity_score,
                metadata=result.metadata,
            )
            formatted_results.append(formatted_result)

        query_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Similarity search completed",
            results_returned=len(formatted_results),
            query_time_ms=round(query_time_ms, 2),
            best_confidence=formatted_results[0].confidence if formatted_results else "none",
        )

        similarity_response = ImageSimilarityResponse(
            results=formatted_results,
            total_results=len(formatted_results),
            query_time_ms=query_time_ms,
            model_info={
                "model_name": f"CLIP-{settings.clip_model_name}",
                "embedding_dimension": settings.embedding_dimension,
                "index_type": "qdrant",
                "query_method": query_method,
                "filters_applied": {
                    "artist_filter": request.artist_filter,
                    "similarity_threshold": request.similarity_threshold,
                },
            },
        )

        # Success
        return success_response(
            message=f"Found {len(formatted_results)} similar images",
            search_results=similarity_response.dict(),
        )

    # Validation error
    except ValueError as validation_error:
        query_time_ms = (time.time() - start_time) * 1000
        logger.error("Similarity search validation error", error=str(validation_error))
        return error_response(
            message="Similarity search validation error",
            details=str(validation_error),
            query_time_ms=query_time_ms,
        )

    # General error
    except Exception as general_error:
        query_time_ms = (time.time() - start_time) * 1000
        logger.error("Similarity search failed", error=str(general_error), exc_info=True)
        return error_response(
            message="Similarity search failed",
            details=str(general_error),
            query_time_ms=query_time_ms,
            status_code=500,
        )


@router.post("/batch")
async def batch_similarity_search(request: BatchMatchRequest):
    """
    Batch image similarity search for processing multiple images efficiently.

    Optimized for processing many images at once with improved throughput.
    """
    from ml.clip_service import clip_service
    from ml.vector_db import qdrant_service

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
                    has_data=batch_image.image_data is not None,
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
                    artist_filter=request.artist_filter,
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
                        metadata=result.metadata,
                    )
                    formatted_results.append(formatted_result)

                results.append(formatted_results)

                logger.info(
                    "Batch image processed successfully",
                    batch_index=i,
                    image_id=batch_image.id,
                    results_found=len(formatted_results),
                )

            except Exception as e:
                # Track failed images
                failed_images.append(i)
                results.append([])  # Empty results for failed image

                logger.error(
                    "Batch image processing failed",
                    batch_index=i,
                    image_id=batch_image.id,
                    error=str(e),
                )

        query_time_ms = (time.time() - start_time) * 1000

        logger.info(
            "Batch similarity search completed",
            images_processed=len(request.images),
            images_failed=len(failed_images),
            query_time_ms=round(query_time_ms, 2),
        )

        batch_response = BatchMatchResponse(
            results=results,
            total_processed=len(request.images),
            total_query_time_ms=query_time_ms,
            failed_images=failed_images,
        )

        return success_response(
            message=f"Processed {len(request.images)} images, {len(failed_images)} failed",
            batch_results=batch_response.dict(),
        )

    except Exception as general_error:
        logger.error("Batch similarity search failed", error=str(general_error), exc_info=True)

        return error_response(
            message="Batch similarity search failed",
            details=str(general_error),
            total_query_time_ms=(time.time() - start_time) * 1000,
            failed_images=list(range(len(request.images))),
            status_code=500,
        )


@router.get("/{entry_id}")
async def find_similar_by_id(
    entry_id: str,
    max_results: int = Query(50, ge=1, le=1000),
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0),
):
    """
    Find images similar to an existing indexed image by its ID.

    Useful for finding related artworks or duplicate detection.
    """
    from ml.vector_db import qdrant_service

    start_time = time.time()

    logger.info(
        "Find similar by ID",
        entry_id=entry_id,
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
                    must=[FieldCondition(key="entry_id", match=MatchValue(value=entry_id))]
                ),
                limit=1,  # We only need the first match
                with_payload=True,
                with_vectors=True,
            )

            if not search_request[0]:  # No points found
                logger.warning("Image not found in index", image_id=entry_id)
                query_time_ms = (time.time() - start_time) * 1000

                return error_response(
                    message="Image not found in index",
                    details=f"Image with ID '{entry_id}' not found in index",
                    source_image_id=entry_id,
                    query_time_ms=query_time_ms,
                    status_code=404,
                )

            # Get the vector embedding for the found image
            source_point = search_request[0][0]  # First result from scroll
            source_embedding = source_point.vector

            logger.info(
                "Source image found",
                image_id=entry_id,
                vector_id=source_point.id,
                artist_id=source_point.payload.get("artist_id"),
            )

            # Step 2: Use the embedding to find similar images
            import numpy as np

            source_embedding_np = np.array(source_embedding, dtype=np.float32)

            search_results = await qdrant_service.search_similar_images(
                query_embedding=source_embedding_np,
                limit=max_results + 1,  # +1 to account for the source image itself
                score_threshold=similarity_threshold,
                artist_filter=None,  # No artist filter for this endpoint
            )

            # Step 3: Filter out the source image and format results
            formatted_results = []
            for result in search_results:
                # Skip the source image itself
                if result.entry_id == entry_id:
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
                    metadata=result.metadata,
                )
                formatted_results.append(formatted_result)

                # Stop when we have enough results (excluding source image)
                if len(formatted_results) >= max_results:
                    break

            query_time_ms = (time.time() - start_time) * 1000

            logger.info(
                "Similar images found by ID",
                source_image_id=entry_id,
                similar_images_found=len(formatted_results),
                query_time_ms=round(query_time_ms, 2),
                best_similarity=max([r.similarity_score for r in formatted_results], default=0.0),
            )

            similarity_response = ImageSimilarityResponse(
                results=formatted_results,
                total_results=len(formatted_results),
                query_time_ms=query_time_ms,
                model_info={
                    "source_image_id": entry_id,
                    "model_name": f"CLIP-{settings.clip_model_name}",
                    "embedding_dimension": settings.embedding_dimension,
                    "index_type": "qdrant",
                    "similarity_threshold": similarity_threshold,
                },
            )

            return success_response(
                message=f"Found {len(formatted_results)} similar images for {entry_id}",
                search_results=similarity_response.dict(),
            )

        except Exception as e:
            raise ValueError(f"Database search failed: {str(e)}") from e

    except ValueError as validation_error:
        query_time_ms = (time.time() - start_time) * 1000

        logger.error(
            "Find similar by ID validation error", image_id=entry_id, error=str(validation_error)
        )

        return error_response(
            message="Find similar by ID validation error",
            details=str(validation_error),
            source_image_id=entry_id,
            query_time_ms=query_time_ms,
        )

    except Exception as general_error:
        query_time_ms = (time.time() - start_time) * 1000

        logger.error(
            "Find similar by ID failed", image_id=entry_id, error=str(general_error), exc_info=True
        )

        return error_response(
            message="Find similar by ID failed",
            details=str(general_error),
            source_image_id=entry_id,
            query_time_ms=query_time_ms,
            status_code=500,
        )
