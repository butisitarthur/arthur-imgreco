"""
Image Management API endpoints - Adding, updating, deleting images from the index.
"""

import time
from datetime import datetime

from fastapi import APIRouter, status, Request
from qdrant_client.models import PointStruct, OptimizersConfigDiff

from core.logging import get_logger
from core.config import settings
from core.responses import success_response, error_response
from core.models import AddImageRequest, BatchImageRequest, ImageResponse, BatchImageResponse

logger = get_logger(__name__)
router = APIRouter()


# BASE PATH: /api/v1/index


@router.post("/", summary="Add single image from URL")
async def add_image_from_url(request: AddImageRequest):
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

    from ml.clip_service import clip_service
    from ml.vector_db import ImageMetadata as VectorImageMetadata, qdrant_service

    start_time = time.time()

    # Get the final vector_id and hierarchical components
    final_vector_id = request.get_vector_id()
    artist_id, entry_id, view_id = request.get_hierarchical_components()

    logger.info(
        "Adding single image",
        artist_id=artist_id,
        entry_id=entry_id,
        view_id=view_id,
        vector_id=final_vector_id,
        url=str(request.image_url),
    )

    try:
        # Step 1: Download and validate the image
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                logger.info("Downloading image", url=str(request.image_url))

                response = await client.get(str(request.image_url))
                response.raise_for_status()

                image_bytes = response.content

                # Validate image data
                if not image_bytes:
                    raise ValueError("Downloaded image is empty")

                # Try to open and validate the image
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    image.verify()  # Verify it's a valid image

                    # Get basic image info for metadata
                    image = Image.open(io.BytesIO(image_bytes))  # Reopen after verify
                    image_info = dict(
                        format=image.format,
                        size=f"{image.width}x{image.height}",
                        mode=image.mode,
                        file_size=len(image_bytes),
                    )

                except Exception as e:
                    raise ValueError(f"Invalid image format: {str(e)}") from e

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
                embedding_norm=float(embedding.dot(embedding) ** 0.5),
            )

        except Exception as e:
            raise ValueError(f"Embedding generation failed: {str(e)}") from e

        # Step 3: Store embedding in Qdrant
        try:
            vector_metadata = VectorImageMetadata(
                artist_id=artist_id or "unknown",
                entry_id=entry_id or final_vector_id,  # Use entry_id or fall back to vector_id
                view_id=view_id or "main",
                image_url=str(request.image_url),
                upload_timestamp=time.time(),
                embedding_model=f"CLIP-{settings.clip_model_name}",
            )

            # Use the final vector_id (either provided or auto-generated)
            vector_id = await qdrant_service.add_image_embedding(
                embedding=embedding, metadata=vector_metadata, vector_id=final_vector_id
            )

            logger.info("Embedding stored in vector database", vector_id=vector_id)

        except Exception as e:
            raise ValueError(f"Vector database storage failed: {str(e)}") from e

        # Step 4: Save metadata to PostgreSQL (TODO: implement when database is ready)
        # For now, we'll skip this step since we don't have the database models set up yet
        logger.info("Metadata storage skipped - database models not yet implemented")

        processing_time = (time.time() - start_time) * 1000

        image_response = ImageResponse(
            vector_id=final_vector_id,
            artist_id=artist_id,
            entry_id=entry_id,
            view_id=view_id,
            hierarchical_id=request.get_hierarchical_id(),
            status="success",
            embedding_generated=True,
            processing_time_ms=processing_time,
            message=f"Successfully indexed image {entry_id or final_vector_id} (embedding: {embedding.shape})",
        )

        return success_response(
            message=f"Successfully indexed image {entry_id or final_vector_id}",
            image=image_response.dict(),
        )

    except ValueError as validation_error:
        # Handle validation errors
        logger.error(
            "Image processing validation error",
            artist_id=artist_id,
            entry_id=entry_id,
            vector_id=final_vector_id,
            error=str(validation_error),
        )

        processing_time = (time.time() - start_time) * 1000

        image_response = ImageResponse(
            vector_id=final_vector_id,
            artist_id=artist_id,
            entry_id=entry_id,
            view_id=view_id,
            hierarchical_id=request.get_hierarchical_id(),
            status="error",
            embedding_generated=False,
            processing_time_ms=processing_time,
            message=f"Validation error: {str(validation_error)}",
        )

        return error_response(
            message="Validation error occurred",
            details=str(validation_error),
            image=image_response.dict(),
        )

    except Exception as general_error:
        logger.error(
            "Failed to add image",
            artist_id=artist_id,
            entry_id=entry_id,
            vector_id=final_vector_id,
            error=str(general_error),
            exc_info=True,
        )

        return error_response(
            message="Failed to add image",
            details=str(general_error),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post("/batch", summary="Add multiple images from URLs")
async def add_images_from_url_batch(request: BatchImageRequest):
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
                image_result = await add_image_from_url(image_request)

                # Extract the ImageResponse data from success_response
                if hasattr(image_result, "body"):
                    import json

                    result_data = json.loads(image_result.body.decode())

                    if result_data.get("success"):
                        # Extract image data from success response and create ImageResponse
                        image_data = result_data.get("image", {})
                        image_response = ImageResponse(**image_data)
                        results.append(image_response)
                        successful += 1

                        status_msg = "success"
                    else:
                        # Create error ImageResponse from error response
                        image_data = result_data.get("image", {})
                        if not image_data:
                            # Fallback if no image data in error response
                            image_data = {
                                "vector_id": image_request.get_vector_id(),
                                "artist_id": image_request.artist_id,
                                "entry_id": image_request.entry_id,
                                "view_id": image_request.view_id,
                                "hierarchical_id": image_request.get_hierarchical_id(),
                                "status": "error",
                                "embedding_generated": False,
                                "processing_time_ms": 0.0,
                                "message": result_data.get("details", "Processing failed"),
                            }

                        image_response = ImageResponse(**image_data)
                        results.append(image_response)
                        failed += 1
                        status_msg = "error"
                else:
                    # Direct response case (shouldn't happen with current implementation)
                    failed += 1
                    error_image_response = ImageResponse(
                        vector_id=image_request.get_vector_id(),
                        artist_id=image_request.artist_id,
                        entry_id=image_request.entry_id,
                        view_id=image_request.view_id,
                        hierarchical_id=image_request.get_hierarchical_id(),
                        status="error",
                        embedding_generated=False,
                        processing_time_ms=0.0,
                        message="Unexpected response format",
                    )
                    results.append(error_image_response)
                    status_msg = "error"

                logger.info(
                    "Batch image processed",
                    vector_id=image_request.get_vector_id(),
                    artist_id=image_request.artist_id,
                    entry_id=image_request.entry_id,
                    status=status_msg,
                )

            except Exception as e:
                # Handle individual image failures
                failed += 1
                error_image_response = ImageResponse(
                    vector_id=image_request.get_vector_id(),
                    artist_id=image_request.artist_id,
                    entry_id=image_request.entry_id,
                    view_id=image_request.view_id,
                    hierarchical_id=image_request.get_hierarchical_id(),
                    status="error",
                    embedding_generated=False,
                    processing_time_ms=0.0,
                    message=f"Individual image processing failed: {str(e)}",
                )
                results.append(error_image_response)

        total_processing_time = (time.time() - start_time) * 1000

        logger.info(
            "Batch processing completed",
            total_processed=len(request.images),
            successful=successful,
            failed=failed,
            processing_time_ms=round(total_processing_time, 2),
        )

        batch_response = BatchImageResponse(
            results=results,
            total_processed=len(request.images),
            successful=successful,
            failed=failed,
            total_processing_time_ms=round(total_processing_time, 2),
        )

        return success_response(
            message=f"Batch processing completed: {successful} successful, {failed} failed",
            batch=batch_response.dict(),
        )

    except Exception as e:
        logger.error("Batch processing failed", error=str(e))
        return error_response(
            message="Batch processing failed",
            details=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


async def _delete_by_id(
    qdrant_service,
    target_id: str,
    id_type: str,
):
    """
    Shared deletion logic for single or batch operations.

    Args:
        qdrant_service: The Qdrant service instance
        target_id: The ID to delete
        id_type: Type of ID ('artist_id', 'entry_id', 'view_id', or 'vector_id')

    Returns:
        Tuple of (success: bool, deleted_count: int, vector_ids: list, error: str or None)
    """
    try:
        # Determine filter key and limit based on ID type
        filter_config = {
            "artist_id": ("artist_id", 10000),
            "entry_id": ("entry_id", 100),
            "view_id": ("view_id", 1),
            "vector_id": (None, 0),  # Special case: direct deletion
        }

        filter_key, limit = filter_config.get(id_type, (None, 0))

        # For vector_id, do direct deletion
        if id_type == "vector_id":
            deleted = await qdrant_service.client.delete(
                collection_name=qdrant_service.collection_name, points_selector=[target_id]
            )
            if deleted:
                return True, 1, [target_id], None
            else:
                return False, 0, [], "Vector ID not found"

        # For metadata-based deletion
        if filter_key:
            search_results = await qdrant_service.client.scroll(
                collection_name=qdrant_service.collection_name,
                scroll_filter={"must": [{"key": filter_key, "match": {"value": target_id}}]},
                limit=limit,
            )

            if not search_results[0]:
                return False, 0, [], f"No vectors found for {id_type}: {target_id}"

            vector_ids_to_delete = [point.id for point in search_results[0]]

            deleted = await qdrant_service.client.delete(
                collection_name=qdrant_service.collection_name,
                points_selector=vector_ids_to_delete,
            )

            return True, len(vector_ids_to_delete), vector_ids_to_delete, None

        return False, 0, [], f"Invalid ID type: {id_type}"

    except Exception as e:
        return False, 0, [], str(e)


@router.delete("/artist/{target_id}", summary="Delete artist images from index by artist_id")
@router.delete("/entry/{target_id}", summary="Delete entry images from index by entry_id")
@router.delete("/view/{target_id}", summary="Delete single image from index by view_id")
@router.delete("/vector/{target_id}", summary="Delete images from index by vector_id")
async def delete_image(request: Request, target_id: str):
    """
    Delete an image from the index.

    This will remove both the vector embedding and metadata.
    Can delete by artist_id, entry_id, view_id, or vector_id (UUID).
    """
    from ml.vector_db import qdrant_service

    # Determine ID type from path
    path = request.url.path
    if "/artist/" in path:
        id_type = "artist_id"
    elif "/entry/" in path:
        id_type = "entry_id"
    elif "/view/" in path:
        id_type = "view_id"
    elif "/vector/" in path:
        id_type = "vector_id"
    else:
        return error_response(
            message="Invalid delete path",
            details=f"Path: {path}",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    logger.info(f"Deleting image by {id_type}", id_type=id_type, target_id=target_id)

    try:
        await qdrant_service.connect()

        success, deleted_count, vector_ids, error = await _delete_by_id(
            qdrant_service, target_id, id_type
        )

        if success:
            logger.info(
                f"Successfully deleted {deleted_count} vectors",
                id_type=id_type,
                target_id=target_id,
                deleted_count=deleted_count,
            )

            return success_response(
                message=f"Successfully deleted {deleted_count} vector(s) for {id_type}: {target_id}",
                deletion={
                    "deleted_vector_ids": vector_ids,
                    "deleted_count": deleted_count,
                    id_type: target_id,
                },
            )
        else:
            logger.warning(
                f"Failed to delete by {id_type}",
                id_type=id_type,
                target_id=target_id,
                error=error,
            )
            return error_response(
                message=f"Failed to delete by {id_type}",
                details=error,
                status_code=(
                    status.HTTP_404_NOT_FOUND
                    if "not found" in error.lower()
                    else status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
            )

    except Exception as e:
        logger.error(
            f"Failed to delete by {id_type}",
            id_type=id_type,
            target_id=target_id,
            error=str(e),
            exc_info=True,
        )
        return error_response(
            message=f"Failed to delete by {id_type}",
            details=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.delete("/batch", summary="Delete multiple images in batch")
async def delete_images_batch(
    request: dict,
):
    """
    Delete multiple images in batch.

    Request body should contain a list of deletions with id_type and target_id:
    {
        "deletions": [
            {"id_type": "view_id", "target_id": "image123"},
            {"id_type": "entry_id", "target_id": "entry456"},
            {"id_type": "artist_id", "target_id": "artist789"}
        ]
    }
    """
    from ml.vector_db import qdrant_service

    deletions = request.get("deletions", [])

    if not deletions:
        return error_response(
            message="No deletions specified",
            details="Request must include 'deletions' array",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    logger.info("Starting batch deletion", deletion_count=len(deletions))

    try:
        await qdrant_service.connect()

        results = []
        total_deleted = 0
        successful = 0
        failed = 0

        for deletion in deletions:
            id_type = deletion.get("id_type")
            target_id = deletion.get("target_id")

            if not id_type or not target_id:
                results.append(
                    {
                        "id_type": id_type,
                        "target_id": target_id,
                        "success": False,
                        "deleted_count": 0,
                        "error": "Missing id_type or target_id",
                    }
                )
                failed += 1
                continue

            # Validate id_type
            if id_type not in ["artist_id", "entry_id", "view_id", "vector_id"]:
                results.append(
                    {
                        "id_type": id_type,
                        "target_id": target_id,
                        "success": False,
                        "deleted_count": 0,
                        "error": f"Invalid id_type: {id_type}",
                    }
                )
                failed += 1
                continue

            # Use shared deletion logic
            success, deleted_count, vector_ids, error = await _delete_by_id(
                qdrant_service, target_id, id_type
            )

            if success:
                results.append(
                    {
                        "id_type": id_type,
                        "target_id": target_id,
                        "success": True,
                        "deleted_count": deleted_count,
                        "deleted_vector_ids": vector_ids,
                    }
                )
                total_deleted += deleted_count
                successful += 1
            else:
                results.append(
                    {
                        "id_type": id_type,
                        "target_id": target_id,
                        "success": False,
                        "deleted_count": 0,
                        "error": error,
                    }
                )
                failed += 1

        logger.info(
            "Batch deletion completed",
            total_deletions=len(deletions),
            successful=successful,
            failed=failed,
            total_vectors_deleted=total_deleted,
        )

        return success_response(
            message=f"Batch deletion completed: {successful} successful, {failed} failed",
            batch_deletion={
                "total_deletions": len(deletions),
                "successful": successful,
                "failed": failed,
                "total_vectors_deleted": total_deleted,
                "results": results,
            },
        )

    except Exception as e:
        logger.error("Batch deletion failed", error=str(e), exc_info=True)
        return error_response(
            message="Batch deletion failed",
            details=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post("/rebuild")
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

        logger.info(
            "Starting index rebuild process", collection=collection_name, backup=backup_name
        )

        # Step 1: Create backup collection
        try:
            # Get current collection info
            collection_info = await qdrant_service.client.get_collection(collection_name)

            logger.info(
                "Collection info before backup",
                collection_name=collection_name,
                vectors_count=collection_info.vectors_count,
                status=collection_info.status.name if collection_info.status else "unknown",
            )

            # Create backup collection with same configuration
            await qdrant_service.client.create_collection(
                collection_name=backup_name, vectors_config=collection_info.config.params.vectors
            )

            # Copy all points to backup with proper pagination
            all_points = []
            offset = None

            # First, let's try a simple scroll without pagination to debug
            logger.info("Starting simple scroll debug test")
            test_scroll = await qdrant_service.client.scroll(
                collection_name=collection_name,
                limit=5,  # Very small test
                with_payload=True,
                with_vectors=False,  # Start without vectors
            )

            logger.info(
                "Test scroll without vectors",
                test_result_type=type(test_scroll),
                test_result_length=(
                    len(test_scroll) if hasattr(test_scroll, "__len__") else "unknown"
                ),
                test_points=len(test_scroll[0]) if len(test_scroll) > 0 and test_scroll[0] else 0,
            )

            # Now try with vectors
            test_scroll_with_vectors = await qdrant_service.client.scroll(
                collection_name=collection_name,
                limit=5,  # Very small test
                with_payload=True,
                with_vectors=True,  # Now with vectors
            )

            logger.info(
                "Test scroll with vectors",
                test_with_vectors_type=type(test_scroll_with_vectors),
                test_with_vectors_length=(
                    len(test_scroll_with_vectors)
                    if hasattr(test_scroll_with_vectors, "__len__")
                    else "unknown"
                ),
                test_with_vectors_points=(
                    len(test_scroll_with_vectors[0])
                    if len(test_scroll_with_vectors) > 0 and test_scroll_with_vectors[0]
                    else 0
                ),
            )

            # If the test worked, continue with the full scroll
            while True:
                scroll_result = await qdrant_service.client.scroll(
                    collection_name=collection_name,
                    limit=1000,  # Smaller batch size
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )

                points, next_offset = scroll_result

                logger.info(
                    "Scroll batch received",
                    batch_size=len(points) if points else 0,
                    next_offset=next_offset,
                    total_so_far=len(all_points),
                )

                if not points:
                    logger.info("No more points in this batch, stopping pagination")
                    break

                all_points.extend(points)

                if next_offset is None:
                    logger.info("No more pages available, stopping pagination")
                    break

                offset = next_offset

            logger.info("All points collected", total_points=len(all_points))

            if all_points:  # If there are points to backup
                logger.info(
                    "Converting records to PointStruct format", record_count=len(all_points)
                )

                # Convert Record objects to PointStruct format
                backup_points = [
                    PointStruct(id=record.id, vector=record.vector, payload=record.payload)
                    for record in all_points
                ]

                logger.info(
                    "PointStruct conversion completed", backup_points_count=len(backup_points)
                )

                await qdrant_service.client.upsert(
                    collection_name=backup_name, points=backup_points
                )

                logger.info("Backup upsert completed successfully")

            logger.info("Backup created successfully", backup_points=len(all_points))

        except Exception as e:
            logger.error("Failed to create backup", error=str(e))
            return error_response(message="Failed to create backup", details=str(e), status="error")

        # Step 2: Recreate main collection
        try:
            # Delete current collection
            await qdrant_service.client.delete_collection(collection_name)

            # Recreate with same configuration
            orig_opt_config = collection_info.config.optimizer_config
            logger.info(f"Original optimizer config: {orig_opt_config}")

            # # Get the indexing threshold to preserve
            # indexing_threshold = 20000  # Default
            # if (
            #     hasattr(orig_opt_config, "indexing_threshold")
            #     and orig_opt_config.indexing_threshold is not None
            # ):
            #     indexing_threshold = orig_opt_config.indexing_threshold
            #     logger.info(f"Will preserve indexing_threshold: {indexing_threshold}")

            # For now, create collection with defaults and then update settings
            await qdrant_service.client.create_collection(
                collection_name=collection_name,
                vectors_config=collection_info.config.params.vectors,
                # Preserve the indexing threshold or use 1000 as default
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=settings.indexing_threshold,
                ),
            )

            # If we had a custom indexing threshold, we would update it here
            # but Qdrant doesn't have an update_collection API for optimizer settings
            logger.info(
                "Collection recreated (indexing_threshold will use default until first optimization)"
            )

            logger.info("Main collection recreated successfully")

        except Exception as e:
            logger.error("Failed to recreate collection", error=str(e))
            # Try to restore from backup
            try:
                backup_points = await qdrant_service.client.scroll(
                    collection_name=backup_name, limit=10000, with_payload=True, with_vectors=True
                )
                if backup_points[0]:
                    # Convert Record objects to PointStruct format
                    restore_points = [
                        PointStruct(id=record.id, vector=record.vector, payload=record.payload)
                        for record in backup_points[0]
                    ]

                    await qdrant_service.client.upsert(
                        collection_name=collection_name, points=restore_points
                    )
                logger.info("Restored from backup after recreation failure")
            except Exception:
                logger.error("Failed to restore from backup")

            return error_response(
                message="Failed to recreate collection", details=str(e), status="error"
            )

        # Step 3: Restore data from backup with proper pagination
        try:
            all_backup_points = []
            offset = None

            while True:
                backup_scroll_result = await qdrant_service.client.scroll(
                    collection_name=backup_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )

                points, next_offset = backup_scroll_result

                logger.info(
                    "Backup scroll batch received",
                    batch_size=len(points) if points else 0,
                    next_offset=next_offset,
                    total_so_far=len(all_backup_points),
                )

                if not points:
                    logger.info("No more backup points in this batch")
                    break

                all_backup_points.extend(points)

                if next_offset is None:
                    logger.info("No more backup pages available")
                    break

                offset = next_offset

            logger.info("All backup points collected", total_points=len(all_backup_points))

            if all_backup_points:
                # Convert Record objects to PointStruct format
                restore_points = [
                    PointStruct(id=record.id, vector=record.vector, payload=record.payload)
                    for record in all_backup_points
                ]

                logger.info(
                    "Converting backup points for restore", restore_points_count=len(restore_points)
                )

                await qdrant_service.client.upsert(
                    collection_name=collection_name, points=restore_points
                )

                logger.info("Data restored from backup", restored_points=len(all_backup_points))

            # Clean up backup collection
            await qdrant_service.client.delete_collection(backup_name)

        except Exception as e:
            logger.error("Failed to restore data from backup", error=str(e))
            return error_response(
                message="Index recreated but data restoration failed",
                details=f"{str(e)}. Backup available at {backup_name}",
                status="partial_success",
                backup_collection=backup_name,
            )

        # Step 4: Verify index integrity
        try:
            final_info = await qdrant_service.client.get_collection(collection_name)

            # Use indexed_vectors_count if available, otherwise fall back to points_count
            vectors_restored = final_info.indexed_vectors_count
            if vectors_restored is None or vectors_restored == 0:
                vectors_restored = final_info.points_count or 0

            logger.info(
                "Index rebuild completed successfully",
                indexed_vectors=final_info.indexed_vectors_count,
                points_count=final_info.points_count,
                vectors_restored=vectors_restored,
            )

            return success_response(
                message="Index rebuilt successfully",
                result={
                    "collection_status": final_info.status.name if final_info.status else "unknown",
                    "vectors_restored": vectors_restored,
                    "indexed_vectors": final_info.indexed_vectors_count or 0,
                    "stored_points": final_info.points_count or 0,
                },
            )

        except Exception as e:
            logger.error("Failed to verify rebuilt index", error=str(e))
            return success_response(
                message="Index rebuilt but verification failed",
                details=str(e),
                status="success_unverified",
            )

    except Exception as e:
        logger.error("Index rebuild failed", error=str(e))
        return error_response(message="Index rebuild failed", details=str(e), status="error")
