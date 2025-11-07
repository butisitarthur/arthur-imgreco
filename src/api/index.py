"""
Image Management API endpoints - Adding, updating, deleting images from the index.
"""

import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, status, File, Form, UploadFile
from qdrant_client.models import PointStruct, OptimizersConfigDiff

from core.logging import get_logger
from core.config import settings
from core.responses import success_response, error_response
from src.core.models import AddImageRequest, BatchImageRequest, ImageResponse, BatchImageResponse

logger = get_logger(__name__)
router = APIRouter()


# BASE PATH: /api/v1/index


@router.post("/add_file", summary="Add single image from file")
async def upload_image_file(
    vector_id: str = Form(..., description="UUID for this vector in the database"),
    artist_id: str = Form(..., description="Unique artist identifier"),
    image_id: str = Form(..., description="Unique image identifier"),
    imgFile: UploadFile = File(..., description="Image file to upload"),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """
    Upload an image file directly instead of providing a URL.

    Useful for images that are not publicly accessible via URL.
    """
    import io
    from PIL import Image

    from ml.clip_service import clip_service
    from ml.vector_db import ImageMetadata as VectorImageMetadata, qdrant_service

    start_time = time.time()

    logger.info(
        "Uploading image file",
        artist_id=artist_id,
        image_id=image_id,
        vector_id=vector_id,
        filename=imgFile.filename,
        content_type=imgFile.content_type,
    )

    try:
        # Step 1: Validate file and read content
        if not imgFile.filename:
            raise ValueError("No filename provided")

        # Check file size (limit to 10MB)
        content = await imgFile.read()
        if not content:
            raise ValueError("Empty file provided")

        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError("File too large (max 10MB)")

        # Step 2: Validate image format
        try:
            image = Image.open(io.BytesIO(content))
            image.verify()  # Verify it's a valid image

            # Get basic image info for metadata
            image = Image.open(io.BytesIO(content))  # Reopen after verify
            image_info = dict(
                format=image.format,
                size=f"{image.width}x{image.height}",
                mode=image.mode,
                file_size=len(content),
                filename=imgFile.filename,
            )

            logger.info("Image file validated", image_info=image_info, vector_id=vector_id)

        except Exception as e:
            raise ValueError(f"Invalid image format: {str(e)}") from e

        # Step 3: Generate CLIP embedding from image data
        try:
            # Create a temporary BytesIO for CLIP processing
            image_buffer = io.BytesIO(content)
            embedding = await clip_service.generate_embedding(image_buffer)

            logger.info(
                "CLIP embedding generated from file",
                embedding_shape=embedding.shape,
                embedding_norm=float(embedding.dot(embedding) ** 0.5),
                vector_id=vector_id,
            )

        except Exception as e:
            raise ValueError(f"Embedding generation failed: {str(e)}") from e

        # Step 4: Store embedding in Qdrant
        try:
            vector_metadata = VectorImageMetadata(
                artist_id=artist_id,
                entry_id=image_id,
                view_id="default",
                image_url=f"file://{imgFile.filename}",  # Use file:// scheme for uploaded files
                upload_timestamp=time.time(),
                embedding_model=f"CLIP-{settings.clip_model_name}",
            )

            # Use the provided UUID for the vector ID
            stored_vector_id = await qdrant_service.add_image_embedding(
                embedding=embedding, metadata=vector_metadata, vector_id=vector_id
            )

            logger.info(
                "File embedding stored in vector database",
                vector_id=stored_vector_id,
                filename=imgFile.filename,
            )

        except Exception as e:
            raise ValueError(f"Vector database storage failed: {str(e)}") from e

        processing_time = (time.time() - start_time) * 1000

        image_response = ImageResponse(
            vector_id=vector_id,
            artist_id=artist_id,
            entry_id=image_id,  # image_id parameter becomes entry_id in response
            view_id="default",  # File uploads don't have hierarchical structure
            hierarchical_id=None,
            status="success",
            embedding_generated=True,
            processing_time_ms=processing_time,
            message=f"Successfully uploaded and indexed file {imgFile.filename} (embedding: {embedding.shape})",
        )

        return success_response(
            message=f"Successfully uploaded and indexed file {imgFile.filename}",
            image=image_response.dict(),
        )

    except ValueError as validation_error:
        # Handle validation errors
        logger.error(
            "File upload validation error",
            artist_id=artist_id,
            image_id=image_id,
            vector_id=vector_id,
            filename=imgFile.filename,
            error=str(validation_error),
        )

        processing_time = (time.time() - start_time) * 1000

        image_response = ImageResponse(
            vector_id=vector_id,
            artist_id=artist_id,
            entry_id=image_id,
            view_id="default",
            hierarchical_id=None,
            status="error",
            embedding_generated=False,
            processing_time_ms=processing_time,
            message=f"Validation error: {str(validation_error)}",
        )

        return error_response(
            message="File upload validation error",
            details=str(validation_error),
            image=image_response.dict(),
        )

    except Exception as e:
        logger.error(
            "File upload failed",
            artist_id=artist_id,
            image_id=image_id,
            vector_id=vector_id,
            filename=imgFile.filename,
            error=str(e),
            exc_info=True,
        )
        return error_response(
            message="Failed to upload image",
            details=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post("/add_url", summary="Add single image from URL")
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


@router.post("/add_url/batch", summary="Add multiple images from URLs")
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


@router.delete("/{image_id}", summary="Delete image from index")
async def delete_image(entry_id: str):
    """
    Delete an image from the index.

    This will remove both the vector embedding and metadata.
    Can delete by either vector_id (UUID) or image_id.
    """
    from ml.vector_db import qdrant_service

    logger.info("Deleting image", entry_id=entry_id)

    try:
        # Connect to Qdrant
        await qdrant_service.connect()

        # Try to find and delete the image by vector_id first (assuming image_id might be vector_id)
        try:
            # Try direct deletion by vector_id
            deleted = await qdrant_service.client.delete(
                collection_name=qdrant_service.collection_name, points_selector=[entry_id]
            )

            if deleted:
                logger.info(
                    "Image deleted by vector_id", vector_id=entry_id, operation_info=deleted
                )

                return {
                    "status": "success",
                    "message": f"Successfully deleted image with ID: {entry_id}",
                    "deleted_vector_id": entry_id,
                }

        except Exception as vector_delete_error:
            logger.warning(
                "Failed to delete by vector_id, trying by metadata search",
                entry_id=entry_id,
                error=str(vector_delete_error),
            )

        # If direct deletion failed, search by metadata (entry_id)
        try:
            # Search for vectors with matching entry_id in metadata
            search_results = await qdrant_service.client.scroll(
                collection_name=qdrant_service.collection_name,
                scroll_filter={"must": [{"key": "entry_id", "match": {"value": entry_id}}]},
                limit=10,  # Should be only one, but allow for multiple matches
            )

            if not search_results[0]:  # No points found
                logger.warning("Image not found for deletion", entry_id=entry_id)
                return error_response(
                    message="Image not found",
                    details=f"No image found with ID: {entry_id}",
                    status_code=status.HTTP_404_NOT_FOUND,
                )

            # Delete all matching vectors
            vector_ids_to_delete = [point.id for point in search_results[0]]

            deleted = await qdrant_service.client.delete(
                collection_name=qdrant_service.collection_name, points_selector=vector_ids_to_delete
            )

            logger.info(
                "Images deleted by metadata search",
                entry_id=entry_id,
                vector_ids=vector_ids_to_delete,
                operation_info=deleted,
            )

            return success_response(
                message=f"Successfully deleted {len(vector_ids_to_delete)} vector(s) for image: {entry_id}",
                deletion={
                    "deleted_vector_ids": vector_ids_to_delete,
                    "deleted_count": len(vector_ids_to_delete),
                    "entry_id": entry_id,
                },
            )

        except Exception as metadata_error:
            logger.error(
                "Failed to delete by metadata search", entry_id=entry_id, error=str(metadata_error)
            )
            return error_response(
                message="Could not find or delete image",
                details=str(metadata_error),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    except Exception as e:
        logger.error("Failed to delete image", entry_id=entry_id, error=str(e), exc_info=True)
        return error_response(
            message="Failed to delete image",
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

            # Get the indexing threshold to preserve
            indexing_threshold = 20000  # Default
            print(">>>>", orig_opt_config)
            if (
                hasattr(orig_opt_config, "indexing_threshold")
                and orig_opt_config.indexing_threshold is not None
            ):
                indexing_threshold = orig_opt_config.indexing_threshold
                logger.info(f"Will preserve indexing_threshold: {indexing_threshold}")

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
