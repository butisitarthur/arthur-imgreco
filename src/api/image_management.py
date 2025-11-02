"""
Image Management API endpoints - Adding, updating, deleting images from the index.
"""

import time
from typing import Optional, List

from fastapi import APIRouter, HTTPException, status, File, Form, UploadFile

from core.logging import get_logger
from api.models import (
    AddImageRequest, BatchImageRequest, ImageResponse, BatchImageResponse,
    ImageMetadata
)

logger = get_logger(__name__)
router = APIRouter()


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
        url=str(request.image_url)
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
                        file_size=len(image_bytes)
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
                embedding_norm=float(embedding.dot(embedding) ** 0.5)
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
                embedding_model="CLIP-ViT-B/32"
            )
            
            # Use the final vector_id (either provided or auto-generated)
            vector_id = await qdrant_service.add_image_embedding(
                embedding=embedding,
                metadata=vector_metadata,
                vector_id=final_vector_id
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
            vector_id=final_vector_id,
            artist_id=artist_id,
            entry_id=entry_id,
            view_id=view_id,
            hierarchical_id=request.get_hierarchical_id(),
            status="success",
            embedding_generated=True,
            processing_time_ms=processing_time,
            message=f"Successfully indexed image {entry_id or final_vector_id} (embedding: {embedding.shape}, vector_id: {final_vector_id})"
        )
        
    except ValueError as validation_error:
        # Handle validation errors
        logger.error(
            "Image processing validation error",
            artist_id=artist_id,
            entry_id=entry_id,
            vector_id=final_vector_id,
            error=str(validation_error)
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return ImageResponse(
            vector_id=final_vector_id,
            artist_id=artist_id,
            entry_id=entry_id,
            view_id=view_id,
            hierarchical_id=request.get_hierarchical_id(),
            status="error",
            embedding_generated=False,
            processing_time_ms=processing_time,
            message=f"Validation error: {str(validation_error)}"
        )
        
    except Exception as general_error:
        logger.error(
            "Failed to add image",
            artist_id=artist_id,
            entry_id=entry_id,
            vector_id=final_vector_id,
            error=str(general_error),
            exc_info=True
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add image: {str(general_error)}"
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
                    vector_id=image_request.get_vector_id(),
                    artist_id=image_request.artist_id,
                    entry_id=image_request.entry_id,
                    status=image_result.status
                )
                
            except Exception as e:
                # Handle individual image failures
                failed += 1
                error_result = ImageResponse(
                    vector_id=image_request.get_vector_id(),
                    artist_id=image_request.artist_id,
                    entry_id=image_request.entry_id,
                    view_id=image_request.view_id,
                    hierarchical_id=image_request.get_hierarchical_id(),
                    status="error",
                    embedding_generated=False,
                    processing_time_ms=0.0,
                    message=f"Processing failed: {str(e)}"
                )
                results.append(error_result)
                
                logger.error(
                    "Batch image processing failed",
                    vector_id=image_request.get_vector_id(),
                    entry_id=image_request.entry_id,
                    artist_id=image_request.artist_id,
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
async def upload_image_file(
    vector_id: str = Form(..., description="UUID for this vector in the database"),
    artist_id: str = Form(..., description="Unique artist identifier"),
    image_id: str = Form(..., description="Unique image identifier"),
    imgFile: UploadFile = File(..., description="Image file to upload"),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
) -> ImageResponse:
    """
    Upload an image file directly instead of providing a URL.
    
    Useful for images that are not publicly accessible via URL.
    """
    import io
    from PIL import Image
    
    from ml.clip_service import clip_service
    from ml.vector_db import ImageMetadata as VectorImageMetadata, qdrant_service
    
    start_time = time.time()
    
    logger.info("Uploading image file", 
               artist_id=artist_id, 
               image_id=image_id, 
               vector_id=vector_id,
               filename=imgFile.filename,
               content_type=imgFile.content_type)
    
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
                filename=imgFile.filename
            )
            
            logger.info("Image file validated", 
                       image_info=image_info,
                       vector_id=vector_id)
            
        except Exception as e:
            raise ValueError(f"Invalid image format: {str(e)}") from e
        
        # Step 3: Generate CLIP embedding from image data
        try:
            # Create a temporary BytesIO for CLIP processing
            image_buffer = io.BytesIO(content)
            embedding = await clip_service.generate_embedding(image_buffer)
            
            logger.info("CLIP embedding generated from file",
                       embedding_shape=embedding.shape,
                       embedding_norm=float(embedding.dot(embedding) ** 0.5),
                       vector_id=vector_id)
            
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
                embedding_model="CLIP-ViT-B/32"
            )
            
            # Use the provided UUID for the vector ID
            stored_vector_id = await qdrant_service.add_image_embedding(
                embedding=embedding,
                metadata=vector_metadata,
                vector_id=vector_id
            )
            
            logger.info("File embedding stored in vector database",
                       vector_id=stored_vector_id,
                       filename=imgFile.filename)
            
        except Exception as e:
            raise ValueError(f"Vector database storage failed: {str(e)}") from e
        
        processing_time = (time.time() - start_time) * 1000
        
        return ImageResponse(
            vector_id=vector_id,
            artist_id=artist_id,
            entry_id=image_id,  # image_id parameter becomes entry_id in response
            view_id=None,  # File uploads don't have hierarchical structure
            hierarchical_id=None,
            status="success",
            embedding_generated=True,
            processing_time_ms=processing_time,
            message=f"Successfully uploaded and indexed file {imgFile.filename} (embedding: {embedding.shape}, vector_id: {vector_id})"
        )
        
    except ValueError as validation_error:
        # Handle validation errors
        logger.error("File upload validation error",
                    artist_id=artist_id,
                    image_id=image_id,
                    vector_id=vector_id,
                    filename=imgFile.filename,
                    error=str(validation_error))
        
        processing_time = (time.time() - start_time) * 1000
        
        return ImageResponse(
            vector_id=vector_id,
            artist_id=artist_id,
            entry_id=image_id,
            view_id=None,
            hierarchical_id=None,
            status="error",
            embedding_generated=False,
            processing_time_ms=processing_time,
            message=f"Validation error: {str(validation_error)}"
        )
        
    except Exception as e:
        logger.error("File upload failed", 
                    artist_id=artist_id, 
                    image_id=image_id, 
                    vector_id=vector_id,
                    filename=imgFile.filename,
                    error=str(e),
                    exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload image: {str(e)}"
        )


@router.delete("/images/{image_id}")
async def delete_image(image_id: str):
    """
    Delete an image from the index.
    
    This will remove both the vector embedding and metadata.
    Can delete by either vector_id (UUID) or image_id.
    """
    from ml.vector_db import qdrant_service
    
    logger.info("Deleting image", image_id=image_id)
    
    try:
        # Connect to Qdrant
        await qdrant_service.connect()
        
        # Try to find and delete the image by vector_id first (assuming image_id might be vector_id)
        try:
            # Try direct deletion by vector_id
            deleted = await qdrant_service.client.delete(
                collection_name=qdrant_service.collection_name,
                points_selector=[image_id]
            )
            
            if deleted:
                logger.info("Image deleted by vector_id", 
                           vector_id=image_id, 
                           operation_info=deleted)
                
                return {
                    "status": "success",
                    "message": f"Successfully deleted image with ID: {image_id}",
                    "deleted_vector_id": image_id
                }
        
        except Exception as vector_delete_error:
            logger.warning("Failed to delete by vector_id, trying by metadata search", 
                          image_id=image_id, 
                          error=str(vector_delete_error))
        
        # If direct deletion failed, search by metadata (entry_id)
        try:
            # Search for vectors with matching entry_id in metadata
            search_results = await qdrant_service.client.scroll(
                collection_name=qdrant_service.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "entry_id",
                            "match": {"value": image_id}
                        }
                    ]
                },
                limit=10  # Should be only one, but allow for multiple matches
            )
            
            if not search_results[0]:  # No points found
                logger.warning("Image not found for deletion", image_id=image_id)
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Image not found: {image_id}"
                )
            
            # Delete all matching vectors
            vector_ids_to_delete = [point.id for point in search_results[0]]
            
            deleted = await qdrant_service.client.delete(
                collection_name=qdrant_service.collection_name,
                points_selector=vector_ids_to_delete
            )
            
            logger.info("Images deleted by metadata search", 
                       image_id=image_id,
                       vector_ids=vector_ids_to_delete,
                       operation_info=deleted)
            
            return {
                "status": "success", 
                "message": f"Successfully deleted {len(vector_ids_to_delete)} vector(s) for image: {image_id}",
                "deleted_vector_ids": vector_ids_to_delete,
                "deleted_count": len(vector_ids_to_delete)
            }
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
            
        except Exception as metadata_error:
            logger.error("Failed to delete by metadata search", 
                        image_id=image_id, 
                        error=str(metadata_error))
            raise ValueError(f"Could not find or delete image: {str(metadata_error)}") from metadata_error
        
    except HTTPException:
        # Re-raise HTTP exceptions without wrapping
        raise
        
    except Exception as e:
        logger.error("Failed to delete image", image_id=image_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete image: {str(e)}"
        ) from e


@router.get("/images/{image_id}")
async def get_image_info(image_id: str):
    """
    Get information about a specific image in the index.
    
    Returns metadata, embedding info, and similarity statistics.
    Can look up by either vector_id (UUID) or image_id.
    """
    from ml.vector_db import qdrant_service
    
    logger.info("Getting image info", image_id=image_id)
    
    try:
        # Connect to Qdrant
        await qdrant_service.connect()
        
        # Try to find the image by vector_id first
        try:
            # Try direct lookup by vector_id
            points = await qdrant_service.client.retrieve(
                collection_name=qdrant_service.collection_name,
                ids=[image_id],
                with_payload=True,
                with_vectors=False  # Don't return the actual embedding vector
            )
            
            if points:
                point = points[0]
                
                # Extract metadata
                metadata = point.payload or {}
                
                logger.info("Image found by vector_id", 
                           vector_id=image_id,
                           metadata=metadata)
                
                return {
                    "status": "found",
                    "lookup_method": "vector_id",
                    "vector_id": str(point.id),
                    "image_id": metadata.get("entry_id"),
                    "artist_id": metadata.get("artist_id"),
                    "view_id": metadata.get("view_id"),
                    "image_url": metadata.get("image_url"),
                    "upload_timestamp": metadata.get("upload_timestamp"),
                    "embedding_model": metadata.get("embedding_model"),
                    "metadata": metadata
                }
        
        except Exception as vector_lookup_error:
            logger.debug("Failed to find by vector_id, trying metadata search", 
                        image_id=image_id, 
                        error=str(vector_lookup_error))
        
        # If direct lookup failed, search by metadata (entry_id)
        try:
            # Search for vectors with matching entry_id in metadata
            search_results = await qdrant_service.client.scroll(
                collection_name=qdrant_service.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "entry_id", 
                            "match": {"value": image_id}
                        }
                    ]
                },
                limit=10,  # Allow for multiple matches
                with_payload=True,
                with_vectors=False
            )
            
            if not search_results[0]:  # No points found
                logger.warning("Image not found", image_id=image_id)
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Image not found: {image_id}"
                )
            
            # Return information about all matching vectors
            results = []
            for point in search_results[0]:
                metadata = point.payload or {}
                results.append({
                    "vector_id": str(point.id),
                    "image_id": metadata.get("entry_id"),
                    "artist_id": metadata.get("artist_id"),
                    "view_id": metadata.get("view_id"), 
                    "image_url": metadata.get("image_url"),
                    "upload_timestamp": metadata.get("upload_timestamp"),
                    "embedding_model": metadata.get("embedding_model"),
                    "metadata": metadata
                })
            
            logger.info("Images found by metadata search", 
                       image_id=image_id,
                       count=len(results))
            
            if len(results) == 1:
                # Single result
                result = results[0]
                result.update({
                    "status": "found",
                    "lookup_method": "metadata_search"
                })
                return result
            else:
                # Multiple results
                return {
                    "status": "found",
                    "lookup_method": "metadata_search", 
                    "count": len(results),
                    "message": f"Found {len(results)} vectors for image_id: {image_id}",
                    "images": results
                }
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
            
        except Exception as metadata_error:
            logger.error("Failed to search by metadata", 
                        image_id=image_id, 
                        error=str(metadata_error))
            raise ValueError(f"Could not find image: {str(metadata_error)}") from metadata_error
        
    except HTTPException:
        # Re-raise HTTP exceptions without wrapping
        raise
        
    except Exception as e:
        logger.error("Failed to get image info", image_id=image_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get image info: {str(e)}"
        ) from e