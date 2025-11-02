"""
Image processing pipeline that combines CLIP embeddings with Qdrant storage.

This module provides the main image processing workflow for the Arthur
Image Recognition system.
"""

import time
from typing import List, Optional, Union, Dict, Any
import io
from dataclasses import dataclass

from PIL import Image

from ml.clip_service import get_clip_service
from ml.vector_db import get_qdrant_service, ImageMetadata, SearchResult
from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ImageProcessingResult:
    """Result of image processing and indexing."""

    success: bool
    vector_id: Optional[str] = None
    embedding_shape: Optional[tuple] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class SimilaritySearchRequest:
    """Request for image similarity search."""

    image_input: Union[str, bytes, io.BytesIO, Image.Image]
    max_results: int = 50
    similarity_threshold: Optional[float] = None
    artist_filter: Optional[List[str]] = None


@dataclass
class SimilaritySearchResponse:
    """Response from image similarity search."""

    results: List[SearchResult]
    query_time: float
    total_results: int
    embedding_time: float
    search_time: float


class ImageProcessingPipeline:
    """
    Main image processing pipeline for Arthur Image Recognition.

    This class orchestrates the entire workflow:
    1. Image loading and preprocessing
    2. CLIP embedding generation
    3. Vector database storage and search
    4. Result formatting and error handling
    """

    def __init__(self):
        self.clip_service = None
        self.qdrant_service = None

    async def _get_services(self):
        """Lazy loading of services."""
        if self.clip_service is None:
            self.clip_service = await get_clip_service()
        if self.qdrant_service is None:
            self.qdrant_service = await get_qdrant_service()

    async def process_and_store_image(
        self,
        image_input: Union[str, bytes, io.BytesIO, Image.Image],
        artist_id: str,
        entry_id: str,
        view_id: str,
        image_url: Optional[str] = None,
    ) -> ImageProcessingResult:
        """
        Process an image and store its embedding in the vector database.

        Args:
            image_input: Image data (URL, bytes, BytesIO, or PIL Image)
            artist_id: Artist identifier
            entry_id: Entry identifier
            view_id: View identifier
            image_url: Optional source URL for metadata

        Returns:
            ImageProcessingResult with success status and details
        """
        start_time = time.time()

        try:
            await self._get_services()

            logger.info(
                "Processing image for storage",
                artist_id=artist_id,
                entry_id=entry_id,
                view_id=view_id,
            )

            # Generate embedding using CLIP
            embedding_start = time.time()
            embedding = await self.clip_service.generate_embedding(image_input)
            embedding_time = time.time() - embedding_start

            # Create metadata
            metadata = ImageMetadata(
                artist_id=artist_id,
                entry_id=entry_id,
                view_id=view_id,
                image_url=image_url,
                upload_timestamp=time.time(),
            )

            # Store in vector database
            storage_start = time.time()
            vector_id = await self.qdrant_service.add_image_embedding(embedding, metadata)
            storage_time = time.time() - storage_start

            total_time = time.time() - start_time

            logger.info(
                "Image processing complete",
                vector_id=vector_id,
                embedding_time=f"{embedding_time:.3f}s",
                storage_time=f"{storage_time:.3f}s",
                total_time=f"{total_time:.3f}s",
                embedding_shape=embedding.shape,
            )

            return ImageProcessingResult(
                success=True,
                vector_id=vector_id,
                embedding_shape=embedding.shape,
                processing_time=total_time,
            )

        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Image processing failed: {str(e)}"

            logger.error(
                "Image processing failed",
                artist_id=artist_id,
                entry_id=entry_id,
                view_id=view_id,
                error=str(e),
                processing_time=f"{total_time:.3f}s",
            )

            return ImageProcessingResult(
                success=False, processing_time=total_time, error_message=error_msg
            )

    async def process_and_store_images_batch(
        self, image_data: List[Dict[str, Any]]
    ) -> List[ImageProcessingResult]:
        """
        Process multiple images in batch for efficiency.

        Args:
            image_data: List of dicts with keys: image_input, artist_id, entry_id, view_id, image_url

        Returns:
            List of ImageProcessingResult objects
        """
        if not image_data:
            return []

        start_time = time.time()
        await self._get_services()

        logger.info("Starting batch image processing", batch_size=len(image_data))

        try:
            # Extract image inputs for batch embedding generation
            image_inputs = [item["image_input"] for item in image_data]

            # Generate embeddings in batch
            embedding_start = time.time()
            embeddings = await self.clip_service.generate_embeddings_batch(image_inputs)
            embedding_time = time.time() - embedding_start

            # Prepare metadata and valid embeddings
            valid_data = []
            valid_embeddings = []
            results = []

            for i, (embedding, item) in enumerate(zip(embeddings, image_data)):
                if embedding is not None:
                    metadata = ImageMetadata(
                        artist_id=item["artist_id"],
                        entry_id=item["entry_id"],
                        view_id=item["view_id"],
                        image_url=item.get("image_url"),
                        upload_timestamp=time.time(),
                    )
                    valid_data.append((i, metadata))
                    valid_embeddings.append(embedding)
                else:
                    # Failed embedding
                    results.append(
                        ImageProcessingResult(
                            success=False,
                            error_message=f"Failed to generate embedding for item {i}",
                        )
                    )

            # Store valid embeddings in batch
            if valid_embeddings:
                storage_start = time.time()
                metadata_list = [item[1] for item in valid_data]
                vector_ids = await self.qdrant_service.add_image_embeddings_batch(
                    valid_embeddings, metadata_list
                )
                storage_time = time.time() - storage_start

                # Create success results
                for (original_idx, metadata), vector_id, embedding in zip(
                    valid_data, vector_ids, valid_embeddings
                ):
                    results.insert(
                        original_idx,
                        ImageProcessingResult(
                            success=True,
                            vector_id=vector_id,
                            embedding_shape=embedding.shape,
                            processing_time=(time.time() - start_time)
                            / len(image_data),  # Approximate per-image time
                        ),
                    )
            else:
                storage_time = 0

            total_time = time.time() - start_time
            successful_count = sum(1 for r in results if r.success)

            logger.info(
                "Batch image processing complete",
                total_images=len(image_data),
                successful=successful_count,
                failed=len(image_data) - successful_count,
                embedding_time=f"{embedding_time:.3f}s",
                storage_time=f"{storage_time:.3f}s",
                total_time=f"{total_time:.3f}s",
            )

            return results

        except Exception as e:
            total_time = time.time() - start_time
            error_msg = f"Batch processing failed: {str(e)}"

            logger.error(
                "Batch image processing failed",
                batch_size=len(image_data),
                error=str(e),
                processing_time=f"{total_time:.3f}s",
            )

            # Return failure results for all items
            return [
                ImageProcessingResult(
                    success=False,
                    processing_time=total_time / len(image_data),
                    error_message=error_msg,
                )
                for _ in image_data
            ]

    async def search_similar_images(
        self, request: SimilaritySearchRequest
    ) -> SimilaritySearchResponse:
        """
        Search for similar images using the processing pipeline.

        Args:
            request: Similarity search request

        Returns:
            SimilaritySearchResponse with results and timing
        """
        start_time = time.time()
        await self._get_services()

        logger.info(
            "Starting similarity search",
            max_results=request.max_results,
            similarity_threshold=request.similarity_threshold,
            artist_filter=request.artist_filter,
        )

        try:
            # Generate embedding for query image
            embedding_start = time.time()
            query_embedding = await self.clip_service.generate_embedding(request.image_input)
            embedding_time = time.time() - embedding_start

            # Search in vector database
            search_start = time.time()
            results = await self.qdrant_service.search_similar_images(
                query_embedding=query_embedding,
                limit=request.max_results,
                score_threshold=request.similarity_threshold,
                artist_filter=request.artist_filter,
            )
            search_time = time.time() - search_start

            total_time = time.time() - start_time

            logger.info(
                "Similarity search complete",
                results_found=len(results),
                embedding_time=f"{embedding_time:.3f}s",
                search_time=f"{search_time:.3f}s",
                total_time=f"{total_time:.3f}s",
                top_score=results[0].similarity_score if results else 0,
            )

            return SimilaritySearchResponse(
                results=results,
                query_time=total_time,
                total_results=len(results),
                embedding_time=embedding_time,
                search_time=search_time,
            )

        except Exception as e:
            total_time = time.time() - start_time

            logger.error(
                "Similarity search failed", error=str(e), processing_time=f"{total_time:.3f}s"
            )

            return SimilaritySearchResponse(
                results=[], query_time=total_time, total_results=0, embedding_time=0, search_time=0
            )

    async def find_similar_by_id(
        self,
        vector_id: str,
        max_results: int = 50,
        similarity_threshold: Optional[float] = None,
        artist_filter: Optional[List[str]] = None,
    ) -> SimilaritySearchResponse:
        """
        Find images similar to an existing image by its vector ID.

        Args:
            vector_id: ID of the existing image
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity threshold
            artist_filter: Optional artist filter

        Returns:
            SimilaritySearchResponse with similar images
        """
        start_time = time.time()
        await self._get_services()

        logger.info("Finding similar images by ID", vector_id=vector_id)

        try:
            # Get the original image metadata and embedding
            # Note: In a real implementation, you might want to store embeddings
            # separately or retrieve them from Qdrant if the API supports it

            # For now, we'll return an empty response with a note about needing
            # the embedding retrieval capability
            logger.warning(
                "Similar-by-ID search requires embedding retrieval",
                vector_id=vector_id,
                note="Feature requires storing embeddings separately or Qdrant embedding retrieval API",
            )

            total_time = time.time() - start_time

            return SimilaritySearchResponse(
                results=[], query_time=total_time, total_results=0, embedding_time=0, search_time=0
            )

        except Exception as e:
            total_time = time.time() - start_time

            logger.error(
                "Similar-by-ID search failed",
                vector_id=vector_id,
                error=str(e),
                processing_time=f"{total_time:.3f}s",
            )

            return SimilaritySearchResponse(
                results=[], query_time=total_time, total_results=0, embedding_time=0, search_time=0
            )

    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics from all pipeline components."""
        await self._get_services()

        try:
            clip_stats = self.clip_service.get_stats()
            qdrant_stats = await self.qdrant_service.get_collection_stats()

            return {
                "clip_service": clip_stats,
                "vector_database": qdrant_stats,
                "pipeline_status": "operational",
            }

        except Exception as e:
            logger.error("Failed to get pipeline stats", error=str(e))
            return {"error": str(e), "pipeline_status": "error"}


# Global pipeline instance
image_pipeline = ImageProcessingPipeline()


async def get_image_pipeline() -> ImageProcessingPipeline:
    """Get the global image processing pipeline instance."""
    return image_pipeline
