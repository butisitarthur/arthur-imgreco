"""
Qdrant vector database integration for similarity search.

This module provides high-performance vector operations using Qdrant
with connection pooling, error handling, and batch operations.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from uuid import uuid4

import numpy as np
from core.config import settings
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    OptimizersConfigDiff,
)

from core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ImageMetadata:
    """Metadata for stored image embeddings."""

    artist_id: str
    entry_id: str
    view_id: str
    image_url: Optional[str] = None
    upload_timestamp: Optional[float] = None
    embedding_model: str = f"CLIP-{settings.clip_model_name}"


@dataclass
class SearchResult:
    """Vector similarity search result."""

    artist_id: str
    entry_id: str
    view_id: str
    similarity_score: float
    metadata: Dict[str, Any]
    vector_id: str


class QdrantService:
    """
    High-performance Qdrant vector database service.

    Features:
    - Async connection management
    - Collection auto-creation and management
    - Batch operations for efficiency
    - Error handling and retries
    - Filtering and search capabilities
    """

    def __init__(self):
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_name = "arthur_images"
        self._connected = False

    async def connect(self) -> None:
        """Initialize connection to Qdrant."""
        if self._connected:
            return

        logger.info("Connecting to Qdrant", url=settings.qdrant_url)

        try:
            # Create client with timeout settings
            self.client = AsyncQdrantClient(
                url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30.0
            )

            # Test connection with better error handling
            try:
                collections = await self.client.get_collections()
                logger.info(
                    "Connected to Qdrant successfully",
                    collections_count=len(collections.collections),
                    qdrant_url=settings.qdrant_url,
                )
            except Exception as conn_error:
                logger.error(
                    "Qdrant connection test failed",
                    url=settings.qdrant_url,
                    error=str(conn_error),
                    error_type=type(conn_error).__name__,
                )
                raise

            # Ensure collection exists
            await self._ensure_collection()

            self._connected = True
            logger.info("Qdrant connection established successfully")

        except Exception as e:
            logger.error(
                "Failed to connect to Qdrant",
                url=settings.qdrant_url,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            raise RuntimeError(f"Could not connect to Qdrant: {e}") from e

    async def _ensure_collection(self) -> None:
        """Ensure the images collection exists with proper configuration."""
        try:
            # Check if collection exists
            collection_info = await self.client.get_collection(self.collection_name)
            logger.info("Collection already exists", name=self.collection_name)
            return

        except Exception as e:
            # Collection doesn't exist or other error, try to create it
            logger.info(
                "Collection check failed, attempting to create",
                name=self.collection_name,
                error=str(e),
            )

            try:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.embedding_dimension,
                        distance=Distance.COSINE,  # Use cosine similarity for CLIP embeddings
                    ),
                    # REMOVE FOR PRODUCTION
                    # Start indexing after 1000 vectors instead of 20000
                    optimizers_config=OptimizersConfigDiff(
                        indexing_threshold=settings.indexing_threshold,
                    ),
                )

                logger.info(
                    "Collection created successfully",
                    name=self.collection_name,
                    dimension=settings.embedding_dimension,
                    distance_metric="cosine",
                    indexing_threshold=settings.indexing_threshold,
                )

            except Exception as creation_error:
                logger.error(
                    "Failed to create collection",
                    name=self.collection_name,
                    error=str(creation_error),
                    error_type=type(creation_error).__name__,
                )
                raise

    async def add_image_embedding(
        self, embedding: np.ndarray, metadata: ImageMetadata, vector_id: Optional[str] = None
    ) -> str:
        """
        Add a single image embedding to the collection.

        Args:
            embedding: Normalized embedding vector
            metadata: Image metadata
            vector_id: Optional custom vector ID

        Returns:
            Vector ID of stored embedding
        """
        await self.connect()

        if vector_id is None:
            vector_id = (
                f"{metadata.artist_id}_{metadata.entry_id}_{metadata.view_id}_{uuid4().hex[:8]}"
            )

        point = PointStruct(
            id=vector_id,
            vector=embedding.tolist(),
            payload={
                "artist_id": metadata.artist_id,
                "entry_id": metadata.entry_id,
                "view_id": metadata.view_id,
                "image_url": metadata.image_url,
                "upload_timestamp": metadata.upload_timestamp or time.time(),
                "embedding_model": metadata.embedding_model,
            },
        )

        try:
            await self.client.upsert(collection_name=self.collection_name, points=[point])

            logger.debug(
                "Added image embedding",
                vector_id=vector_id,
                artist_id=metadata.artist_id,
                entry_id=metadata.entry_id,
            )

            return vector_id

        except Exception as e:
            logger.error("Failed to add embedding", vector_id=vector_id, error=str(e))
            raise RuntimeError(f"Could not store embedding: {e}") from e

    async def add_image_embeddings_batch(
        self,
        embeddings: List[np.ndarray],
        metadata_list: List[ImageMetadata],
        vector_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add multiple image embeddings in a single batch operation.

        Args:
            embeddings: List of normalized embedding vectors
            metadata_list: List of image metadata objects
            vector_ids: Optional list of custom vector IDs

        Returns:
            List of vector IDs of stored embeddings
        """
        await self.connect()

        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings must match number of metadata objects")

        if vector_ids is None:
            vector_ids = [
                f"{meta.artist_id}_{meta.entry_id}_{meta.view_id}_{uuid4().hex[:8]}"
                for meta in metadata_list
            ]

        points = []
        for i, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
            point = PointStruct(
                id=vector_ids[i],
                vector=embedding.tolist(),
                payload={
                    "artist_id": metadata.artist_id,
                    "entry_id": metadata.entry_id,
                    "view_id": metadata.view_id,
                    "image_url": metadata.image_url,
                    "upload_timestamp": metadata.upload_timestamp or time.time(),
                    "embedding_model": metadata.embedding_model,
                },
            )
            points.append(point)

        try:
            # Process in batches to avoid memory issues
            batch_size = 100
            stored_ids = []

            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                batch_ids = vector_ids[i : i + batch_size]

                await self.client.upsert(collection_name=self.collection_name, points=batch)

                stored_ids.extend(batch_ids)

                logger.debug(
                    "Stored embedding batch",
                    batch_size=len(batch),
                    batch_start=i,
                    total_batches=(len(points) + batch_size - 1) // batch_size,
                )

            logger.info(
                "Batch embedding storage complete",
                total_stored=len(stored_ids),
                collection=self.collection_name,
            )

            return stored_ids

        except Exception as e:
            logger.error("Failed to store embedding batch", error=str(e))
            raise RuntimeError(f"Could not store embedding batch: {e}") from e

    async def search_similar_images(
        self,
        query_embedding: np.ndarray,
        limit: int = 50,
        score_threshold: Optional[float] = None,
        artist_filter: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search for similar images using vector similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score threshold
            artist_filter: Optional list of artist IDs to filter by

        Returns:
            List of search results sorted by similarity score
        """
        await self.connect()

        # Build filter conditions
        filter_conditions = None
        if artist_filter:
            filter_conditions = Filter(
                must=[FieldCondition(key="artist_id", match=MatchValue(any=artist_filter))]
            )

        try:
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions,
                with_payload=True,
            )

            results = []
            for scored_point in search_results:
                result = SearchResult(
                    artist_id=scored_point.payload["artist_id"],
                    entry_id=scored_point.payload["entry_id"],
                    view_id=scored_point.payload["view_id"],
                    similarity_score=float(scored_point.score),
                    metadata=scored_point.payload,
                    vector_id=str(scored_point.id),
                )
                results.append(result)

            logger.debug(
                "Similarity search complete",
                query_results=len(results),
                max_score=max((r.similarity_score for r in results), default=0),
                min_score=min((r.similarity_score for r in results), default=0),
            )

            return results

        except Exception as e:
            logger.error("Failed to perform similarity search", error=str(e))
            raise RuntimeError(f"Similarity search failed: {e}") from e

    async def get_image_by_id(self, vector_id: str) -> Optional[SearchResult]:
        """
        Retrieve a specific image by its vector ID.

        Args:
            vector_id: Vector ID to retrieve

        Returns:
            SearchResult if found, None otherwise
        """
        await self.connect()

        try:
            points = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[vector_id],
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                return None

            point = points[0]
            return SearchResult(
                artist_id=point.payload["artist_id"],
                entry_id=point.payload["entry_id"],
                view_id=point.payload["view_id"],
                similarity_score=1.0,  # Perfect match for the same ID
                metadata=point.payload,
                vector_id=str(point.id),
            )

        except Exception as e:
            logger.error("Failed to retrieve image by ID", vector_id=vector_id, error=str(e))
            return None

    async def delete_images_by_artist(self, artist_id: str) -> int:
        """
        Delete all images for a specific artist.

        Args:
            artist_id: Artist ID to delete

        Returns:
            Number of images deleted
        """
        await self.connect()

        try:
            # First, count how many points will be deleted
            count_result = await self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="artist_id", match=MatchValue(value=artist_id))]
                ),
            )

            if count_result.count == 0:
                logger.info("No images found for artist", artist_id=artist_id)
                return 0

            # Delete the points
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="artist_id", match=MatchValue(value=artist_id))]
                ),
            )

            deleted_count = count_result.count
            logger.info("Deleted images for artist", artist_id=artist_id, count=deleted_count)

            return deleted_count

        except Exception as e:
            logger.error("Failed to delete images for artist", artist_id=artist_id, error=str(e))
            raise RuntimeError(f"Could not delete images: {e}") from e

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        await self.connect()

        try:
            collection_info = await self.client.get_collection(self.collection_name)

            return {
                "collection_name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "disk_data_size": getattr(collection_info, "disk_data_size", 0),
                "ram_data_size": getattr(collection_info, "ram_data_size", 0),
            }

        except Exception as e:
            logger.error("Failed to get collection stats", error=str(e))
            return {"error": str(e)}

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy and accessible."""
        try:
            if not self.client:
                await self.connect()

            # Simple health check
            collections = await self.client.get_collections()
            return True

        except Exception as e:
            logger.error("Qdrant health check failed", error=str(e))
            return False


# Global Qdrant service instance
qdrant_service = QdrantService()


async def get_qdrant_service() -> QdrantService:
    """Get the global Qdrant service instance."""
    return qdrant_service
