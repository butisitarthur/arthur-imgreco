"""
Shared data models for API v1 endpoints.
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, HttpUrl

# =====================================
# SIMILARITY SEARCH MODELS
# =====================================


class ImageSimilarityResult(BaseModel):
    """A single similarity search result."""

    artist_id: str
    entry_id: str
    view_id: str = "default"
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    confidence: str = Field(..., description="High, Medium, or Low")
    embedding_distance: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ImageSimilarityRequest(BaseModel):
    """Request for image similarity search."""

    image_url: Optional[HttpUrl] = None
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    max_results: int = Field(50, ge=1, le=1000)
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    artist_filter: Optional[List[str]] = None


class ImageSimilarityResponse(BaseModel):
    """Response containing similarity search results."""

    results: List[ImageSimilarityResult]
    total_results: int
    query_time_ms: float
    model_info: Dict[str, Any]


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


# =====================================
# STATUS MODELS
# =====================================


class SystemStatus(BaseModel):
    """Detailed health check response model."""

    status: str
    message: str
    timestamp: float  # Unix timestamp
    version: str
    indexing_threshold: Optional[int] = None  # Qdrant indexing threshold
    services: Dict[str, str]


class IndexStatus(BaseModel):
    """Statistics about the image index."""

    total_artists: int
    total_entries: int
    total_images: int
    index_size_mb: float
    last_updated: float  # Unix timestamp
    vector_dimension: int
    similarity_model: str

    # Indexing status fields
    indexed_vectors: Optional[int] = None  # Number of vectors in search index
    stored_points: Optional[int] = None  # Number of stored points/records
    indexing_status: Optional[str] = None  # "ready", "pending", "building", etc.

    cached_time: Optional[float] = None  # Unix timestamp when this result was cached
    cached: bool


class ArtistStatus(BaseModel):
    """Analytics data for a specific artist."""

    artist_id: str
    total_images: int
    latest_upload: Optional[float]  # Unix timestamp
    avg_similarity_score: float
    image_distribution: dict
    vector_size_bytes: int


class EntryStatus(BaseModel):
    """Status and analytics for a specific entry."""

    entry_id: str
    latest_upload: Optional[float] = None  # Unix timestamp
    view_count: int = 0
    total_images: int = 0


class ImageStatus(BaseModel):
    """Status and metadata for a specific image."""

    vector_id: str
    entry_id: Optional[str] = None
    artist_id: Optional[str] = None
    view_id: Optional[str] = None
    image_url: Optional[str] = None
    upload_timestamp: Optional[float] = None
    embedding_model: Optional[str] = None
    lookup_method: str = "unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =====================================
# INDEX MODELS
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
    """Request to add a single image with flexible ID structure."""

    # Option 1: Direct vector_id (takes precedence if provided)
    vector_id: Optional[str] = Field(
        None, description="Direct vector ID (overrides hierarchical if provided)"
    )

    # Option 2: Hierarchical structure (auto-generates vector_id if vector_id not provided)
    artist_id: Optional[str] = Field(None, description="Unique artist identifier")
    entry_id: Optional[str] = Field(None, description="Unique artwork/entry identifier")
    view_id: Optional[str] = Field(None, description="View identifier (main, detail, side, etc.)")

    # Required fields
    image_url: HttpUrl = Field(..., description="URL of the image to index")
    metadata: Optional[ImageMetadata] = None

    def model_post_init(self, __context):
        """Post-initialization to generate vector_id if needed."""
        from core.hierarchical_ids import resolve_vector_id

        try:
            final_vector_id, _ = resolve_vector_id(
                vector_id=self.vector_id,
                artist_id=self.artist_id,
                entry_id=self.entry_id,
                view_id=self.view_id,
            )
            self.vector_id = final_vector_id
        except ValueError as e:
            raise ValueError(str(e)) from e

    def get_vector_id(self) -> str:
        """Get the final vector_id (either provided or auto-generated)."""
        return self.vector_id

    def get_hierarchical_components(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Get the hierarchical components if available."""
        return self.artist_id, self.entry_id, self.view_id

    def get_hierarchical_id(self) -> Optional[str]:
        """Get the hierarchical identifier if all components are available."""
        from core.hierarchical_ids import create_hierarchical_id, validate_hierarchical_components

        if validate_hierarchical_components(self.artist_id, self.entry_id, self.view_id):
            return create_hierarchical_id(self.artist_id, self.entry_id, self.view_id)
        return None


class BatchImageRequest(BaseModel):
    """Request to add multiple images in batch."""

    images: List[AddImageRequest] = Field(..., description="List of images to add")


class ImageResponse(BaseModel):
    """Response after adding an image."""

    vector_id: str
    artist_id: Optional[str] = None
    entry_id: Optional[str] = None
    view_id: Optional[str] = None
    hierarchical_id: Optional[str] = None
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


# =====================================
# HEALTH MODELS
# =====================================


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: float  # Unix timestamp
    version: str
    message: str
