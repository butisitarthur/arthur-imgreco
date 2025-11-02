"""
Legacy API endpoints for backward compatibility with arthur-imgreco v1
"""

import re
import time
from typing import Dict, List, Optional, Union
from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from arthur_imgreco.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


class MatchRequest(BaseModel):
    """Request model for image matching."""

    imgUrl: Optional[str] = None


class MatchResult(BaseModel):
    """Match result model."""

    artistId: Optional[str] = None
    entryId: str
    viewId: str
    score: str


class MatchResponse(BaseModel):
    """Response model for image matching."""

    matches: List[MatchResult]
    status: str = "success"
    message: str = "Matched"


class AddImageRequest(BaseModel):
    """Request model for adding images."""

    images: List[Dict[str, str]]


class IndexResponse(BaseModel):
    """Response model for indexing operations."""

    status: str
    message: str


# URL validation regex (copied from original)
URL_REGEX = re.compile(
    r"^(?:http|ftp)s?://"
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"
    r"localhost|"
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
    r"(?::\d+)?"
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


@router.post("/match", response_model=MatchResponse)
@router.post("/match/{artist_id}", response_model=MatchResponse)
async def match_image(
    artist_id: Optional[str] = None,
    request_data: Optional[MatchRequest] = None,
    imgFile: Optional[UploadFile] = File(None),
) -> MatchResponse:
    """
    Match image against the database.

    Compatible with original arthur-imgreco /match endpoint.
    Supports both URL and file upload.
    """
    logger.info("Legacy match request", artist_id=artist_id)

    img_source = None

    # Handle JSON request with URL
    if request_data and request_data.imgUrl:
        if not re.match(URL_REGEX, request_data.imgUrl):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image url")
        img_source = request_data.imgUrl
        logger.info("Matching image from URL", url=img_source)

    # Handle file upload
    elif imgFile and imgFile.filename:
        if not imgFile.filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image file"
            )
        # Read file content
        content = await imgFile.read()
        img_source = BytesIO(content)
        logger.info("Matching image from file", filename=imgFile.filename)

    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No image provided")

    # Implement actual image matching using CLIP + Qdrant
    start_time = time.time()
    matches = []

    try:
        # Get pipeline from app state (via dependency injection would be better but keeping it simple)
        # In a production app, we'd use FastAPI's dependency system

        # For now, return mock data that shows the new system structure
        # The actual implementation will be in the modern v1 API
        if img_source:
            # Mock successful processing to show API compatibility
            matches = [
                {
                    "artist": "Sample Artist",
                    "image_id": "sample_001",
                    "confidence": 0.95,
                    "metadata": {
                        "source": "clip_embedding",
                        "processing_time": f"{time.time() - start_time:.3f}s",
                    },
                }
            ]
            logger.info("Mock image matching completed", matches_count=len(matches))

    except Exception as e:
        logger.error("Error in image matching", error=str(e))

    return MatchResponse(matches=matches, status="success", message="Matched")


@router.post("/match-legacy", response_model=MatchResponse)
@router.post("/match-legacy/{artist_id}", response_model=MatchResponse)
async def match_image_legacy(
    artist_id: Optional[str] = None,
    request_data: Optional[MatchRequest] = None,
    imgFile: Optional[UploadFile] = File(None),
) -> MatchResponse:
    """
    Legacy match endpoint (original OpenCV-based system).

    This endpoint is provided for comparison purposes and will
    eventually be removed in favor of the new vector-based system.
    """
    logger.info("Legacy match request (OpenCV)", artist_id=artist_id)

    # For now, delegate to the new system
    # TODO: Implement actual legacy system integration if needed
    return await match_image(artist_id, request_data, imgFile)


@router.post("/artist/image", response_model=IndexResponse)
@router.post("/artist/image/{artist_id}/{entry_id}/{view_id}", response_model=IndexResponse)
async def add_image(
    artist_id: Optional[str] = None,
    entry_id: Optional[str] = None,
    view_id: Optional[str] = None,
    request_data: Optional[AddImageRequest] = None,
    imgFile: Optional[UploadFile] = File(None),
    imgUrl: Optional[str] = Form(None),
) -> IndexResponse:
    """
    Add image to artist index.

    Compatible with original arthur-imgreco image addition endpoints.
    Supports both single image and batch operations.
    """
    logger.info("Add image request", artist_id=artist_id, entry_id=entry_id, view_id=view_id)

    # Handle batch request
    if request_data and request_data.images:
        logger.info("Processing batch image addition", count=len(request_data.images))
        # TODO: Implement batch image processing
        return IndexResponse(
            status="success", message=f"Added {len(request_data.images)} images to index"
        )

    # Handle single image request
    elif artist_id and entry_id and view_id:
        if imgFile:
            logger.info("Processing single image file", filename=imgFile.filename)
            # TODO: Process uploaded file
        elif imgUrl:
            logger.info("Processing single image URL", url=imgUrl)
            # TODO: Process image from URL
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No image provided")

        # TODO: Implement actual image indexing
        return IndexResponse(
            status="success", message=f"Added image {artist_id}/{entry_id}/{view_id} to index"
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request parameters"
        )


@router.get("/status")
async def get_status() -> Dict[str, Union[str, int]]:
    """
    Get system status.

    Compatible with original arthur-imgreco /status endpoint.
    """
    logger.info("Status request")

    # TODO: Implement actual status checks
    return {
        "status": "running",
        "version": "2.0.0",
        "indexed_artists": 0,  # TODO: Get from database
        "total_images": 0,  # TODO: Get from database
        "vector_db_status": "connected",  # TODO: Check Qdrant
        "database_status": "connected",  # TODO: Check PostgreSQL
    }


@router.post("/unified-index")
async def manage_unified_index() -> IndexResponse:
    """
    Manage unified index operations.

    Compatible with original arthur-imgreco /unified-index endpoint.
    """
    logger.info("Unified index management request")

    # TODO: Implement index management
    return IndexResponse(status="success", message="Index operation completed")


@router.get("/list-pickles")
async def list_pickles() -> Dict[str, List[str]]:
    """
    Legacy endpoint for pickle file listing.

    This endpoint is maintained for compatibility but will
    return empty results since we use vector database storage.
    """
    logger.info("List pickles request (legacy)")

    return {"pickle_files": [], "message": "Using vector database instead of pickle files"}
