"""
Modern API v1 - Main router that includes all endpoint modules.
"""

from fastapi import APIRouter

from src.api import status, index, match

# Create the main v1 router
router = APIRouter()

# Include all sub-routers with appropriate tags
router.include_router(
    index.router,
    prefix="/index",
    tags=["Index Management"],
    responses={404: {"description": "Not found"}},
)

router.include_router(
    match.router,
    prefix="/match",
    tags=["Similarity Search"],
    responses={404: {"description": "Not found"}},
)

router.include_router(
    status.router,
    prefix="/status",
    tags=["System Status"],
    responses={404: {"description": "Not found"}},
)
