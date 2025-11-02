"""
Modern API v1 - Main router that includes all endpoint modules.
"""

from fastapi import APIRouter

from api import image_management, similarity_search, system_management

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