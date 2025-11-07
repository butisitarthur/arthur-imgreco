"""
API Response utilities for consistent success/error handling.

Matches the frontend TypeScript types:
- ApiSuccess = { success: true; message: string; [key: string]: unknown }
- ApiError = { success: false; message: string; details: string | null; [key: string]: unknown }
"""

from typing import Any, Optional
from fastapi import status
from fastapi.responses import JSONResponse


def success_response(
    message: str, status_code: int = status.HTTP_200_OK, **kwargs: Any
) -> JSONResponse:
    """
    Create a successful API response.

    Args:
        message: Success message for the frontend
        status_code: HTTP status code (default: 200)
        **kwargs: Additional fields to include in response (e.g., data, count, etc.)

    Returns:
        JSONResponse with success=True structure
    """
    response_data = {
        "success": True,
        "message": message,
    }

    # Add any additional fields passed as kwargs
    response_data.update(kwargs)

    return JSONResponse(content=response_data, status_code=status_code)


def error_response(
    message: str,
    details: Optional[str] = None,
    status_code: int = status.HTTP_400_BAD_REQUEST,
    **kwargs: Any,
) -> JSONResponse:
    """
    Create an error API response.

    Args:
        message: Error message for the frontend
        details: Optional error details (can be None)
        status_code: HTTP status code (default: 200 to avoid frontend try/catch)
        **kwargs: Additional fields to include in response

    Returns:
        JSONResponse with success=False structure
    """
    response_data = {
        "success": False,
        "message": message,
        "details": details,
    }

    # Add any additional fields passed as kwargs
    response_data.update(kwargs)
    return JSONResponse(content=response_data, status_code=status_code)
