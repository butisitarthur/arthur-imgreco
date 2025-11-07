"""
Standardized error handling utilities for Arthur Image Recognition 2.0

This module provides consistent error handling patterns, exception types,
and response formatting across all API endpoints.
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException, status

from core.logging import get_logger

logger = get_logger(__name__)


class ArthurError(Exception):
    """Base exception class for Arthur-specific errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ServiceError(ArthurError):
    """Error in external service (ML, database, etc.)."""

    pass


class ValidationError(ArthurError):
    """Input validation error."""

    pass


class NotFoundError(ArthurError):
    """Resource not found error."""

    pass


def handle_service_error(
    error: Exception,
    operation: str,
    logger_instance=None,
    default_message: str = "Service operation failed",
) -> HTTPException:
    """
    Standardized service error handler.

    Args:
        error: The caught exception
        operation: Description of the operation that failed
        logger_instance: Logger to use (defaults to module logger)
        default_message: Default error message for user

    Returns:
        HTTPException with appropriate status code and message
    """
    log = logger_instance or logger

    # Log the detailed error
    log.error(f"Service error during {operation}", error=str(error), exc_info=True)

    # Determine appropriate HTTP status and user message
    if isinstance(error, (ConnectionError, TimeoutError)):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        user_message = "Service temporarily unavailable. Please try again."
    elif isinstance(error, ValidationError):
        status_code = status.HTTP_400_BAD_REQUEST
        user_message = error.message
    elif isinstance(error, NotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
        user_message = error.message
    elif isinstance(error, ServiceError):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        user_message = error.message
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        user_message = default_message

    return HTTPException(
        status_code=status_code,
        detail={
            "error": user_message,
            "operation": operation,
            "error_code": getattr(error, "error_code", "INTERNAL_ERROR"),
        },
    )


def log_and_raise_http_error(
    status_code: int,
    message: str,
    operation: str = "unknown",
    details: Optional[Dict[str, Any]] = None,
    logger_instance=None,
) -> None:
    """
    Log an error and raise HTTPException.

    Args:
        status_code: HTTP status code
        message: Error message
        operation: Operation context
        details: Additional error details
        logger_instance: Logger to use
    """
    log = logger_instance or logger

    log.error(f"HTTP error {status_code} during {operation}", message=message, details=details)

    raise HTTPException(
        status_code=status_code,
        detail={"error": message, "operation": operation, **(details or {})},
    )


def safe_execute(operation_name: str, logger_instance=None):
    """
    Decorator for safe execution with standardized error handling.

    Usage:
        @safe_execute("image processing")
        async def process_image(image_url: str):
            # Your code here
            pass
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                raise handle_service_error(e, operation_name, logger_instance)

        return wrapper

    return decorator
