"""
Structured logging configuration using structlog.
"""

import sys
import logging as stdlib_logging
from typing import Any

import structlog
from structlog.stdlib import LoggerFactory

from core.config import settings


def configure_logging() -> None:
    """Configure structured logging for the application."""

    # Configure stdlib logging
    stdlib_logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(stdlib_logging, settings.log_level.upper()),
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            (
                structlog.dev.ConsoleRenderer()
                if settings.debug
                else structlog.processors.JSONRenderer()
            ),
        ],
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


# Application-specific log helpers
def log_api_request(
    logger: structlog.stdlib.BoundLogger, method: str, path: str, **kwargs: Any
) -> None:
    """Log API request information."""
    logger.info("API request", method=method, path=path, **kwargs)


def log_api_response(
    logger: structlog.stdlib.BoundLogger, status_code: int, duration: float, **kwargs: Any
) -> None:
    """Log API response information."""
    logger.info("API response", status_code=status_code, duration_ms=duration * 1000, **kwargs)


def log_ml_operation(logger: structlog.stdlib.BoundLogger, operation: str, **kwargs: Any) -> None:
    """Log ML operation information."""
    logger.info("ML operation", operation=operation, **kwargs)


def log_db_operation(
    logger: structlog.stdlib.BoundLogger, operation: str, table: str = None, **kwargs: Any
) -> None:
    """Log database operation information."""
    logger.info("DB operation", operation=operation, table=table, **kwargs)
