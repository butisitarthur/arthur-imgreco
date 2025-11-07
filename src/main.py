"""
Main FastAPI application for Arthur Image Recognition 2.0
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from core.config import settings
from core.logging import (
    configure_logging,
    get_logger,
    log_api_request,
    log_api_response,
)
from api import health, v1

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter("arthur_requests_total", "Total requests", ["method", "endpoint", "status"])
REQUEST_DURATION = Histogram("arthur_request_duration_seconds", "Request duration")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    logger.info("Starting Arthur Image Recognition 2.0", version=settings.app_version)

    # Startup tasks
    try:
        # Initialize core services
        logger.info("Initializing ML services...")
        from ml.clip_service import CLIPEmbeddingService
        from ml.vector_db import QdrantService

        # Initialize CLIP service
        clip_service = CLIPEmbeddingService()
        await clip_service.load_model()
        logger.info("CLIP model loaded successfully")

        # Initialize vector database service
        logger.info("Initializing vector database connection...")
        qdrant_service = QdrantService()
        # Note: Qdrant connection is established on first use
        logger.info("Vector database service initialized")

        # Store services in app state for access from routes
        app.state.clip_service = clip_service
        app.state.qdrant_service = qdrant_service

        logger.info("Application startup complete")
        yield

    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    finally:
        # Cleanup tasks
        logger.info("Shutting down Arthur Image Recognition 2.0")
        # Cleanup CLIP service if needed
        if hasattr(app.state, "clip_service") and app.state.clip_service:
            # CLIP service cleanup is handled automatically by PyTorch
            pass
        # Qdrant client cleanup is handled automatically by the client


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Modern, scalable image recognition server with vector similarity search",
        openapi_url=f"{settings.api_v1_str}/openapi.json",
        docs_url=f"{settings.api_v1_str}/docs",
        redoc_url=f"{settings.api_v1_str}/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request logging middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next) -> Response:
        """Log API requests and responses."""
        start_time = time.time()

        # Log request
        log_api_request(
            logger,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None,
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Update metrics
        REQUEST_COUNT.labels(
            method=request.method, endpoint=request.url.path, status=response.status_code
        ).inc()
        REQUEST_DURATION.observe(duration)

        # Log response
        log_api_response(
            logger,
            status_code=response.status_code,
            duration=duration,
        )

        return response

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(v1.router, prefix=settings.api_v1_str, tags=["API v1"])

    # Metrics endpoint
    @app.get("/api/v1/metrics", include_in_schema=False)
    async def metrics() -> Response:
        """Prometheus metrics endpoint."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error"},
        )

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=9000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
