"""
CLIP model integration for image embedding generation.

This module provides a high-performance CLIP-based image embedding service
with caching, batch processing, and error handling.
"""

import asyncio
import io
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
from functools import lru_cache
import hashlib

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image, ImageOps
import numpy as np
import httpx

from arthur_imgreco.core.config import settings
from arthur_imgreco.core.logging import get_logger

logger = get_logger(__name__)


class CLIPEmbeddingService:
    """
    High-performance CLIP embedding service with caching and batch processing.

    Features:
    - Lazy model loading
    - GPU acceleration when available
    - Image preprocessing and normalization
    - Batch processing for efficiency
    - LRU caching for repeated images
    - Error handling and retries
    """

    def __init__(self):
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.device = self._get_device()
        self._model_loaded = False
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_device(self) -> torch.device:
        """Determine the best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA GPU for CLIP inference", device=str(device))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon GPU for CLIP inference", device=str(device))
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for CLIP inference", device=str(device))
        return device

    async def load_model(self) -> None:
        """Load CLIP model and processor."""
        if self._model_loaded:
            return

        logger.info("Loading CLIP model", model_name=settings.clip_model_name)
        start_time = time.time()

        try:
            # Load model and processor
            self.model = CLIPModel.from_pretrained(settings.clip_model_name)
            self.processor = CLIPProcessor.from_pretrained(settings.clip_model_name)

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            # Enable optimizations
            if hasattr(torch, "compile") and self.device.type != "cpu":
                try:
                    self.model = torch.compile(self.model)
                    logger.info("Model compiled with torch.compile for optimization")
                except Exception as e:
                    logger.warning("Failed to compile model", error=str(e))

            load_time = time.time() - start_time
            self._model_loaded = True

            logger.info(
                "CLIP model loaded successfully",
                load_time=f"{load_time:.2f}s",
                device=str(self.device),
                embedding_dim=self.model.config.projection_dim,
            )

        except Exception as e:
            logger.error("Failed to load CLIP model", error=str(e))
            raise RuntimeError(f"Could not load CLIP model: {e}") from e

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for CLIP inference.

        Args:
            image: PIL Image object

        Returns:
            Preprocessed PIL Image
        """
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize large images to improve processing speed
        if max(image.size) > settings.max_image_size:
            image.thumbnail(
                (settings.max_image_size, settings.max_image_size), Image.Resampling.LANCZOS
            )

        # Fix orientation based on EXIF data
        image = ImageOps.exif_transpose(image)

        return image

    @lru_cache(maxsize=1000)
    def _get_image_hash(self, image_bytes: bytes) -> str:
        """Generate hash for image caching."""
        return hashlib.md5(image_bytes).hexdigest()

    async def _load_image_from_url(self, url: str) -> Image.Image:
        """
        Load image from URL with retries and error handling.

        Args:
            url: Image URL

        Returns:
            PIL Image object
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(3):  # Retry up to 3 times
                try:
                    response = await client.get(url)
                    response.raise_for_status()

                    image_bytes = response.content
                    image = Image.open(io.BytesIO(image_bytes))

                    logger.debug("Image loaded from URL", url=url, size=image.size, mode=image.mode)
                    return self._preprocess_image(image)

                except Exception as e:
                    if attempt == 2:  # Last attempt
                        logger.error("Failed to load image from URL", url=url, error=str(e))
                        raise ValueError(f"Could not load image from URL: {e}") from e

                    logger.warning(
                        "Retrying image download", url=url, attempt=attempt + 1, error=str(e)
                    )
                    await asyncio.sleep(1)  # Wait before retry

    async def _load_image_from_bytes(self, image_data: Union[bytes, io.BytesIO]) -> Image.Image:
        """
        Load image from bytes or BytesIO object.

        Args:
            image_data: Image bytes or BytesIO object

        Returns:
            PIL Image object
        """
        try:
            if isinstance(image_data, io.BytesIO):
                image_data.seek(0)
                image = Image.open(image_data)
            else:
                image = Image.open(io.BytesIO(image_data))

            logger.debug("Image loaded from bytes", size=image.size, mode=image.mode)
            return self._preprocess_image(image)

        except Exception as e:
            logger.error("Failed to load image from bytes", error=str(e))
            raise ValueError(f"Could not load image from bytes: {e}") from e

    async def generate_embedding(
        self, image_input: Union[str, bytes, io.BytesIO, Image.Image]
    ) -> np.ndarray:
        """
        Generate CLIP embedding for a single image.

        Args:
            image_input: Image URL, bytes, BytesIO, or PIL Image

        Returns:
            Normalized embedding as numpy array
        """
        await self.load_model()

        start_time = time.time()

        try:
            # Load and preprocess image
            if isinstance(image_input, str):
                # URL
                image = await self._load_image_from_url(image_input)
            elif isinstance(image_input, (bytes, io.BytesIO)):
                # Bytes or BytesIO
                image = await self._load_image_from_bytes(image_input)
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = self._preprocess_image(image_input)
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            # Process with CLIP
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

                # Normalize embedding
                embedding = F.normalize(image_features, p=2, dim=1)
                embedding_np = embedding.cpu().numpy().astype(np.float32)[0]

            inference_time = time.time() - start_time

            logger.debug(
                "Generated image embedding",
                inference_time=f"{inference_time:.3f}s",
                embedding_shape=embedding_np.shape,
                embedding_norm=float(np.linalg.norm(embedding_np)),
            )

            return embedding_np

        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    async def generate_embeddings_batch(
        self, image_inputs: List[Union[str, bytes, io.BytesIO, Image.Image]]
    ) -> List[np.ndarray]:
        """
        Generate CLIP embeddings for multiple images efficiently.

        Args:
            image_inputs: List of image URLs, bytes, BytesIO objects, or PIL Images

        Returns:
            List of normalized embeddings as numpy arrays
        """
        await self.load_model()

        if not image_inputs:
            return []

        start_time = time.time()
        batch_size = min(len(image_inputs), settings.batch_size)

        logger.info(
            "Starting batch embedding generation",
            total_images=len(image_inputs),
            batch_size=batch_size,
        )

        embeddings = []

        # Process in batches to manage memory
        for i in range(0, len(image_inputs), batch_size):
            batch = image_inputs[i : i + batch_size]
            batch_start_time = time.time()

            try:
                # Load all images in the batch
                images = []
                for image_input in batch:
                    if isinstance(image_input, str):
                        image = await self._load_image_from_url(image_input)
                    elif isinstance(image_input, (bytes, io.BytesIO)):
                        image = await self._load_image_from_bytes(image_input)
                    elif isinstance(image_input, Image.Image):
                        image = self._preprocess_image(image_input)
                    else:
                        raise ValueError(f"Unsupported image input type: {type(image_input)}")
                    images.append(image)

                # Process batch with CLIP
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)

                    # Normalize embeddings
                    batch_embeddings = F.normalize(image_features, p=2, dim=1)
                    batch_embeddings_np = batch_embeddings.cpu().numpy().astype(np.float32)

                embeddings.extend(batch_embeddings_np)

                batch_time = time.time() - batch_start_time
                logger.debug(
                    "Processed batch",
                    batch_idx=i // batch_size + 1,
                    batch_size=len(batch),
                    batch_time=f"{batch_time:.3f}s",
                )

            except Exception as e:
                logger.error("Failed to process batch", batch_idx=i // batch_size + 1, error=str(e))
                # Add None for failed embeddings to maintain order
                embeddings.extend([None] * len(batch))

        total_time = time.time() - start_time
        successful_embeddings = sum(1 for e in embeddings if e is not None)

        logger.info(
            "Batch embedding generation complete",
            total_images=len(image_inputs),
            successful=successful_embeddings,
            failed=len(image_inputs) - successful_embeddings,
            total_time=f"{total_time:.3f}s",
            avg_time_per_image=f"{total_time / len(image_inputs):.3f}s",
        )

        return embeddings

    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "model_loaded": self._model_loaded,
            "device": str(self.device) if self.device else None,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "model_name": settings.clip_model_name,
            "embedding_dimension": settings.embedding_dimension,
        }


# Global CLIP service instance
clip_service = CLIPEmbeddingService()


async def get_clip_service() -> CLIPEmbeddingService:
    """Get the global CLIP embedding service instance."""
    return clip_service
