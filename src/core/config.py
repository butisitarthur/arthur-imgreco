"""
Configuration management using Pydantic Settings.
"""

from typing import List
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration settings."""

    # Application
    app_name: str = Field(default="Arthur Image Recognition", env="APP_NAME")
    app_version: str = Field(default="2.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # API
    api_v1_str: str = Field(default="/api/v1", env="API_V1_STR")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    max_upload_size: int = Field(default=100 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 100MB

    # Redis
    redis_url: str = Field(default="", env="REDIS_URL")

    # Qdrant
    qdrant_url: str = Field(default="", env="QDRANT_URL")
    qdrant_api_key: str = Field(default="", env="QDRANT_API_KEY")

    # ML Model Configuration
    clip_model_name: str = Field(default="ViT-B/32", env="CLIP_MODEL_NAME")
    embedding_dimension: int = Field(default=512, env="EMBEDDING_DIMENSION")
    max_image_size: int = Field(default=1024, env="MAX_IMAGE_SIZE")

    # Performance
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    index_build_concurrency: int = Field(default=2, env="INDEX_BUILD_CONCURRENCY")

    # Security (for future authentication system)
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    indexing_threshold: int = Field(default=20000, env="INDEXING_THRESHOLD")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        """Parse CORS origins from environment."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
