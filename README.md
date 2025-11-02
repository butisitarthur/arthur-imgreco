# Arthur Image Recognition <!-- omit in toc -->

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-00a393.svg?style=flat-square)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat-square)](https://www.docker.com/)

**Modern, scalable image recognition server with vector similarity search**

Arthur Image Recognition 2.0 is an AI-powered image recognition and similarity search system. Built with modern machine learning technologies, it provides semantic image understanding using CLIP embeddings and high-performance vector similarity search.

<br>

<details>
<summary markdown>Table of contents</summary>

- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Using Docker (Recommended)](#using-docker-recommended)
- [Adding Images to the Database](#adding-images-to-the-database)
  - [Adding Individual Images](#adding-individual-images)
  - [Batch Image Upload](#batch-image-upload)
  - [Monitoring Upload Progress](#monitoring-upload-progress)
- [Example Usage](#example-usage)
  - [Basic Image Matching (Legacy Compatible)](#basic-image-matching-legacy-compatible)
  - [Enhanced Similarity Search](#enhanced-similarity-search)
  - [Batch Processing](#batch-processing)
  - [Local Development](#local-development)
- [API Reference](#api-reference)
  - [Legacy Compatibility Endpoints](#legacy-compatibility-endpoints)
  - [Enhanced API v1 Endpoints](#enhanced-api-v1-endpoints)
- [Database Requirements](#database-requirements)
  - [PostgreSQL Setup](#postgresql-setup)
  - [Qdrant Configuration](#qdrant-configuration)
- [Architecture](#architecture)
  - [Tech Stack](#tech-stack)
- [Deployment](#deployment)
  - [Production Docker Setup](#production-docker-setup)
  - [Kubernetes Deployment](#kubernetes-deployment)
  - [Environment Variables](#environment-variables)
- [Performance \& Scaling](#performance--scaling)
  - [Expected Performance](#expected-performance)
  - [Scaling Recommendations](#scaling-recommendations)
  - [Resource Requirements](#resource-requirements)
- [Monitoring \& Observability](#monitoring--observability)
  - [Health Checks](#health-checks)
  - [Metrics \& Monitoring](#metrics--monitoring)
  - [Alerting](#alerting)
- [Development](#development)
  - [Running Tests](#running-tests)
  - [Code Quality](#code-quality)
  - [Adding New Features](#adding-new-features)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Getting Help](#getting-help)
- [License](#license)
- [Contributing](#contributing)

</details>

<br>

## Quick Start

### Prerequisites

-   **Docker** and **Docker Compose**
-   **Python 3.14+** (for local development)
-   **Poetry** (for dependency management)

<br>

### Using Docker (Recommended)

1. **Clone and navigate to the project:**

    ```bash
    cd arthur-imgreco
    ```

2. **Copy environment configuration:**

    ```bash
    cp .env.example .env
    # Edit .env with your configuration
    ```

3. **Start all services:**

    ```bash
    docker-compose up -d
    ```

4. **Verify the setup:**

    ```bash
    curl http://localhost:8000/health
    ```

5. **Access the API documentation:**
    - **Interactive docs**: http://localhost:8000/api/v1/docs
    - **ReDoc**: http://localhost:8000/api/v1/redoc
    - **Grafana**: http://localhost:3000 (admin/admin123)

<br>

## Adding Images to the Database

### Adding Individual Images

**Method 1: Using the Legacy API (Simple)**

```bash
# Add image by URL
curl -X POST http://localhost:8000/artist/image/artist123/entry456/view789 \
  -F "imgUrl=https://example.com/artwork.jpg"

# Add image by file upload
curl -X POST http://localhost:8000/artist/image/artist123/entry456/view789 \
  -F "imgFile=@/path/to/artwork.jpg"
```

**Method 2: Using Python**

```python
import requests

# Add by URL
response = requests.post(
    'http://localhost:8000/artist/image/artist123/entry456/view789',
    data={'imgUrl': 'https://example.com/artwork.jpg'}
)

# Add by file upload
with open('artwork.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/artist/image/artist123/entry456/view789',
        files={'imgFile': f}
    )

print(response.json())
```

### Batch Image Upload

**Method 1: JSON Batch Upload**

```python
import requests

# Batch upload multiple images
batch_data = {
    "images": [
        {
            "artist_id": "artist123",
            "entry_id": "entry001",
            "view_id": "view001",
            "url": "https://example.com/art1.jpg"
        },
        {
            "artist_id": "artist123",
            "entry_id": "entry002",
            "view_id": "view002",
            "url": "https://example.com/art2.jpg"
        }
    ]
}

response = requests.post(
    'http://localhost:8000/artist/image',
    json=batch_data
)
```

**Method 2: Directory Bulk Upload Script**

```python
import os
import requests
from pathlib import Path

def bulk_upload_directory(directory_path, artist_id, base_url="http://localhost:8000"):
    """Upload all images from a directory"""

    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp'}

    for i, image_path in enumerate(Path(directory_path).iterdir()):
        if image_path.suffix.lower() in image_extensions:

            # Generate IDs based on filename
            entry_id = f"entry_{i:06d}"
            view_id = f"view_{i:06d}"

            with open(image_path, 'rb') as f:
                response = requests.post(
                    f'{base_url}/artist/image/{artist_id}/{entry_id}/{view_id}',
                    files={'imgFile': f}
                )

            if response.status_code == 200:
                print(f"✅ Uploaded: {image_path.name}")
            else:
                print(f"❌ Failed: {image_path.name} - {response.text}")

# Usage
bulk_upload_directory("/path/to/artworks", "artist123")
```

**Method 3: CSV-Driven Batch Upload**

```python
import pandas as pd
import requests

def upload_from_csv(csv_file, base_url="http://localhost:8000"):
    """Upload images from a CSV file with metadata"""

    # CSV format: artist_id,entry_id,view_id,image_url,title,description
    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        response = requests.post(
            f'{base_url}/artist/image/{row.artist_id}/{row.entry_id}/{row.view_id}',
            data={
                'imgUrl': row.image_url,
                'title': row.get('title', ''),
                'description': row.get('description', '')
            }
        )

        if response.status_code == 200:
            print(f"✅ Uploaded: {row.artist_id}/{row.entry_id}")
        else:
            print(f"❌ Failed: {row.artist_id}/{row.entry_id}")

# Usage
upload_from_csv("artworks.csv")
```

### Monitoring Upload Progress

```bash
# Check indexing status
curl http://localhost:8000/api/v1/index/stats

# Monitor system health during uploads
curl http://localhost:8000/health

# Check Docker container logs
docker compose logs -f arthur-imgreco
```

<br>

## Example Usage

### Basic Image Matching (Legacy Compatible)

```python
import requests

# Match by URL
response = requests.post('http://localhost:8000/match',
    json={'imgUrl': 'https://example.com/artwork.jpg'})

# Match by file upload
with open('artwork.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/match',
        files={'imgFile': f})

print(response.json())
```

<br>
### Enhanced Similarity Search

```python
import requests

# Advanced similarity search
response = requests.post('http://localhost:8000/api/v1/similarity/search',
    json={
        'image_url': 'https://example.com/artwork.jpg',
        'artist_filter': ['artist123', 'artist456'],
        'similarity_threshold': 0.8,
        'max_results': 20
    })

results = response.json()
print(f"Found {results['total_results']} matches in {results['query_time_ms']}ms")
```

<br>
### Batch Processing

```python
# Process multiple images
response = requests.post('http://localhost:8000/api/v1/similarity/batch',
    json={
        'images': [
            'https://example.com/art1.jpg',
            'https://example.com/art2.jpg'
        ],
        'max_results_per_image': 10
    })
```

<br>

### Local Development

1. **Install dependencies:**

    ```bash
    poetry install
    poetry shell
    ```

2. **Start external services:**

    ```bash
    docker-compose up -d postgres redis qdrant
    ```

3. **Run the application:**
    ```bash
    uvicorn arthur_imgreco.main:app --reload --host 0.0.0.0 --port 8000
    ```

<br>

## API Reference

### Legacy Compatibility Endpoints

**Fully compatible with arthur-imgreco v1:**

```http
POST /match
POST /match/{artist_id}
POST /artist/image
POST /artist/image/{artist_id}/{entry_id}/{view_id}
GET /status
POST /unified-index
```

<br>

### Enhanced API v1 Endpoints

**New modern endpoints with advanced features:**

```http
# Enhanced similarity search
POST /api/v1/similarity/search
POST /api/v1/similarity/batch
GET /api/v1/similarity/{image_id}

# Analytics and insights
GET /api/v1/index/stats
GET /api/v1/artists/{artist_id}/analytics

# System management
POST /api/v1/index/rebuild
GET /api/v1/models/info
GET /api/v1/metrics
```

<br>

## Database Requirements

### PostgreSQL Setup

The application requires PostgreSQL 17+ with the pgvector extension:

```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create application database
CREATE DATABASE arthur_imgreco;

-- Create user (in production, use strong passwords)
CREATE USER arthur_app WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE arthur_imgreco TO arthur_app;
```

### Qdrant Configuration

Qdrant runs in Docker and requires no additional setup. The application automatically creates the required collections:

-   **Images collection**: Stores CLIP embeddings for artwork images
-   **Artists collection**: Stores artist-level aggregated embeddings

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │───▶│  Vector Search   │───▶│   PostgreSQL    │
│   Gateway       │    │   (Qdrant)       │    │   Metadata      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Image         │    │   ML Pipeline    │    │     Redis       │
│   Processing    │    │   (CLIP/ViT)     │    │     Cache       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Tech Stack

-   **API Framework**: FastAPI 0.115+ with automatic OpenAPI docs
-   **ML Models**: CLIP (OpenAI), PyTorch 2.6+, Transformers 4.57+
-   **Vector Database**: Qdrant 1.15+ for similarity search
-   **Database**: PostgreSQL 17+ with pgvector extension
-   **Caching**: Redis 7+
-   **Monitoring**: Prometheus + Grafana
-   **Deployment**: Docker + Docker Compose

## Deployment

### Production Docker Setup

1. **Create production environment file:**

    ```bash
    cp .env.example .env.production
    # Update with production values
    ```

2. **Use production compose file:**
    ```bash
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    ```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Monitor deployment
kubectl get pods -l app=arthur-imgreco
```

### Environment Variables

Key configuration options:

```bash
# Application
APP_NAME=Arthur Image Recognition
DEBUG=false
LOG_LEVEL=INFO

# Database connections
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379
QDRANT_URL=http://host:6333

# ML Configuration
CLIP_MODEL_NAME=ViT-B/32
MAX_IMAGE_SIZE=1024
BATCH_SIZE=32

# Security
SECRET_KEY=your-secret-key
CORS_ORIGINS=["https://yourdomain.com"]

# Performance
MAX_WORKERS=4
INDEX_BUILD_CONCURRENCY=2
```

## Performance & Scaling

### Expected Performance

| Metric           | Performance        |
| ---------------- | ------------------ |
| Query Time       | **<1 second**      |
| Image Processing | **1-2 seconds**    |
| Index Build      | **2-3 minutes**    |
| Memory Usage     | **Optimized**      |
| Concurrent Users | **High**           |
| Max Images       | **Millions**       |
| Throughput       | **30+ images/min** |

### Scaling Recommendations

-   **Horizontal scaling**: Run multiple FastAPI instances behind load balancer
-   **Vector database**: Qdrant supports clustering for distributed search
-   **Caching**: Redis cluster for high-availability caching
-   **Database**: PostgreSQL read replicas for query scaling

### Resource Requirements

**Minimum (Development):**

-   CPU: 2 cores
-   RAM: 4GB
-   Storage: 10GB

**Production (1M images):**

-   CPU: 8+ cores
-   RAM: 32GB+
-   Storage: 500GB+ SSD
-   GPU: Optional, improves embedding generation speed

## Monitoring & Observability

### Health Checks

```bash
# Basic health
curl http://localhost:8000/health

# Detailed health with service status
curl http://localhost:8000/api/health
```

### Metrics & Monitoring

-   **Prometheus metrics**: http://localhost:8000/api/v1/metrics
-   **Grafana dashboards**: http://localhost:3000
-   **Application logs**: Structured JSON logs with correlation IDs
-   **Performance tracking**: Request duration, error rates, ML model performance

### Alerting

Key metrics to monitor:

-   API response time > 5 seconds
-   Error rate > 1%
-   Vector database connection failures
-   High memory usage (>80%)
-   Index build failures

## Development

### Running Tests

```bash
# Install dev dependencies
poetry install --with dev

# Run all tests
pytest

# Run with coverage
pytest --cov=arthur_imgreco --cov-report=html

# Run specific test types
pytest tests/unit/
pytest tests/integration/
```

### Code Quality

```bash
# Format code
black src/ tests/
ruff check src/ tests/ --fix

# Type checking
mypy src/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Adding New Features

1. **Create feature branch**: `git checkout -b feature/new-feature`
2. **Implement with tests**: Follow existing patterns
3. **Update documentation**: API docs, README updates
4. **Run quality checks**: Tests, linting, type checking
5. **Create pull request**: Include performance impact analysis

## Troubleshooting

### Common Issues

**Service won't start:**

```bash
# Check logs
docker-compose logs arthur-imgreco

# Verify dependencies
docker-compose ps
```

**Slow queries:**

```bash
# Check index status
curl http://localhost:8000/api/v1/index/stats

# Monitor resource usage
docker stats
```

**Memory issues:**

```bash
# Reduce batch size
export BATCH_SIZE=16

# Enable memory monitoring
export LOG_LEVEL=DEBUG
```

### Getting Help

-   **Documentation**: Check the `/docs` directory
-   **Issues**: GitHub Issues for bug reports
-   **Discussions**: GitHub Discussions for questions
-   **Logs**: Enable DEBUG logging for detailed troubleshooting

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Ensure all checks pass
5. Submit a pull request

---

**Built with ❤️ for the Arthur team**
