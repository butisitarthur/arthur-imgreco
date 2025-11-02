# Arthur Image Recognition <!-- omit in toc -->

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-00a393.svg?style=flat-square)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-orange.svg?style=flat-square)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat-square)](https://www.docker.com/)

**Modern, scalable image recognition server with vector similarity search**

Arthur Image Recognition 2.0 is an AI-powered image recognition and similarity search system. Built with modern machine learning technologies, it provides semantic image understanding using CLIP embeddings and high-performance vector similarity search.

<sub>Built with Claude Sonnet 4, Nov 2025</sub>

<br>

[Developer guide](docs/DEVELOPER_GUIDE.md) - Feed to AI for further development tasks

<details>
<summary markdown>Table of contents</summary>

-   [Quick Start](#quick-start)
    -   [Prerequisites](#prerequisites)
    -   [Using Docker (Recommended)](#using-docker-recommended)
-   [Adding Images to the Database](#adding-images-to-the-database)
    -   [Adding Individual Images](#adding-individual-images)
    -   [Batch Image Upload](#batch-image-upload)
    -   [Monitoring Upload Progress](#monitoring-upload-progress)
-   [Example Usage](#example-usage)
    -   [Basic Image Matching (Legacy Compatible)](#basic-image-matching-legacy-compatible)
    -   [Enhanced Similarity Search](#enhanced-similarity-search)
    -   [Batch Processing](#batch-processing)
    -   [Local Development](#local-development)
-   [API Reference](#api-reference)
    -   [Legacy Compatibility Endpoints](#legacy-compatibility-endpoints)
    -   [Enhanced API v1 Endpoints](#enhanced-api-v1-endpoints)
-   [Database Requirements](#database-requirements)
    -   [PostgreSQL Setup](#postgresql-setup)
    -   [Qdrant Configuration](#qdrant-configuration)
-   [Architecture](#architecture)
    -   [Tech Stack](#tech-stack)
-   [Deployment](#deployment)
    -   [Production Docker Setup](#production-docker-setup)
    -   [Kubernetes Deployment](#kubernetes-deployment)
    -   [Environment Variables](#environment-variables)
-   [Performance \& Scaling](#performance--scaling)
    -   [Expected Performance](#expected-performance)
    -   [Scaling Recommendations](#scaling-recommendations)
    -   [Resource Requirements](#resource-requirements)
-   [Monitoring \& Observability](#monitoring--observability)
    -   [Health Checks](#health-checks)
    -   [Metrics \& Monitoring](#metrics--monitoring)
    -   [Alerting](#alerting)
-   [Development](#development)
    -   [Running Tests](#running-tests)
    -   [Code Quality](#code-quality)
    -   [Adding New Features](#adding-new-features)
-   [Troubleshooting](#troubleshooting)
    -   [Common Issues](#common-issues)
    -   [Getting Help](#getting-help)
-   [License](#license)
-   [Contributing](#contributing)

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
    curl http://localhost:9000/health
    ```

5. **Access the API documentation:**
    - **Interactive docs**: http://localhost:9000/api/v1/docs
    - **ReDoc**: http://localhost:9000/api/v1/redoc
    - **Grafana**: http://localhost:3000 (admin/admin123)

<br>

## Adding Images to the Database

### Adding Individual Images

**Method 1: Modern API (Recommended)**

```python
import requests

# Add image by URL with metadata
response = requests.post(
    'http://localhost:9000/api/v1/images',
    json={
        "image_url": "https://example.com/artwork.jpg",
        "artist_id": "artist123",
        "image_id": "unique_image_001",
        "metadata": {
            "title": "Starry Night",
            "description": "Famous painting by Van Gogh",
            "tags": ["post-impressionism", "night", "stars"],
            "artist_name": "Vincent van Gogh",
            "creation_date": "1889",
            "medium": "oil on canvas"
        }
    }
)

print(response.json())
# Returns: {"image_id": "unique_image_001", "status": "success", ...}
```

**Method 2: File Upload**

```python
# Upload image file directly
files = {'image_file': open('artwork.jpg', 'rb')}
data = {
    'artist_id': 'artist123',
    'image_id': 'unique_image_002',
    'title': 'Local Artwork',
    'tags': 'modern,abstract,colorful'
}

response = requests.post(
    'http://localhost:9000/api/v1/images/upload',
    files=files,
    data=data
)
```

**Method 3: cURL (Simple)**

```bash
# Add image via URL
curl -X POST http://localhost:9000/api/v1/images \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/art.jpg",
    "artist_id": "artist123",
    "image_id": "image001"
  }'

# Upload file
curl -X POST http://localhost:9000/api/v1/images/upload \
  -F "artist_id=artist123" \
  -F "image_id=image002" \
  -F "image_file=@artwork.jpg"
```

### Batch Image Upload

**Method 1: Batch API (Most Efficient)**

```python
import requests

# Batch upload multiple images
batch_data = {
    "images": [
        {
            "image_url": "https://example.com/art1.jpg",
            "artist_id": "artist123",
            "image_id": "batch_001",
            "metadata": {
                "title": "Artwork 1",
                "tags": ["modern", "abstract"]
            }
        },
        {
            "image_url": "https://example.com/art2.jpg",
            "artist_id": "artist123",
            "image_id": "batch_002",
            "metadata": {
                "title": "Artwork 2",
                "tags": ["classical", "portrait"]
            }
        }
    ]
}

response = requests.post(
    'http://localhost:9000/api/v1/images/batch',
    json=batch_data
)

result = response.json()
print(f"Processed: {result['successful']} successful, {result['failed']} failed")
```

**Method 2: Directory Bulk Upload**

```python
import requests
from pathlib import Path

def bulk_upload_directory(directory_path, artist_id, base_url="http://localhost:9000"):
    """Upload all images from a directory using modern API"""

    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.tiff', '.bmp'}
    images = []

    # Prepare batch request
    for i, image_path in enumerate(Path(directory_path).iterdir()):
        if image_path.suffix.lower() in image_extensions:
            images.append({
                "image_url": f"file://{image_path.absolute()}",  # For local files
                "artist_id": artist_id,
                "image_id": f"{artist_id}_{i:06d}",
                "metadata": {
                    "title": image_path.stem,
                    "source_file": image_path.name
                }
            })

    # Send batch request
    if images:
        response = requests.post(
            f'{base_url}/api/v1/images/batch',
            json={"images": images}
        )

        result = response.json()
        print(f"✅ Processed {len(images)} images")
        print(f"   Successful: {result['successful']}")
        print(f"   Failed: {result['failed']}")
        return result

    print("No images found in directory")
    return None

# Usage
bulk_upload_directory("/path/to/artworks", "artist123")
```

**Method 3: CSV-Driven Upload**

```python
import pandas as pd
import requests

def upload_from_csv(csv_file, base_url="http://localhost:9000"):
    """Upload images from CSV using batch API"""

    # CSV format: artist_id,image_id,image_url,title,description,tags
    df = pd.read_csv(csv_file)

    # Convert CSV to batch format
    images = []
    for _, row in df.iterrows():
        image_data = {
            "image_url": row['image_url'],
            "artist_id": row['artist_id'],
            "image_id": row['image_id'],
            "metadata": {
                "title": row.get('title', ''),
                "description": row.get('description', ''),
                "tags": row.get('tags', '').split(',') if row.get('tags') else []
            }
        }
        images.append(image_data)

    # Process in batches of 50
    batch_size = 50
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]

        response = requests.post(
            f'{base_url}/api/v1/images/batch',
            json={"images": batch}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch {i//batch_size + 1}: {result['successful']}/{len(batch)} successful")
        else:
            print(f"❌ Batch {i//batch_size + 1} failed: {response.text}")

# Usage
upload_from_csv("artworks.csv")
```

### Monitoring and Management

**Check Upload Status:**

```bash
# View index statistics
curl http://localhost:9000/api/v1/index/stats

# Get specific image info
curl http://localhost:9000/api/v1/images/unique_image_001

# Monitor system health
curl http://localhost:9000/health
```

**Delete Images:**

````bash
# Remove image from index
curl -X DELETE http://localhost:9000/api/v1/images/image_to_delete
```<br>

## Example Usage

### Adding Images (Modern API)

```python
import requests

# Add single image with metadata
response = requests.post('http://localhost:9000/api/v1/images',
    json={
        'image_url': 'https://example.com/artwork.jpg',
        'artist_id': 'vangogh',
        'image_id': 'starry_night_001',
        'metadata': {
            'title': 'The Starry Night',
            'tags': ['post-impressionism', 'night', 'swirls'],
            'creation_date': '1889'
        }
    })

print(f"✅ {response.json()['message']}")
````

### Similarity Search

```python
import requests

# Search for similar images
response = requests.post('http://localhost:9000/api/v1/similarity/search',
    json={
        'image_url': 'https://example.com/query.jpg',
        'artist_filter': ['vangogh', 'monet'],
        'similarity_threshold': 0.8,
        'max_results': 20
    })

results = response.json()
print(f"Found {results['total_results']} matches in {results['query_time_ms']}ms")

for match in results['results']:
    print(f"  {match['artist_id']}/{match['image_id']} - {match['similarity_score']:.3f}")
```

### Batch Operations

```python
# Add multiple images efficiently
batch_request = {
    "images": [
        {
            "image_url": "https://example.com/art1.jpg",
            "artist_id": "picasso",
            "image_id": "guernica_001"
        },
        {
            "image_url": "https://example.com/art2.jpg",
            "artist_id": "picasso",
            "image_id": "les_demoiselles_001"
        }
    ]
}

response = requests.post('http://localhost:9000/api/v1/images/batch', json=batch_request)
result = response.json()
print(f"Processed: {result['successful']} successful, {result['failed']} failed")
```

### Legacy Compatibility

```python
import requests

# Legacy API (backward compatible)
response = requests.post('http://localhost:9000/match',
    json={'imgUrl': 'https://example.com/artwork.jpg'})

print(response.json())
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
    uvicorn arthur_imgreco.main:app --reload --host 0.0.0.0 --port 9000
    ```

<br>

## API Reference

### Modern API v1 Endpoints (Recommended)

**Image Management:**

```http
POST /api/v1/images              # Add single image
POST /api/v1/images/batch        # Add multiple images efficiently
POST /api/v1/images/upload       # Upload image file directly
GET /api/v1/images/{image_id}    # Get image information
DELETE /api/v1/images/{image_id} # Remove image from index
```

**Similarity Search:**

```http
POST /api/v1/similarity/search   # Advanced similarity search
POST /api/v1/similarity/batch    # Batch similarity processing
GET /api/v1/similarity/{image_id} # Find similar to existing image
```

**Analytics & System:**

```http
GET /api/v1/index/stats          # Index statistics and performance
GET /api/v1/artists/{id}/analytics # Artist-specific analytics
GET /api/v1/models/info          # ML model information
POST /api/v1/index/rebuild       # Trigger index rebuild
```

### Legacy Compatibility Endpoints

**For backward compatibility with arthur-imgreco v1:**

```http
POST /match                      # Image matching (legacy format)
POST /artist/image/{artist_id}/{entry_id}/{view_id}  # Legacy image addition
GET /status                      # System status (legacy format)
POST /unified-index              # Legacy index management
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
curl http://localhost:9000/health

# Detailed health with service status
curl http://localhost:9000/api/health
```

### Metrics & Monitoring

-   **Prometheus metrics**: http://localhost:9000/api/v1/metrics
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
curl http://localhost:9000/api/v1/index/stats

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
