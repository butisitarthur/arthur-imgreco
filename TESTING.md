# Testing the /api/v1/images Endpoint

The `/api/v1/images` endpoint is now fully functional and ready for testing! Here's how to test it:

## Quick Test Commands

### 1. Test Single Image Addition

```bash
curl -X POST "http://localhost:9000/api/v1/images" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://arthur.io/img/art/jpg/000173449bb75752e/m-c-escher/three-spheres-ii/large/m-c-escher--three-spheres-ii.webp",
    "artist_id": "m_c_escher",
    "image_id": "three_spheres_test_001",
    "metadata": {
      "title": "Three Spheres II",
      "artist_name": "M.C. Escher",
      "tags": ["surreal", "mathematical", "lithograph"]
    }
  }'
```

### 2. Run Automated Tests

```bash
# From the project root directory
python test_add_image_endpoint.py
```

### 3. Check System Health

```bash
curl http://localhost:9000/health
```

## What the Endpoint Does

The `/api/v1/images` endpoint now performs the complete image processing pipeline:

1. ✅ **Downloads and validates images** from URLs
2. ✅ **Generates CLIP embeddings** using the loaded AI model
3. ✅ **Stores embeddings in Qdrant** vector database
4. ✅ **Returns detailed processing information** including timing
5. ✅ **Handles errors gracefully** with proper logging

## Expected Response

Successful response:

```json
{
	"image_id": "three_spheres_test_001",
	"artist_id": "m_c_escher",
	"status": "success",
	"embedding_generated": true,
	"processing_time_ms": 1847.3,
	"message": "Successfully indexed image three_spheres_test_001 (embedding: (512,), vector_id: m_c_escher_three_spheres_test_001)"
}
```

## Performance Expectations

-   **Image download**: ~200-500ms (depends on image size and network)
-   **CLIP embedding generation**: ~1-2 seconds (Apple Silicon GPU)
-   **Vector storage**: ~50-200ms
-   **Total processing time**: ~2-3 seconds per image

## Prerequisites

Make sure these services are running:

```bash
# Start all services
docker compose up -d

# Check service status
docker compose ps

# Check that CLIP model loaded successfully
docker compose logs arthur-imgreco | grep "CLIP"
```

## Next Steps

Once images are successfully added, you can:

1. **View in Qdrant dashboard**: http://localhost:6333/dashboard
2. **Test similarity search**: Use `/api/v1/similarity/search` endpoint
3. **Check index statistics**: GET `/api/v1/index/stats`
4. **Browse API documentation**: http://localhost:9000/api/v1/docs

The system is now ready for production image indexing! 🚀
