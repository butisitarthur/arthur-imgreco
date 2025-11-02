import asyncio
import time
from arthur_imgreco.ml.clip_service import CLIPEmbeddingService

async def test_clip():
    print("Testing CLIP embedding generation...")
    
    # Initialize service
    service = CLIPEmbeddingService()
    
    # Load model
    start_time = time.time()
    await service.load_model()
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    
    # Test with a sample image URL
    test_url = "https://arthur.io/img/art/jpg/000173449bb75752e/m-c-escher/three-spheres-ii/large/m-c-escher--three-spheres-ii.webp"
    
    try:
        print(f"Processing image: {test_url}")
        start_time = time.time()
        embedding = await service.generate_embedding(test_url)
        process_time = time.time() - start_time
        
        print(f"✅ Embedding generated successfully!")
        print(f"   Shape: {embedding.shape}")
        print(f"   Processing time: {process_time:.3f}s")
        print(f"   Sample values: {embedding[:5]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_clip())
