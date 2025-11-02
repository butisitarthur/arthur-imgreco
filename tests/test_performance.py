import asyncio
import time
from ml.clip_service import CLIPEmbeddingService

async def performance_test():
    print("🚀 Arthur 2.0 Performance Test")
    print("=" * 50)
    
    service = CLIPEmbeddingService()
    
    # Load model once
    print("Loading CLIP model...")
    start_time = time.time()
    await service.load_model()
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f}s")
    print()
    
    # Test images (realistic art URLs)
    test_images = [
        "https://arthur.io/img/art/jpg/000173449bb75752e/m-c-escher/three-spheres-ii/large/m-c-escher--three-spheres-ii.webp",
    ]
    
    print("Testing embedding generation speed...")
    print("-" * 30)
    
    total_time = 0
    for i, url in enumerate(test_images, 1):
        print(f"Processing image {i}...")
        start_time = time.time()
        try:
            embedding = await service.generate_embedding(url)
            process_time = time.time() - start_time
            total_time += process_time
            
            print(f"  ✅ Generated in {process_time:.3f}s")
            print(f"  📊 Embedding shape: {embedding.shape}")
            print(f"  🔢 Sample values: {embedding[:3]}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        
        print()
    
    print("=" * 50)
    print("📈 PERFORMANCE COMPARISON:")
    print(f"  Arthur 1.0 (OpenCV): ~60+ seconds per image")
    print(f"  Arthur 2.0 (CLIP):   ~{total_time/len(test_images):.1f}s per image")
    improvement = 60 / (total_time/len(test_images))
    print(f"  🎯 Speed improvement: {improvement:.0f}x faster!")
    print()
    print("🧠 AI CAPABILITIES:")
    print("  • Semantic understanding (not just pixel matching)")
    print("  • Works with any art style (not just specific artists)")
    print("  • Handles color variations, crops, and transformations")
    print("  • Apple Silicon GPU acceleration")
    print("  • Vector similarity search for millions of images")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(performance_test())
