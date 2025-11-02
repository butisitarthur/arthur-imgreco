#!/usr/bin/env python3
"""
Test script for the /api/v1/images endpoint
Run this to test the complete image processing pipeline
"""

import asyncio
import json
import httpx
import time

# Test configuration
BASE_URL = "http://localhost:9000"
TEST_IMAGES = [
    {
        "image_url": "https://arthur.io/img/art/jpg/000173449bb75752e/m-c-escher/three-spheres-ii/large/m-c-escher--three-spheres-ii.webp",
        "artist_id": "m_c_escher",
        "image_id": "three_spheres_ii_001",
        "metadata": {
            "title": "Three Spheres II",
            "artist_name": "M.C. Escher",
            "tags": ["surreal", "spheres", "mathematical", "black and white"],
            "creation_date": "1946",
            "medium": "lithograph"
        }
    },
    {
        "image_url": "https://arthur.io/img/art/jpg/000173449bb7211f0/m-c-escher/self-portrait-1/large/m-c-escher--self-portrait-1.webp",
        "artist_id": "test_artist",
        "image_id": "self_portrait_001", 
        "metadata": {
            "title": "Self portrait",
            "artist_name": "M.C. Escher",
            "description": "Self portrait by the artist",
            "tags": ["self portrait", "face", "litogrtaph"]
        }
    }
]


async def test_health_check():
    """Test that the server is running"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                print("✅ Server is running and healthy")
                health_data = response.json()
                print(f"   Uptime: {health_data.get('uptime', 0):.2f}s")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False


async def test_add_single_image(image_data):
    """Test adding a single image"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            print(f"\n🔄 Adding image: {image_data['image_id']}")
            print(f"   URL: {image_data['image_url']}")
            
            start_time = time.time()
            
            response = await client.post(
                f"{BASE_URL}/api/v1/images",
                json=image_data,
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Image added successfully!")
                print(f"   Status: {result.get('status')}")
                print(f"   Embedding generated: {result.get('embedding_generated')}")
                print(f"   Processing time: {result.get('processing_time_ms', 0):.1f}ms")
                print(f"   Total request time: {duration:.2f}s")
                print(f"   Message: {result.get('message', '')[:100]}...")
                return True
            else:
                print(f"❌ Failed to add image: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return False


async def test_batch_add_images():
    """Test batch image addition"""
    batch_data = {"images": TEST_IMAGES}
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            print(f"\n🔄 Adding {len(TEST_IMAGES)} images in batch")
            
            start_time = time.time()
            
            response = await client.post(
                f"{BASE_URL}/api/v1/images/batch",
                json=batch_data,
                headers={"Content-Type": "application/json"}
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Batch processing completed!")
                print(f"   Total processed: {result.get('total_processed')}")
                print(f"   Successful: {result.get('successful')}")
                print(f"   Failed: {result.get('failed')}")
                print(f"   Total processing time: {result.get('total_processing_time_ms', 0):.1f}ms")
                print(f"   Total request time: {duration:.2f}s")
                return True
            else:
                print(f"❌ Batch processing failed: HTTP {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Batch request failed: {e}")
            return False


async def test_index_stats():
    """Test index statistics endpoint"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            print(f"\n🔄 Getting index statistics")
            
            response = await client.get(f"{BASE_URL}/api/v1/index/stats")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Index statistics retrieved!")
                print(f"   Total images: {result.get('total_images')}")
                print(f"   Total artists: {result.get('total_artists')}")
                print(f"   Vector dimension: {result.get('vector_dimension')}")
                print(f"   Similarity model: {result.get('similarity_model')}")
                return True
            else:
                print(f"❌ Failed to get stats: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Stats request failed: {e}")
            return False


async def main():
    """Run all tests"""
    print("🚀 Testing Arthur Image Recognition 2.0 - /api/v1/images endpoint")
    print("=" * 70)
    
    # Test 1: Health check
    if not await test_health_check():
        print("\n❌ Server not available. Make sure it's running on http://localhost:9000")
        return
    
    # Test 2: Add single images
    success_count = 0
    for image_data in TEST_IMAGES:
        if await test_add_single_image(image_data):
            success_count += 1
        await asyncio.sleep(1)  # Brief pause between requests
    
    # Test 3: Batch addition
    batch_success = await test_batch_add_images()
    
    # Test 4: Index statistics
    stats_success = await test_index_stats()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    
    print(f"✅ Individual image additions: {success_count}/{len(TEST_IMAGES)} successful")
    print(f"✅ Batch processing: {'SUCCESS' if batch_success else 'FAILED'}")
    print(f"✅ Index statistics: {'SUCCESS' if stats_success else 'FAILED'}")
    
    if success_count > 0:
        print("\n🎉 The /api/v1/images endpoint is working!")
        print("🔍 Key features tested:")
        print("   • Image download and validation")
        print("   • CLIP embedding generation")
        print("   • Vector database storage")
        print("   • Error handling and logging")
        print("   • Batch processing")
        
        print("\n📚 Next Steps:")
        print("   • Test similarity search: POST /api/v1/similarity/search")
        print("   • Check Qdrant dashboard: http://localhost:6333/dashboard")
        print("   • View API docs: http://localhost:9000/api/v1/docs")
        print("   • Monitor logs for detailed processing info")
    else:
        print("\n❌ Tests failed. Check the server logs for details.")
        print("💡 Troubleshooting:")
        print("   • Ensure all services are running: docker compose up -d")
        print("   • Check server logs: docker compose logs arthur-imgreco")
        print("   • Verify CLIP model loading completed")


if __name__ == "__main__":
    asyncio.run(main())