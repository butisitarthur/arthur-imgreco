#!/usr/bin/env python3
"""
Arthur Image Recognition 2.0 - Quick Test Script

This script validates that the system is working correctly.
Run this after the server has started successfully.
"""

import asyncio
import aiohttp
import sys
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"


async def test_health():
    """Test the health endpoint."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Health check passed")
                    print(f"   Status: {data.get('status')}")
                    print(f"   Uptime: {data.get('uptime', 0):.2f}s")
                    return True
                else:
                    print(f"❌ Health check failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False


async def test_legacy_status():
    """Test the legacy status endpoint for backwards compatibility."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ Legacy status endpoint working")
                    print(f"   Status: {data.get('status')}")
                    return True
                else:
                    print(f"❌ Legacy status failed: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Legacy status error: {e}")
            return False


async def test_docs():
    """Test that API documentation is available."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/docs") as response:
                if response.status == 200:
                    print("✅ API documentation available at http://localhost:8000/docs")
                    return True
                else:
                    print(f"❌ Docs unavailable: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Docs error: {e}")
            return False


async def test_image_match():
    """Test image matching with a sample URL."""
    # Using a public test image URL
    test_image_url = "https://via.placeholder.com/300x200.png?text=Test+Image"

    async with aiohttp.ClientSession() as session:
        try:
            # Test the legacy match endpoint
            data = aiohttp.FormData()
            data.add_field("imgUrl", test_image_url)

            async with session.post(f"{BASE_URL}/match", data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    print("✅ Image matching endpoint working")
                    print(f"   Status: {result.get('status')}")
                    print(f"   Matches: {len(result.get('matches', []))}")
                    return True
                else:
                    print(f"❌ Image matching failed: {response.status}")
                    text = await response.text()
                    print(f"   Error: {text}")
                    return False
        except Exception as e:
            print(f"❌ Image matching error: {e}")
            return False


async def run_tests():
    """Run all tests."""
    print("🚀 Arthur Image Recognition 2.0 - System Tests")
    print("=" * 50)

    tests = [
        ("Health Check", test_health),
        ("Legacy Status", test_legacy_status),
        ("API Documentation", test_docs),
        ("Image Matching", test_image_match),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        success = await test_func()
        results.append((test_name, success))

    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} {test_name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Arthur Image Recognition 2.0 is ready!")
        print("\n📚 Next Steps:")
        print("   • Visit http://localhost:8000/docs for API documentation")
        print("   • Test with real images using the /match endpoint")
        print("   • Check system metrics and logs")
        print("   • Deploy using Docker Compose for production")
    else:
        print("\n⚠️  Some tests failed. Check the server logs for details.")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)
