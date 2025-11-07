#!/usr/bin/env python3
"""
Test script for the updated image endpoints that require UUID.
"""

import httpx
import json
import uuid


def test_add_image_with_uuid():
    """Test the updated /images endpoint that requires a vector_id UUID."""
    url = "http://localhost:9000/api/v1/images"

    # Generate a unique UUID for this test
    test_uuid = str(uuid.uuid4())

    payload = {
        "vector_id": test_uuid,
        "image_url": (
            "https://arthur.io/img/art/jpg/000173448c8e9663c/kumi-yamashita/pathway/large/kumi-yamashita--pathway.jpg"
        ),
        "artist_id": "test_artist",
        "image_id": "test_image_001",
    }

    headers = {"Content-Type": "application/json"}

    print("Testing /images endpoint with required UUID...")
    print(f"Vector ID: {test_uuid}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            resp_data = response.json()
            if resp_data.get("vector_id") == test_uuid:
                print("✅ Single image endpoint working with UUID!")
            else:
                print("❌ UUID mismatch in response!")
        else:
            print("❌ Single image endpoint failed!")

    except Exception as e:
        print(f"❌ Request failed: {e}")


def test_missing_uuid():
    """Test that the endpoint returns an error when UUID is missing."""
    url = "http://localhost:9000/api/v1/images"

    payload = {
        # Missing vector_id
        "image_url": (
            "https://arthur.io/img/art/jpg/000173448c8e9663c/kumi-yamashita/pathway/large/kumi-yamashita--pathway.jpg"
        ),
        "artist_id": "test_artist",
        "image_id": "test_image_002",
    }

    print("\nTesting /images endpoint without UUID (should fail)...")

    try:
        with httpx.Client() as client:
            response = client.post(url, json=payload)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 422:  # Validation error
            print("✅ Correctly rejected request without UUID!")
        else:
            print("❌ Should have rejected request without UUID!")

    except Exception as e:
        print(f"❌ Request failed: {e}")


def test_batch_with_uuids():
    """Test the batch endpoint with UUIDs."""
    url = "http://localhost:9000/api/v1/images/batch"

    uuid1 = str(uuid.uuid4())
    uuid2 = str(uuid.uuid4())

    payload = {
        "images": [
            {
                "vector_id": uuid1,
                "image_url": (
                    "https://arthur.io/img/art/jpg/000173448c8f12562/kumi-yamashita/kelly/large/kumi-yamashita--kelly.jpg"
                ),
                "artist_id": "kumi_yamashita",
                "image_id": "kelly_001",
            },
            {
                "vector_id": uuid2,
                "image_url": (
                    "https://arthur.io/img/art/jpg/000173448c8f2e675/kumi-yamashita/someone-elses-mess/large/kumi-yamashita--someone-elses-mess.jpg"
                ),
                "artist_id": "kumi_yamashita",
                "image_id": "someone_mess_001",
            },
        ]
    }

    print(f"\nTesting /images/batch endpoint with UUIDs...")
    print(f"UUID 1: {uuid1}")
    print(f"UUID 2: {uuid2}")

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        if response.status_code == 200:
            resp_data = response.json()
            if len(resp_data.get("results", [])) == 2:
                print("✅ Batch endpoint working with UUIDs!")
            else:
                print("❌ Batch processing incomplete!")
        else:
            print("❌ Batch endpoint failed!")

    except Exception as e:
        print(f"❌ Request failed: {e}")


if __name__ == "__main__":
    test_add_image_with_uuid()
    test_missing_uuid()
    test_batch_with_uuids()
