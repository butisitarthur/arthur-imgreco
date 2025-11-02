#!/usr/bin/env python3
"""
Test script for the batch similarity endpoint.
"""

import httpx
import json
import asyncio

def test_batch_similarity():
    url = "http://localhost:9000/api/v1/similarity/batch"
    
    payload = {
        "images": [
            {
                "id": "pathway",
                "image_url": "https://arthur.io/img/art/jpg/000173448c8e9663c/kumi-yamashita/pathway/large/kumi-yamashita--pathway.jpg"
            },
            {
                "id": "kelly", 
                "image_url": "https://arthur.io/img/art/jpg/000173448c8f12562/kumi-yamashita/kelly/large/kumi-yamashita--kelly.jpg"
            }
        ],
        "max_results_per_image": 2,
        "similarity_threshold": 0.5
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("Testing batch similarity endpoint...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()
    
    try:
        with httpx.Client() as client:
            response = client.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Batch similarity endpoint working!")
        else:
            print("❌ Batch similarity endpoint failed!")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_batch_similarity()