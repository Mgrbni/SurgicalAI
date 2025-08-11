#!/usr/bin/env python3
"""
Quick API test script to verify the enhanced pipeline is working
"""

import requests
import json
from pathlib import Path

# Test health endpoint
print("🏥 Testing health endpoint...")
try:
    response = requests.get('http://localhost:8001/api/health')
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")

# Test analysis endpoint with sample image
print("\n🔬 Testing analysis endpoint...")
sample_image = Path("data/samples/lesion.jpg")
if sample_image.exists():
    try:
        with open(sample_image, 'rb') as f:
            files = {
                'file': ('lesion.jpg', f, 'image/jpeg'),
                'payload': (None, json.dumps({'subunit': 'cheek_lateral'}))
            }
            response = requests.post('http://localhost:8001/api/analyze', files=files)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Analysis successful: {data.get('ok', False)}")
                print(f"Run ID: {data.get('run_id', 'None')}")
                print(f"Diagnosis: {data.get('diagnosis', {})}")
                print(f"VLM Observer: {'vlm_observer' in data}")
                print(f"Fusion Result: {'fusion' in data}")
                print(f"Artifacts: {len(data.get('artifacts_list', []))}")
            else:
                print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"Sample image not found: {sample_image}")
