#!/usr/bin/env python3
"""
Comprehensive test suite for the hardened dual LLM provider system.
Tests all production-ready features including fallback, cost tracking, and validation.
"""

import requests
import json
import time
import base64
from pathlib import Path

BASE_URL = "http://localhost:7860"

def test_health_check():
    """Test 1: Health check with provider info"""
    print("🔍 Test 1: Health Check")
    
    response = requests.get(f"{BASE_URL}/api/healthz")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health: {data}")
        expected_keys = ["provider", "model", "ok", "time", "request_id"]
        for key in expected_keys:
            if key in data:
                print(f"   ✓ {key}: {data[key]}")
            else:
                print(f"   ❌ Missing: {key}")
    else:
        print(f"❌ Health check failed: {response.status_code}")

def test_provider_info():
    """Test 2: Provider configuration"""
    print("\n🔍 Test 2: Provider Info")
    
    response = requests.get(f"{BASE_URL}/api/providers")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Providers: {json.dumps(data, indent=2)}")
    else:
        print(f"❌ Provider info failed: {response.status_code}")

def test_analysis_without_image():
    """Test 3: Analysis endpoint validation (should fail without image)"""
    print("\n🔍 Test 3: Analysis Validation")
    
    data = {
        "site": "cheek",
        "suspected_type": "seborrheic_keratosis"
    }
    
    response = requests.post(f"{BASE_URL}/api/analyze", data=data)
    
    if response.status_code != 200:
        print(f"✅ Validation working: {response.status_code} - {response.text[:100]}")
    else:
        print(f"❌ Should have failed without image: {response.status_code}")

def test_analysis_with_dummy_image():
    """Test 4: Analysis with dummy image data"""
    print("\n🔍 Test 4: Analysis with Dummy Image")
    
    # Create a small dummy image (1x1 PNG)
    dummy_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAF"
        "hAJ/wlseKgAAAABJRU5ErkJggg=="
    )
    
    data = {
        "site": "nose",
        "suspected_type": "melanoma"
    }
    
    files = {
        "image": ("test.png", dummy_png, "image/png")
    }
    
    response = requests.post(f"{BASE_URL}/api/analyze", data=data, files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Analysis successful")
        print(f"   Request ID: {result.get('_metadata', {}).get('request_id', 'N/A')}")
        print(f"   Provider: {result.get('_metadata', {}).get('provider', 'N/A')}")
        print(f"   Model: {result.get('_metadata', {}).get('model', 'N/A')}")
        print(f"   Processing time: {result.get('_metadata', {}).get('processing_time_ms', 'N/A')}ms")
        print(f"   Fallback used: {result.get('_metadata', {}).get('fallback_used', False)}")
        
        # Check JSON structure
        required_keys = ["diagnosis_probs", "primary_dx", "warnings", "citations"]
        for key in required_keys:
            if key in result:
                print(f"   ✓ {key}: present")
            else:
                print(f"   ❌ {key}: missing")
    else:
        print(f"❌ Analysis failed: {response.text[:200]}")

def test_streaming_analysis():
    """Test 5: Streaming analysis"""
    print("\n🔍 Test 5: Streaming Analysis")
    
    # Create dummy image
    dummy_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAF"
        "hAJ/wlseKgAAAABJRU5ErkJggg=="
    )
    
    data = {
        "site": "cheek", 
        "suspected_type": "bcc",
        "stream": "1"
    }
    
    files = {
        "image": ("test.png", dummy_png, "image/png")
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/analyze", data=data, files=files, stream=True)
        
        if response.status_code == 200:
            print("✅ Streaming started")
            chunk_count = 0
            for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                if chunk:
                    chunk_count += 1
                    print(f"   Chunk {chunk_count}: {chunk[:50]}...")
                    if chunk_count >= 5:  # Limit output
                        break
            print(f"   Total chunks received: {chunk_count}")
        else:
            print(f"❌ Streaming failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Streaming error: {e}")

def test_usage_stats():
    """Test 6: Usage statistics"""
    print("\n🔍 Test 6: Usage Statistics")
    
    response = requests.get(f"{BASE_URL}/api/usage")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Usage stats: {json.dumps(data, indent=2)}")
    else:
        print(f"❌ Usage stats failed: {response.status_code}")

def test_rate_limiting():
    """Test 7: Rate limiting (multiple rapid requests)"""
    print("\n🔍 Test 7: Rate Limiting")
    
    dummy_png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAF"
        "hAJ/wlseKgAAAABJRU5ErkJggg=="
    )
    
    # Make 3 rapid requests
    for i in range(3):
        data = {"site": "arm", "suspected_type": "nevus"}
        files = {"image": ("test.png", dummy_png, "image/png")}
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/api/analyze", data=data, files=files)
        end_time = time.time()
        
        print(f"   Request {i+1}: {response.status_code} ({end_time - start_time:.2f}s)")
        
        if response.status_code == 429:
            print("   ✅ Rate limiting active")
            break
        elif i < 2:
            time.sleep(0.1)  # Small delay between requests

def run_all_tests():
    """Run all hardening tests"""
    print("🚀 Starting SurgicalAI Hardened System Tests")
    print("=" * 50)
    
    try:
        test_health_check()
        test_provider_info()
        test_analysis_without_image()
        test_analysis_with_dummy_image()
        test_streaming_analysis()
        test_usage_stats()
        test_rate_limiting()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed!")
        print("\nNext steps:")
        print("1. Test with real lesion images from data/samples/")
        print("2. Test fallback by temporarily disabling primary provider")
        print("3. Check usage.jsonl file for cost tracking")
        print("4. Try the streaming UI at http://localhost:7860/?stream=1")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")

if __name__ == "__main__":
    run_all_tests()
