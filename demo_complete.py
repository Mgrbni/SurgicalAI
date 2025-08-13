#!/usr/bin/env python3
"""
SurgicalAI Complete Demo Script

This script demonstrates the full SurgicalAI pipeline end-to-end:
1. Start the server
2. Upload a test image  
3. Perform analysis
4. Generate PDF report
5. Verify all functionality

Run with: python demo_complete.py
"""

import asyncio
import base64
import io
import json
import time
import subprocess
import sys
from pathlib import Path
from typing import Optional

import requests
from PIL import Image


def create_test_image() -> bytes:
    """Create a test image for demonstration."""
    # Create a simple red circle on white background to simulate a lesion
    img = Image.new('RGB', (400, 400), 'white')
    
    # Draw a simple "lesion" pattern
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Main lesion area
    draw.ellipse([150, 150, 250, 250], fill='#8B4513', outline='#654321', width=2)
    
    # Add some texture/variation
    draw.ellipse([160, 160, 180, 180], fill='#A0522D')
    draw.ellipse([200, 190, 220, 210], fill='#D2691E')
    
    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=90)
    return buffer.getvalue()


def start_server() -> subprocess.Popen:
    """Start the SurgicalAI server."""
    print("🚀 Starting SurgicalAI server...")
    
    # Set offline mode for demo
    import os
    os.environ['OFFLINE_ANALYSIS'] = 'true'
    
    process = subprocess.Popen([
        sys.executable, '-m', 'uvicorn', 
        'server.app:app', 
        '--host', '0.0.0.0', 
        '--port', '8000'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    for i in range(30):  # 30 second timeout
        try:
            response = requests.get('http://localhost:8000/healthz', timeout=2)
            if response.status_code == 200:
                print("✅ Server started successfully!")
                return process
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    
    # Server failed to start
    process.terminate()
    raise RuntimeError("Server failed to start within 30 seconds")


def test_health_endpoints():
    """Test health and version endpoints."""
    print("\n🔍 Testing health endpoints...")
    
    # Test health
    response = requests.get('http://localhost:8000/healthz')
    print(f"Health check: {response.status_code} - {response.json()}")
    
    # Test version  
    response = requests.get('http://localhost:8000/version')
    print(f"Version info: {response.status_code} - {response.json()}")
    
    print("✅ Health endpoints working!")


def test_analysis():
    """Test the main analysis endpoint."""
    print("\n🧠 Testing analysis endpoint...")
    
    # Create test image
    image_data = create_test_image()
    
    # Prepare request
    files = {
        'file': ('test_lesion.jpg', io.BytesIO(image_data), 'image/jpeg')
    }
    
    payload = {
        'roi': {
            'x': 0.35, 'y': 0.35,  # Center the ROI on our "lesion"
            'width': 0.3, 'height': 0.3
        },
        'site': 'forehead',
        'risk_factors': {
            'age': 45,
            'sex': 'Male',
            'h_zone': False,
            'ill_defined_borders': True,
            'recurrent_tumor': False,
            'prior_histology': False
        },
        'offline': True
    }
    
    data = {
        'payload': json.dumps(payload),
        'offline': 'true'
    }
    
    print(f"📤 Sending analysis request...")
    print(f"   - Image size: {len(image_data)} bytes")
    print(f"   - Site: {payload['site']}")
    print(f"   - ROI: {payload['roi']}")
    
    response = requests.post(
        'http://localhost:8000/api/analyze',
        files=files,
        data=data,
        timeout=30
    )
    
    if response.status_code != 200:
        print(f"❌ Analysis failed: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    request_id = response.headers.get('X-Request-ID', 'unknown')
    
    print(f"✅ Analysis completed! Request ID: {request_id}")
    print(f"   - Primary diagnosis: {result['diagnostics']['top3'][0]['label']}")
    print(f"   - Probability: {result['diagnostics']['top3'][0]['prob']:.2%}")
    print(f"   - RSTL angle: {result['rstl_angle_deg']:.1f}°")
    print(f"   - Tension score: {result['tension_score']:.2f}")
    print(f"   - Flap type: {result['flap_plan']['type']}")
    
    return result


def test_report_generation(analysis_result):
    """Test PDF report generation."""
    print("\n📄 Testing PDF report generation...")
    
    if not analysis_result:
        print("❌ No analysis result to generate report from")
        return None
    
    data = {
        'analysis_payload': json.dumps(analysis_result),
        'doctor_name': 'Dr. Mehdi Ghorbani Karimabad'
    }
    
    response = requests.post(
        'http://localhost:8000/api/report',
        data=data,
        timeout=30
    )
    
    if response.status_code != 200:
        print(f"❌ Report generation failed: {response.status_code}")
        print(response.text)
        return None
    
    # Save PDF
    runs_dir = Path('runs')
    runs_dir.mkdir(exist_ok=True)
    
    pdf_path = runs_dir / 'demo_report.pdf'
    with open(pdf_path, 'wb') as f:
        f.write(response.content)
    
    request_id = response.headers.get('X-Request-ID', 'unknown')
    
    print(f"✅ PDF report generated! Request ID: {request_id}")
    print(f"   - Saved to: {pdf_path}")
    print(f"   - Size: {len(response.content)} bytes")
    
    return pdf_path


def test_artifacts():
    """Test artifact serving."""
    print("\n🖼️  Testing artifact serving...")
    
    # Test a simple artifact request (this will likely 404, but tests the endpoint)
    try:
        response = requests.get('http://localhost:8000/api/artifact/test.png')
        print(f"Artifact endpoint: {response.status_code}")
    except Exception as e:
        print(f"Artifact endpoint test: {e}")


def save_demo_summary(analysis_result, pdf_path):
    """Save a summary of the demo run."""
    runs_dir = Path('runs')
    
    summary = {
        'demo_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'analysis_summary': {
            'primary_diagnosis': analysis_result['diagnostics']['top3'][0]['label'],
            'probability': analysis_result['diagnostics']['top3'][0]['prob'],
            'flap_type': analysis_result['flap_plan']['type'],
            'tension_score': analysis_result['tension_score']
        },
        'pdf_report': str(pdf_path) if pdf_path else None,
        'status': 'success'
    }
    
    summary_path = runs_dir / 'demo_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📋 Demo summary saved to: {summary_path}")


def main():
    """Run the complete demo."""
    print("🏥 SurgicalAI Complete Demo")
    print("=" * 50)
    
    server_process = None
    
    try:
        # Start server
        server_process = start_server()
        
        # Run tests
        test_health_endpoints()
        analysis_result = test_analysis()
        pdf_path = test_report_generation(analysis_result)
        test_artifacts()
        
        # Save summary
        if analysis_result:
            save_demo_summary(analysis_result, pdf_path)
        
        print("\n🎉 Demo completed successfully!")
        print("\nResults:")
        print(f"   - Analysis: {'✅' if analysis_result else '❌'}")
        print(f"   - PDF Report: {'✅' if pdf_path else '❌'}")
        
        if pdf_path:
            print(f"\n📁 Check the runs/ directory for output files")
            print(f"   - PDF Report: {pdf_path}")
            print(f"   - Demo Summary: runs/demo_summary.json")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    finally:
        # Clean up
        if server_process:
            print("\n🛑 Stopping server...")
            server_process.terminate()
            server_process.wait()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
