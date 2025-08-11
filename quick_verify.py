#!/usr/bin/env python3
"""
Quick verification of SurgicalAI system structure and basic imports.
Tests what can be verified without heavy dependencies.
"""

import os
import sys
import json
from pathlib import Path

def test_basic_structure():
    """Test basic project structure."""
    print("📁 Testing project structure...")
    
    required_files = [
        "surgicalai_demo/__init__.py",
        "surgicalai_demo/pipeline.py", 
        "surgicalai_demo/features.py",
        "surgicalai_demo/gradcam.py",
        "surgicalai_demo/transforms.py",
        "surgicalai_demo/vlm_observer.py",
        "surgicalai_demo/fusion.py",
        "settings.yaml",
        "requirements.txt"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            return False
    
    return True

def test_settings():
    """Test settings configuration."""
    print("⚙️  Testing settings...")
    
    try:
        import yaml
        with open("settings.yaml") as f:
            settings = yaml.safe_load(f)
        
        required_sections = ["heatmap", "fusion", "vlm"]
        for section in required_sections:
            if section in settings:
                print(f"   ✅ {section} configuration")
            else:
                print(f"   ❌ {section} configuration missing")
                return False
        
        # Check specific settings
        heatmap = settings.get("heatmap", {})
        if "colormap" in heatmap and "alpha" in heatmap:
            print(f"   ✅ Heatmap parameters configured")
        else:
            print(f"   ❌ Heatmap parameters incomplete")
            
        fusion = settings.get("fusion", {})
        if "cnn_weight" in fusion and "vlm_weight" in fusion:
            print(f"   ✅ Fusion weights configured")
        else:
            print(f"   ❌ Fusion weights incomplete")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Settings error: {e}")
        return False

def test_basic_imports():
    """Test basic imports without heavy dependencies."""
    print("📦 Testing basic imports...")
    
    # Add surgicalai_demo to path
    sys.path.append(str(Path(__file__).parent / "surgicalai_demo"))
    
    basic_modules = [
        ("numpy", "np"),
        ("cv2", "cv2"),
        ("PIL", "Image"),
        ("yaml", "yaml"),
        ("pathlib", "Path")
    ]
    
    for module, alias in basic_modules:
        try:
            if alias == "np":
                import numpy as np
                print(f"   ✅ {module} v{np.__version__}")
            elif alias == "cv2":
                import cv2
                print(f"   ✅ {module} v{cv2.__version__}")
            elif alias == "Image":
                from PIL import Image
                print(f"   ✅ PIL (Pillow)")
            elif alias == "yaml":
                import yaml
                print(f"   ✅ {module}")
            elif alias == "Path":
                from pathlib import Path
                print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            return False
    
    return True

def test_vlm_config():
    """Test VLM configuration."""
    print("🤖 Testing VLM configuration...")
    
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    vlm_provider = os.getenv("VLM_PROVIDER", "none")
    
    print(f"   OpenAI API Key: {'✅' if has_openai else '❌'}")
    print(f"   Anthropic API Key: {'✅' if has_anthropic else '❌'}")
    print(f"   VLM Provider: {vlm_provider}")
    
    if has_openai or has_anthropic:
        print("   ✅ VLM integration available")
        return True
    else:
        print("   ⚠️  VLM integration requires API keys")
        return False

def test_sample_data():
    """Test sample data availability."""
    print("🖼️  Testing sample data...")
    
    sample_files = [
        "data/samples/face.jpg",
        "data/samples/lesion.jpg"
    ]
    
    for sample_file in sample_files:
        if Path(sample_file).exists():
            print(f"   ✅ {sample_file}")
        else:
            print(f"   ❌ {sample_file}")
    
    return any(Path(f).exists() for f in sample_files)

def test_api_structure():
    """Test API structure."""
    print("🌐 Testing API structure...")
    
    api_files = [
        "server/__init__.py",
        "server/api_advanced.py",
        "client/index.html",
        "client/app.js"
    ]
    
    for api_file in api_files:
        if Path(api_file).exists():
            print(f"   ✅ {api_file}")
        else:
            print(f"   ❌ {api_file}")
    
    return any(Path(f).exists() for f in api_files)

def main():
    """Run basic verification tests."""
    print("🔬 SurgicalAI Basic System Verification")
    print("=" * 50)
    print()
    
    tests = [
        ("Project Structure", test_basic_structure),
        ("Settings Configuration", test_settings),
        ("Basic Imports", test_basic_imports),
        ("VLM Configuration", test_vlm_config),
        ("Sample Data", test_sample_data),
        ("API Structure", test_api_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        print()
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    print()
    print("📊 Summary")
    print("-" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print()
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Basic system verification successful!")
        print()
        print("Next steps:")
        print("1. Install remaining dependencies: pip install torch torchvision")
        print("2. Run advanced demo: python demo_advanced_system.py")
        print("3. Start API server: python server/api_advanced.py")
    else:
        print("⚠️  Some tests failed. Check error messages above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
