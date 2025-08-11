#!/usr/bin/env python3
"""
Simple test to verify the basic system is working.
"""
import json
import os
import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from server.settings import SETTINGS
        print(f"✅ Settings loaded: provider={SETTINGS.provider}")
    except Exception as e:
        print(f"❌ Settings import failed: {e}")
        return False
    
    try:
        from server.schemas import AnalysisOutput
        print("✅ Schemas imported")
    except Exception as e:
        print(f"❌ Schemas import failed: {e}")
        return False
    
    try:
        from server.usage import log_usage
        print("✅ Usage logging imported")
    except Exception as e:
        print(f"❌ Usage import failed: {e}")
        return False
    
    return True

def test_env_config():
    """Test environment configuration"""
    print("\n🔍 Testing environment configuration...")
    
    # Check .env file
    env_file = project_root / ".env"
    if env_file.exists():
        print(f"✅ .env file found: {env_file}")
        
        with open(env_file) as f:
            content = f.read()
            
        if "OPENAI_API_KEY" in content:
            print("✅ OpenAI API key configured")
        if "ANTHROPIC_API_KEY" in content:
            print("✅ Anthropic API key configured")
        if "PROVIDER=" in content:
            provider_line = [line for line in content.split('\n') if line.startswith('PROVIDER=')]
            if provider_line:
                print(f"✅ Provider set: {provider_line[0]}")
    else:
        print("❌ .env file not found")

def test_schemas():
    """Test schema validation"""
    print("\n🔍 Testing schema validation...")
    
    try:
        from server.schemas import AnalysisOutput
        
        # Test valid data
        test_data = {
            "diagnosis_probs": [
                {"condition": "seborrheic_keratosis", "probability": 0.7},
                {"condition": "basal_cell_carcinoma", "probability": 0.2},
                {"condition": "melanoma", "probability": 0.1}
            ],
            "primary_dx": "seborrheic_keratosis",
            "warnings": ["Monitor for changes"],
            "citations": ["Dermatology textbook, Chapter 5"]
        }
        
        analysis = AnalysisOutput(**test_data)
        print("✅ Schema validation works")
        print(f"   Primary diagnosis: {analysis.primary_dx}")
        print(f"   Probabilities: {len(analysis.diagnosis_probs)} conditions")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        return False

def test_file_structure():
    """Test file structure"""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        "server/__init__.py",
        "server/settings.py", 
        "server/schemas.py",
        "server/usage.py",
        "server/llm/__init__.py",
        "server/llm/base.py",
        "requirements.txt",
        ".env"
    ]
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")

def main():
    """Run all basic tests"""
    print("🚀 Basic System Validation")
    print("=" * 40)
    
    # Change to project directory
    os.chdir(project_root)
    
    all_good = True
    
    if not test_imports():
        all_good = False
    
    test_env_config()
    
    if not test_schemas():
        all_good = False
    
    test_file_structure()
    
    print("\n" + "=" * 40)
    if all_good:
        print("✅ Basic validation passed!")
        print("\nNext: Start server with:")
        print("python -m uvicorn server.server:app --host 0.0.0.0 --port 7860")
    else:
        print("❌ Some issues found - check imports and dependencies")

if __name__ == "__main__":
    main()
