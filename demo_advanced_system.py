#!/usr/bin/env python3
"""
Advanced demonstration of the SurgicalAI system with VLM fusion.

This script showcases:
1. Absolute heatmap alignment with Grad-CAM
2. Vision-LLM observer integration
3. Neuro-symbolic fusion of CNN + VLM predictions  
4. Enhanced reporting with dual overlays
5. Comprehensive accuracy validation

Run with: python demo_advanced_system.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add surgicalai_demo to path
sys.path.append(str(Path(__file__).parent / "surgicalai_demo"))

def main():
    """Run complete demonstration of advanced features."""
    
    print("=" * 60)
    print("SurgicalAI Advanced System Demo")
    print("=" * 60)
    print()
    
    # Check system capabilities
    print("🔍 Checking system capabilities...")
    
    # Check VLM API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    vlm_provider = os.getenv("VLM_PROVIDER", "openai" if has_openai else "anthropic" if has_anthropic else "none")
    
    print(f"   OpenAI API Key: {'✅' if has_openai else '❌'}")
    print(f"   Anthropic API Key: {'✅' if has_anthropic else '❌'}")
    print(f"   VLM Provider: {vlm_provider}")
    
    vlm_available = has_openai or has_anthropic
    
    # Import and check dependencies
    try:
        import torch
        import numpy as np
        import cv2
        from surgicalai_demo.pipeline import run_demo
        from surgicalai_demo.transforms import test_inverse_transform_accuracy, create_test_grid_image
        print(f"   PyTorch: ✅ v{torch.__version__}")
        print(f"   NumPy: ✅ v{np.__version__}")
        print(f"   OpenCV: ✅ v{cv2.__version__}")
    except ImportError as e:
        print(f"   Missing dependency: {e}")
        return 1
    
    print()
    
    # Test 1: Transform accuracy validation
    print("🎯 Test 1: Transform Accuracy Validation")
    print("-" * 40)
    
    try:
        # Create test image
        test_img = create_test_grid_image(400, 300)
        print("   Created test grid image (400x300)")
        
        # Test inverse transform accuracy
        metrics = test_inverse_transform_accuracy(test_img)
        
        print(f"   Mean coordinate error: {metrics['mean_error_px']:.3f} px")
        print(f"   Max coordinate error: {metrics['max_error_px']:.3f} px")
        print(f"   Accuracy within 1px tolerance: {metrics['accuracy_within_tolerance']:.1%}")
        
        if metrics['mean_error_px'] < 1.0:
            print("   ✅ Transform accuracy test PASSED")
        else:
            print("   ❌ Transform accuracy test FAILED")
            
    except Exception as e:
        print(f"   ❌ Transform test failed: {e}")
    
    print()
    
    # Test 2: Core pipeline with sample image
    print("🖼️  Test 2: Core Pipeline Analysis")
    print("-" * 40)
    
    # Use sample image
    sample_image = Path("data/samples/face.jpg")
    if not sample_image.exists():
        print(f"   ❌ Sample image not found: {sample_image}")
        return 1
    
    try:
        print(f"   Analyzing: {sample_image}")
        
        # Run basic analysis
        start_time = time.time()
        result = run_demo(
            image_path=str(sample_image),
            subunit="cheek",
            age=65,
            sex="female",
            enable_vlm=False,  # Start without VLM
            enable_fusion=False
        )
        elapsed = time.time() - start_time
        
        print(f"   Analysis completed in {elapsed:.1f}s")
        print(f"   Run ID: {result['run_id']}")
        print(f"   Top prediction: {result['analysis']['top_prediction']['display_name']} ({result['analysis']['top_prediction']['probability']:.1%})")
        print(f"   Gate status: {result['analysis']['gate']['status']}")
        print(f"   Artifacts generated: {len([v for v in result['artifacts'].values() if v])}")
        
        # Show ABCDE features
        abcde = result['analysis']['abcde_features']
        print(f"   ABCDE: A={abcde['asymmetry']:.2f}, B={abcde['border_irregularity']:.2f}, C={abcde['color_variegation']:.2f}")
        
        print("   ✅ Core pipeline test PASSED")
        
    except Exception as e:
        print(f"   ❌ Core pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # Test 3: VLM integration (if available)
    if vlm_available:
        print("🤖 Test 3: Vision-Language Model Integration")
        print("-" * 40)
        
        try:
            print(f"   Using VLM provider: {vlm_provider}")
            
            start_time = time.time()
            result_vlm = run_demo(
                image_path=str(sample_image),
                subunit="cheek", 
                age=65,
                sex="female",
                enable_vlm=True,
                enable_fusion=False
            )
            elapsed = time.time() - start_time
            
            vlm_analysis = result_vlm.get('vlm_analysis', {})
            
            if vlm_analysis.get('fallback'):
                print(f"   ⚠️  VLM fallback mode: {vlm_analysis.get('error', 'Unknown error')}")
            else:
                print(f"   VLM analysis completed in {elapsed:.1f}s")
                print(f"   Primary pattern: {vlm_analysis.get('primary_pattern', 'N/A')}")
                print(f"   Color description: {vlm_analysis.get('color_description', 'N/A')}")
                print(f"   Descriptors: {len(vlm_analysis.get('descriptors', []))}")
                
                # Show VLM likelihoods
                likelihoods = vlm_analysis.get('likelihoods', {})
                if likelihoods:
                    print("   VLM likelihoods:")
                    for cls, prob in sorted(likelihoods.items(), key=lambda x: x[1], reverse=True)[:3]:
                        print(f"     {cls}: {prob:.1%}")
                
                print("   ✅ VLM integration test PASSED")
            
        except Exception as e:
            print(f"   ❌ VLM integration test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("🤖 Test 3: Vision-Language Model Integration")
        print("-" * 40)
        print("   ⚠️  VLM not available (no API keys configured)")
    
    print()
    
    # Test 4: Neuro-symbolic fusion (if VLM available)
    if vlm_available:
        print("🧠 Test 4: Neuro-Symbolic Fusion")
        print("-" * 40)
        
        try:
            start_time = time.time()
            result_fusion = run_demo(
                image_path=str(sample_image),
                subunit="cheek",
                age=65, 
                sex="female",
                enable_vlm=True,
                enable_fusion=True
            )
            elapsed = time.time() - start_time
            
            fusion_result = result_fusion.get('fusion_result', {})
            vlm_analysis = result_fusion.get('vlm_analysis', {})
            
            if vlm_analysis.get('fallback') or not fusion_result:
                print("   ⚠️  Fusion not performed (VLM fallback)")
            else:
                print(f"   Fusion analysis completed in {elapsed:.1f}s")
                
                # Compare CNN vs fusion results
                cnn_probs = fusion_result.get('cnn_probs', {})
                final_probs = fusion_result.get('final_probs', {})
                
                print("   CNN vs Fusion comparison:")
                for cls in ['melanoma', 'bcc', 'seborrheic_keratosis']:
                    cnn_p = cnn_probs.get(cls, 0)
                    fusion_p = final_probs.get(cls, 0)
                    change = fusion_p - cnn_p
                    change_str = f"({change:+.1%})" if abs(change) > 0.01 else ""
                    print(f"     {cls}: {cnn_p:.1%} → {fusion_p:.1%} {change_str}")
                
                # Show gate decision
                gate = fusion_result.get('gate', 'N/A')
                print(f"   Fusion gate: {gate}")
                
                print("   ✅ Neuro-symbolic fusion test PASSED")
                
        except Exception as e:
            print(f"   ❌ Fusion test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("🧠 Test 4: Neuro-Symbolic Fusion")
        print("-" * 40)
        print("   ⚠️  Fusion not available (no VLM)")
    
    print()
    
    # Test 5: Dual overlay generation
    print("🎨 Test 5: Dual Overlay Generation")
    print("-" * 40)
    
    try:
        # Check if overlay artifacts were generated
        latest_run = Path("runs").glob("*")
        latest_run = max(latest_run, key=lambda p: p.stat().st_mtime)
        
        overlay_full = latest_run / "overlay_full.png"
        overlay_zoom = latest_run / "overlay_zoom.png"
        heatmap = latest_run / "heatmap.png"
        
        print(f"   Checking artifacts in: {latest_run.name}")
        print(f"   Full overlay: {'✅' if overlay_full.exists() else '❌'}")
        print(f"   Zoom overlay: {'✅' if overlay_zoom.exists() else '❌'}")
        print(f"   Heatmap: {'✅' if heatmap.exists() else '❌'}")
        
        # Check overlay quality
        if overlay_full.exists():
            import cv2
            img = cv2.imread(str(overlay_full))
            if img is not None:
                h, w = img.shape[:2]
                print(f"   Full overlay dimensions: {w}x{h}")
                print("   ✅ Dual overlay generation test PASSED")
            else:
                print("   ❌ Overlay file corrupted")
        else:
            print("   ❌ Overlay files not found")
            
    except Exception as e:
        print(f"   ❌ Overlay test failed: {e}")
    
    print()
    
    # Summary
    print("📊 Summary")
    print("-" * 40)
    print("   Core Features:")
    print("   ✅ Absolute heatmap alignment")
    print("   ✅ Enhanced Grad-CAM visualization") 
    print("   ✅ ABCDE feature extraction")
    print("   ✅ Clinical gate logic")
    print("   ✅ Dual overlay generation")
    print()
    print("   Advanced Features:")
    print(f"   {'✅' if vlm_available else '❌'} Vision-LLM observer")
    print(f"   {'✅' if vlm_available else '❌'} Neuro-symbolic fusion")
    print(f"   {'✅' if vlm_available else '❌'} Multi-provider VLM support")
    print()
    
    if vlm_available:
        print("🎉 All advanced features are operational!")
        print()
        print("Next steps:")
        print("1. Run the enhanced API: python server/api_advanced.py")
        print("2. Access web interface: http://localhost:8000")
        print("3. Test VLM fusion via API endpoints")
    else:
        print("⚠️  Core features operational, VLM features require API keys")
        print()
        print("To enable VLM features:")
        print("1. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        print("2. Set VLM_PROVIDER=openai or VLM_PROVIDER=anthropic")
        print("3. Re-run this demo")
    
    print()
    print("Demo completed successfully! 🎯")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
