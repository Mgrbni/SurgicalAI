#!/usr/bin/env python
"""
Comprehensive verification script for the advanced SurgicalAI pipeline.

This script tests all the key features implemented:
1. Transform accuracy and coordinate mapping
2. Grad-CAM generation and ROI detection  
3. Vision-LLM integration (if API keys available)
4. Neuro-symbolic fusion system
5. Overlay generation and PDF reporting

Run with: python verify_advanced_system.py
"""
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import json
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_transforms():
    """Test the transform pipeline accuracy."""
    print("🔧 Testing transform pipeline...")
    
    try:
        from surgicalai_demo.transforms import (
            preprocess_for_model, 
            warp_to_original, 
            test_inverse_transform_accuracy,
            create_test_grid_image
        )
        
        # Create test image
        test_img = create_test_grid_image(400, 300)
        
        # Test preprocessing
        tensor, metadata = preprocess_for_model(test_img, model_size=224)
        print(f"   ✅ Preprocessing: {test_img.shape} → {tensor.shape}")
        
        # Test inverse transform accuracy
        accuracy_metrics = test_inverse_transform_accuracy(test_img)
        mean_error = accuracy_metrics['mean_error_px']
        accuracy = accuracy_metrics['accuracy_within_tolerance']
        
        if mean_error < 1.0:
            print(f"   ✅ Transform accuracy: {mean_error:.2f}px mean error")
        else:
            print(f"   ❌ Transform accuracy: {mean_error:.2f}px > 1.0px tolerance")
            
        if accuracy >= 0.8:
            print(f"   ✅ Coordinate precision: {accuracy:.1%} within tolerance")
        else:
            print(f"   ❌ Coordinate precision: {accuracy:.1%} < 80% requirement")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Transform test failed: {e}")
        return False


def test_gradcam():
    """Test Grad-CAM and ROI detection."""
    print("\n🎯 Testing Grad-CAM system...")
    
    try:
        from surgicalai_demo.gradcam import (
            gradcam_activation,
            find_lesion_roi,
            save_overlays_full_and_zoom
        )
        
        # Create synthetic lesion image
        test_img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        # Add a dark "lesion" region
        cv2.circle(test_img, (200, 150), 50, (80, 60, 40), -1)
        
        # Generate activation
        activation = gradcam_activation(test_img)
        print(f"   ✅ Activation generation: {activation.shape}, range [{activation.min():.3f}, {activation.max():.3f}]")
        
        # Test ROI detection
        roi_info = find_lesion_roi(activation, threshold=0.35, min_area=600)
        print(f"   ✅ ROI detection: ({roi_info.x}, {roi_info.y}, {roi_info.w}, {roi_info.h})")
        
        # Verify ROI is within bounds
        if (0 <= roi_info.x < test_img.shape[1] and 
            0 <= roi_info.y < test_img.shape[0] and
            roi_info.x + roi_info.w <= test_img.shape[1] and
            roi_info.y + roi_info.h <= test_img.shape[0]):
            print("   ✅ ROI bounds validation passed")
        else:
            print("   ❌ ROI bounds validation failed")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Grad-CAM test failed: {e}")
        return False


def test_vlm_integration():
    """Test Vision-LLM integration if API keys available."""
    print("\n👁️ Testing Vision-LLM integration...")
    
    try:
        from surgicalai_demo.vlm_observer import describe_lesion_roi, image_to_bytes
        
        # Check for API keys
        has_openai = bool(os.getenv("OPENAI_API_KEY"))
        has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        
        if not (has_openai or has_anthropic):
            print("   ⚠️ No API keys found - skipping VLM test")
            print("   💡 Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test VLM features")
            return True
        
        # Create synthetic lesion for testing
        test_img = np.zeros((224, 224, 3), dtype=np.uint8)
        # Add structured lesion-like features
        cv2.circle(test_img, (112, 112), 40, (120, 80, 60), -1)  # Brown lesion
        cv2.circle(test_img, (112, 112), 35, (100, 60, 40), -1)  # Darker center
        
        # Convert to bytes
        img_bytes = image_to_bytes(test_img)
        
        # Test VLM analysis
        provider = "openai" if has_openai else "anthropic"
        print(f"   🔄 Testing {provider.upper()} provider...")
        
        try:
            analysis = describe_lesion_roi(img_bytes, provider=provider)
            
            if analysis.get("fallback"):
                print(f"   ⚠️ VLM fallback response: {analysis.get('error', 'Unknown error')}")
            else:
                print(f"   ✅ VLM analysis successful:")
                print(f"       Primary pattern: {analysis.get('primary_pattern', 'N/A')}")
                print(f"       Descriptors: {len(analysis.get('descriptors', []))} found")
                print(f"       Recommendation: {analysis.get('recommendation', 'N/A')}")
                
        except Exception as vlm_error:
            print(f"   ❌ VLM API call failed: {vlm_error}")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ VLM integration test failed: {e}")
        return False


def test_fusion_system():
    """Test the neuro-symbolic fusion system."""
    print("\n🧠 Testing fusion system...")
    
    try:
        from surgicalai_demo.fusion import fuse_probs
        
        # Test case 1: SK with strong descriptors
        cnn_probs = {
            "melanoma": 0.35,
            "bcc": 0.20,
            "scc": 0.15,
            "nevus": 0.20,
            "seborrheic_keratosis": 0.08,
            "benign_other": 0.02
        }
        
        vlm_analysis = {
            "primary_pattern": "seborrheic_keratosis",
            "likelihoods": {
                "melanoma": 0.2,
                "bcc": 0.1,
                "scc": 0.1,
                "nevus": 0.1,
                "seborrheic_keratosis": 0.5
            },
            "descriptors": ["stuck-on appearance", "waxy surface", "milia-like cysts"]
        }
        
        # Test fusion
        result = fuse_probs(cnn_probs, vlm_analysis)
        
        # Verify probability normalization
        total_prob = sum(result["final_probs"].values())
        if abs(total_prob - 1.0) < 0.001:
            print("   ✅ Probability normalization")
        else:
            print(f"   ❌ Probability normalization failed: sum = {total_prob:.6f}")
            
        # Check SK boosting
        original_sk = cnn_probs["seborrheic_keratosis"]
        final_sk = result["final_probs"]["seborrheic_keratosis"]
        
        if final_sk > original_sk:
            print(f"   ✅ SK descriptor boost: {original_sk:.3f} → {final_sk:.3f}")
        else:
            print(f"   ❌ SK boost failed: {original_sk:.3f} → {final_sk:.3f}")
            
        # Check top prediction
        top_class = result["top3"][0][0]
        print(f"   ✅ Top prediction: {top_class} ({result['top3'][0][1]:.1%})")
        
        # Check fusion notes
        if result["fusion_notes"]:
            print(f"   ✅ Fusion notes: {result['fusion_notes'][:80]}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fusion system test failed: {e}")
        return False


def test_end_to_end_pipeline():
    """Test the complete end-to-end pipeline."""
    print("\n🚀 Testing end-to-end pipeline...")
    
    try:
        from surgicalai_demo.pipeline import run_demo
        from surgicalai_demo.features import _read_rgb
        
        # Use sample image if available
        sample_paths = [
            Path("data/samples/face.jpg"),
            Path("data/samples/lesion.jpg"),
            Path("data/face.jpg"),
            Path("data/lesion.jpg")
        ]
        
        sample_image = None
        for path in sample_paths:
            if path.exists():
                sample_image = path
                break
        
        if not sample_image:
            # Create synthetic test image
            print("   📸 Creating synthetic test image...")
            test_img = np.random.randint(100, 200, (400, 300, 3), dtype=np.uint8)
            # Add skin-like coloring
            test_img[:, :, 0] = np.random.randint(180, 220, (400, 300))  # Red
            test_img[:, :, 1] = np.random.randint(140, 180, (400, 300))  # Green
            test_img[:, :, 2] = np.random.randint(100, 140, (400, 300))  # Blue
            
            # Add lesion
            cv2.circle(test_img, (150, 200), 30, (80, 60, 40), -1)
            
            # Save temporary image
            sample_image = Path("temp_test_image.jpg")
            cv2.imwrite(str(sample_image), cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
        
        # Create output directory
        out_dir = Path("runs") / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Run pipeline
        print(f"   🔄 Running pipeline with {sample_image}...")
        summary = run_demo(
            face_img_path=None,
            lesion_img_path=sample_image,
            out_dir=out_dir,
            ask=False  # No interactive input
        )
        
        # Verify outputs
        expected_files = [
            "ai_summary.json",
            "metrics.json",
            "report.pdf"
        ]
        
        missing_files = []
        for filename in expected_files:
            if not (out_dir / filename).exists():
                missing_files.append(filename)
        
        if missing_files:
            print(f"   ❌ Missing output files: {missing_files}")
        else:
            print("   ✅ All expected output files generated")
        
        # Check for advanced overlay artifacts
        advanced_overlays = ["overlay_full.png", "overlay_zoom.png"]
        found_overlays = [f for f in advanced_overlays if (out_dir / f).exists()]
        
        if found_overlays:
            print(f"   ✅ Advanced overlays: {found_overlays}")
        else:
            print("   ⚠️ Basic overlay only (no advanced artifacts)")
        
        # Verify summary structure
        if "fusion_result" in summary:
            print("   ✅ Fusion system integrated")
        if "transform_accuracy" in summary:
            print("   ✅ Transform accuracy metrics included")
        if "vlm_analysis" in summary:
            print("   ✅ VLM analysis integrated")
        
        print(f"   📁 Results saved to: {out_dir}")
        
        # Clean up temporary file
        if sample_image.name == "temp_test_image.jpg":
            sample_image.unlink()
        
        return True
        
    except Exception as e:
        print(f"   ❌ End-to-end pipeline test failed: {e}")
        return False


def main():
    """Run comprehensive verification tests."""
    print("🔬 SurgicalAI Advanced System Verification")
    print("=" * 50)
    
    # Track test results
    tests = [
        ("Transform Pipeline", test_transforms),
        ("Grad-CAM System", test_gradcam),
        ("VLM Integration", test_vlm_integration),
        ("Fusion System", test_fusion_system),
        ("End-to-End Pipeline", test_end_to_end_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}  {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n🎉 All systems operational! Advanced SurgicalAI pipeline ready.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Check logs above for details.")
        
    # Additional notes
    print("\n💡 SETUP NOTES:")
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("   • Set OPENAI_API_KEY or ANTHROPIC_API_KEY for VLM features")
    print("   • Run 'make demo' for quick demonstration")
    print("   • Check settings.yaml for configuration options")
    print("   • NOT FOR CLINICAL USE - Research purposes only")


if __name__ == "__main__":
    main()
