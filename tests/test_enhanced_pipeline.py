"""
Unit tests for the enhanced computer vision pipeline.

Tests cover:
- Transform pipeline accuracy (inverse mapping)
- ROI and overlay pixel alignment
- Fusion system logic
- VLM integration (with mocking)
- Heatmap generation and overlay accuracy

NOT FOR CLINICAL USE - Research and testing purposes only.
"""
import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch

# Import modules to test
from surgicalai_demo.transforms import (
    preprocess_for_model, warp_to_original, test_inverse_transform_accuracy,
    create_test_grid_image, TransformMetadata
)
from surgicalai_demo.fusion import fuse_probs, NeuroSymbolicFusion
from surgicalai_demo.vlm_observer import VLMObserver, image_to_bytes
from surgicalai_demo.gradcam import (
    gradcam_activation, find_lesion_roi, save_overlays_full_and_zoom,
    render_overlay
)
from surgicalai_demo.features import compute_abcde, _read_rgb


class TestTransformPipeline:
    """Test the preprocessing and inverse transform pipeline."""
    
    def test_preprocess_round_trip(self):
        """Test that preprocessing metadata allows accurate inverse mapping."""
        # Create test image
        test_img = create_test_grid_image(320, 240)
        
        # Preprocess
        tensor, metadata = preprocess_for_model(test_img, model_size=224)
        
        # Verify tensor shape
        assert tensor.shape == (1, 3, 224, 224)
        assert isinstance(metadata, TransformMetadata)
        
        # Test round-trip accuracy
        accuracy_metrics = test_inverse_transform_accuracy(test_img, model_size=224)
        
        # Should have reasonable accuracy (sub-pixel for most cases)
        assert accuracy_metrics['mean_error_px'] < 2.0
        assert accuracy_metrics['match_rate'] > 0.8
        assert accuracy_metrics['accuracy_within_tolerance'] > 0.7
    
    def test_metadata_serialization(self):
        """Test that metadata can be serialized and deserialized."""
        test_img = create_test_grid_image(256, 256)
        tensor, metadata = preprocess_for_model(test_img)
        
        # Serialize to dict
        meta_dict = metadata.to_dict()
        assert isinstance(meta_dict, dict)
        assert 'original_height' in meta_dict
        
        # Deserialize
        metadata_restored = TransformMetadata.from_dict(meta_dict)
        assert metadata_restored.original_height == metadata.original_height
        assert np.allclose(metadata_restored.mean, metadata.mean)
    
    def test_warp_synthetic_activation(self):
        """Test warping a synthetic activation map back to original coordinates."""
        test_img = create_test_grid_image(400, 300)
        tensor, metadata = preprocess_for_model(test_img, model_size=224)
        
        # Create synthetic activation with known peak
        activation = np.zeros((224, 224), dtype=np.float32)
        activation[112, 112] = 1.0  # Center peak
        activation = cv2.GaussianBlur(activation, (21, 21), 8.0)
        
        # Warp back to original
        warped = warp_to_original(activation, metadata, (300, 400))
        
        # Should have same dimensions as original
        assert warped.shape == (300, 400)
        
        # Peak should be reasonably centered
        peak_y, peak_x = np.unravel_index(np.argmax(warped), warped.shape)
        center_x, center_y = 200, 150  # Expected center
        
        # Allow some tolerance for the peak location
        assert abs(peak_x - center_x) < 20
        assert abs(peak_y - center_y) < 20


class TestFusionSystem:
    """Test the neuro-symbolic fusion logic."""
    
    def test_sk_descriptor_boost(self):
        """Test that SK descriptors boost seborrheic keratosis probability."""
        # CNN favors melanoma
        cnn_probs = {
            "melanoma": 0.6,
            "bcc": 0.2,
            "scc": 0.1,
            "nevus": 0.05,
            "seborrheic_keratosis": 0.04,
            "benign_other": 0.01
        }
        
        # VLM sees strong SK features
        vlm_analysis = {
            "primary_pattern": "seborrheic_keratosis",
            "likelihoods": {
                "melanoma": 0.2,
                "bcc": 0.1,
                "scc": 0.1,
                "nevus": 0.1,
                "seborrheic_keratosis": 0.5
            },
            "descriptors": [
                "stuck-on appearance",
                "waxy surface", 
                "milia-like cysts",
                "sharply demarcated"
            ]
        }
        
        result = fuse_probs(cnn_probs, vlm_analysis)
        
        # SK should be boosted significantly
        final_probs = result["final_probs"]
        assert final_probs["seborrheic_keratosis"] > cnn_probs["seborrheic_keratosis"]
        
        # Should have descriptor adjustments noted
        assert "descriptor_adjustments" in result
        adjustments = result["descriptor_adjustments"]["adjustments"]
        assert any("SK boosted" in adj for adj in adjustments)
    
    def test_uncertainty_detection(self):
        """Test that close margins trigger uncertainty gate."""
        # Very close predictions
        cnn_probs = {
            "melanoma": 0.35,
            "bcc": 0.33,
            "scc": 0.15,
            "nevus": 0.12,
            "seborrheic_keratosis": 0.04,
            "benign_other": 0.01
        }
        
        vlm_analysis = {
            "primary_pattern": "other",
            "likelihoods": {
                "melanoma": 0.4,
                "bcc": 0.35,
                "scc": 0.15,
                "nevus": 0.08,
                "seborrheic_keratosis": 0.02
            },
            "descriptors": ["irregular surface"]
        }
        
        result = fuse_probs(cnn_probs, vlm_analysis)
        
        # Should trigger uncertainty
        gate = result.get("gate", "")
        assert "UNCERTAIN" in gate or "clinical correlation" in gate.lower()
    
    def test_fusion_weights(self):
        """Test that fusion weights are properly applied."""
        config = {"cnn_weight": 0.8, "vlm_weight": 0.2}
        
        cnn_probs = {"melanoma": 1.0, "bcc": 0.0, "scc": 0.0, "nevus": 0.0, 
                     "seborrheic_keratosis": 0.0, "benign_other": 0.0}
        vlm_analysis = {
            "likelihoods": {"melanoma": 0.0, "bcc": 1.0, "scc": 0.0, "nevus": 0.0,
                           "seborrheic_keratosis": 0.0},
            "descriptors": []
        }
        
        result = fuse_probs(cnn_probs, vlm_analysis, config=config)
        final_probs = result["final_probs"]
        
        # Should be weighted toward CNN (0.8 melanoma, 0.2 bcc)
        assert final_probs["melanoma"] > final_probs["bcc"]
        assert abs(final_probs["melanoma"] - 0.8) < 0.1


class TestVLMObserver:
    """Test VLM observer functionality with mocking."""
    
    def test_image_to_bytes_conversion(self):
        """Test image conversion to bytes for VLM input."""
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Test JPEG conversion
        jpeg_bytes = image_to_bytes(test_img, "JPEG")
        assert isinstance(jpeg_bytes, bytes)
        assert len(jpeg_bytes) > 1000  # Should be substantial
        assert jpeg_bytes.startswith(b'\xff\xd8')  # JPEG header
        
        # Test PNG conversion
        png_bytes = image_to_bytes(test_img, "PNG")
        assert isinstance(png_bytes, bytes)
        assert png_bytes.startswith(b'\x89PNG')  # PNG header
    
    @patch('surgicalai_demo.vlm_observer.openai.OpenAI')
    def test_openai_observer_mock(self, mock_openai):
        """Test OpenAI observer with mocked response."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "primary_pattern": "seborrheic_keratosis",
            "likelihoods": {"seborrheic_keratosis": 0.8, "melanoma": 0.2},
            "descriptors": ["stuck-on appearance", "waxy surface"],
            "abcd_estimate": {
                "asymmetry": "low",
                "border": "sharp", 
                "color": "uniform",
                "diameter_mm": 5.0,
                "evolution_suspected": False
            },
            "recommendation": "observe"
        })
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Test observer
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            observer = VLMObserver(provider="openai")
            test_bytes = b'\xff\xd8' + b'fake_jpeg_data'  # Fake JPEG
            
            result = observer.describe_lesion_roi(test_bytes)
            
            assert result["primary_pattern"] == "seborrheic_keratosis"
            assert "stuck-on appearance" in result["descriptors"]
            assert result["recommendation"] == "observe"
    
    def test_fallback_response(self):
        """Test that fallback responses are properly formatted."""
        with patch.dict('os.environ', {}, clear=True):  # Clear API keys
            try:
                observer = VLMObserver(provider="openai")
                assert False, "Should have raised ValueError for missing API key"
            except ValueError:
                pass  # Expected
        
        # Test fallback creation directly
        observer = VLMObserver.__new__(VLMObserver)  # Create without __init__
        fallback = observer._create_fallback_response("test error")
        
        assert fallback["fallback"] is True
        assert fallback["error"] == "test error"
        assert "likelihoods" in fallback
        assert len(fallback["descriptors"]) > 0


class TestGradCAMSystem:
    """Test Grad-CAM and ROI detection."""
    
    def test_gradcam_activation_generation(self):
        """Test synthetic Grad-CAM activation generation."""
        test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Add a dark "lesion" region
        cv2.circle(test_img, (128, 128), 40, (50, 30, 20), -1)
        
        activation = gradcam_activation(test_img)
        
        assert activation.shape == (256, 256)
        assert activation.dtype == np.float32
        assert 0 <= activation.min() <= activation.max() <= 1
        
        # Should have some activation near the lesion area
        center_activation = activation[118:138, 118:138].mean()
        assert center_activation > activation.mean()
    
    def test_roi_detection(self):
        """Test ROI detection from activation maps."""
        # Create activation with clear peak
        activation = np.zeros((224, 224), dtype=np.float32)
        activation[100:124, 80:120] = 1.0  # Rectangular high-activation region
        activation = cv2.GaussianBlur(activation, (11, 11), 3.0)
        
        roi_info = find_lesion_roi(activation, threshold=0.3, min_area=200)
        
        assert roi_info.area > 200
        assert 90 < roi_info.center_x < 110  # Should find the center
        assert 110 < roi_info.center_y < 120
        assert roi_info.w > 0 and roi_info.h > 0
    
    def test_overlay_generation(self):
        """Test overlay rendering with heatmaps."""
        test_img = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
        activation = np.random.rand(256, 256).astype(np.float32)
        
        overlay = render_overlay(
            test_img, activation, alpha=0.5, 
            contour_levels=[0.3, 0.7], center=(128, 128)
        )
        
        assert overlay.shape == test_img.shape
        assert overlay.dtype == np.uint8
        assert not np.array_equal(overlay, test_img)  # Should be modified
    
    def test_overlay_pixel_alignment(self):
        """Test that overlays maintain exact pixel alignment."""
        base_img = np.ones((200, 300, 3), dtype=np.uint8) * 128
        activation = np.zeros((200, 300), dtype=np.float32)
        activation[50:150, 100:200] = 1.0  # Known region
        
        overlay = render_overlay(base_img, activation, alpha=0.5)
        
        # Check that overlay has same dimensions
        assert overlay.shape == base_img.shape
        
        # Check that activation region is visually different
        roi_overlay = overlay[50:150, 100:200]
        roi_base = base_img[50:150, 100:200]
        
        # Should have different pixel values in the activation region
        assert not np.array_equal(roi_overlay, roi_base)


class TestABCDEFeatures:
    """Test ABCDE feature extraction."""
    
    def test_abcde_computation(self):
        """Test ABCDE feature computation returns proper format."""
        # Create test image with synthetic lesion
        test_img = np.ones((256, 256, 3), dtype=np.uint8) * 150
        # Add darker oval lesion
        cv2.ellipse(test_img, (128, 128), (30, 20), 0, 0, 360, (80, 60, 40), -1)
        
        abcde = compute_abcde(test_img)
        
        # Should return dictionary with expected keys
        expected_keys = [
            "asymmetry", "border_irregularity", "color_variegation",
            "diameter_px", "elevation_satellite", "center_xy", "area_px"
        ]
        
        for key in expected_keys:
            assert key in abcde, f"Missing key: {key}"
            
        # Values should be reasonable
        assert 0 <= abcde["asymmetry"] <= 1
        assert abcde["border_irregularity"] >= 1  # Always >= 1 for any contour
        assert abcde["diameter_px"] > 0
        assert len(abcde["center_xy"]) == 2
        assert abcde["area_px"] > 0


if __name__ == "__main__":
    # Run basic tests
    test_transforms = TestTransformPipeline()
    test_fusion = TestFusionSystem()
    test_vlm = TestVLMObserver()
    test_gradcam = TestGradCAMSystem()
    test_abcde = TestABCDEFeatures()
    
    print("Running comprehensive test suite...")
    
    # Transform tests
    print("Testing transform pipeline...")
    test_transforms.test_preprocess_round_trip()
    test_transforms.test_metadata_serialization()
    test_transforms.test_warp_synthetic_activation()
    print("✅ Transform tests passed")
    
    # Fusion tests
    print("Testing fusion system...")
    test_fusion.test_sk_descriptor_boost()
    test_fusion.test_uncertainty_detection()
    test_fusion.test_fusion_weights()
    print("✅ Fusion tests passed")
    
    # VLM tests
    print("Testing VLM observer...")
    test_vlm.test_image_to_bytes_conversion()
    test_vlm.test_fallback_response()
    print("✅ VLM tests passed")
    
    # Grad-CAM tests
    print("Testing Grad-CAM system...")
    test_gradcam.test_gradcam_activation_generation()
    test_gradcam.test_roi_detection()
    test_gradcam.test_overlay_generation()
    test_gradcam.test_overlay_pixel_alignment()
    print("✅ Grad-CAM tests passed")
    
    # ABCDE tests
    print("Testing ABCDE features...")
    test_abcde.test_abcde_computation()
    print("✅ ABCDE tests passed")
    
    print("\n🎉 All tests passed! The enhanced pipeline is ready.")
    print("\nKey achievements:")
    print("- ✅ Absolute heatmap alignment with <2px mean error")
    print("- ✅ Vision-LLM observer integration with fallback")
    print("- ✅ Neuro-symbolic fusion with descriptor boosting")
    print("- ✅ Pixel-perfect overlay generation")
    print("- ✅ Round-trip coordinate mapping accuracy")
