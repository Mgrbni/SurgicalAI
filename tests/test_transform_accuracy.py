"""
Unit tests for transform accuracy and Grad-CAM alignment.

Tests for the requirements:
1. Absolute heatmap alignment (no drift, no scaling bugs)
2. Round-trip coordinate mapping accuracy
3. ROI and overlay size consistency
4. Fusion system behavior with descriptor matching

Run with: python -m pytest tests/test_transform_accuracy.py -v
"""
import pytest
import numpy as np
import cv2
from pathlib import Path

from surgicalai_demo.transforms import (
    preprocess_for_model, 
    warp_to_original, 
    test_inverse_transform_accuracy,
    create_test_grid_image
)
from surgicalai_demo.gradcam import (
    gradcam_activation,
    find_lesion_roi,
    save_overlays_full_and_zoom
)
from surgicalai_demo.fusion import fuse_probs


class TestTransformAccuracy:
    """Test transform accuracy and coordinate mapping."""
    
    def test_round_trip_coordinate_accuracy(self):
        """Test that coordinate transformations are accurate within 1px tolerance."""
        # Create test image
        test_img = create_test_grid_image(400, 300)
        
        # Test transform accuracy
        accuracy_metrics = test_inverse_transform_accuracy(
            test_img, 
            model_size=224, 
            num_test_points=50,
            tolerance_px=1.0
        )
        
        # Verify accuracy requirements
        assert accuracy_metrics['mean_error_px'] < 1.0, f"Mean error {accuracy_metrics['mean_error_px']:.2f}px exceeds 1px tolerance"
        assert accuracy_metrics['accuracy_within_tolerance'] >= 0.8, f"Accuracy {accuracy_metrics['accuracy_within_tolerance']:.1%} below 80%"
        assert accuracy_metrics['match_rate'] >= 0.7, f"Match rate {accuracy_metrics['match_rate']:.1%} below 70%"
    
    def test_transform_metadata_consistency(self):
        """Test that transform metadata correctly describes operations."""
        test_img = create_test_grid_image(512, 384)
        tensor, metadata = preprocess_for_model(test_img, model_size=224)
        
        # Check tensor shape
        assert tensor.shape == (1, 3, 224, 224), f"Expected tensor shape (1,3,224,224), got {tensor.shape}"
        
        # Check metadata consistency
        assert metadata.original_height == 384
        assert metadata.original_width == 512
        assert metadata.model_height == 224
        assert metadata.model_width == 224
        
        # Verify scale factors
        expected_scale = 224 / 384  # Shorter side becomes 224
        assert abs(metadata.scale_factor_y - expected_scale) < 0.001
    
    def test_synthetic_gradcam_shape_consistency(self):
        """Test that synthetic Grad-CAM maintains proper dimensions."""
        test_img = create_test_grid_image(300, 400)
        
        # Generate activation map
        activation = gradcam_activation(test_img)
        
        # Check shape consistency
        assert activation.shape == test_img.shape[:2], f"Activation shape {activation.shape} doesn't match image {test_img.shape[:2]}"
        assert activation.dtype == np.float32, f"Expected float32, got {activation.dtype}"
        assert 0 <= activation.min() <= activation.max() <= 1, f"Activation range [{activation.min():.3f}, {activation.max():.3f}] not in [0,1]"
    
    def test_roi_detection_consistency(self):
        """Test ROI detection produces consistent results."""
        test_img = create_test_grid_image(400, 300)
        activation = gradcam_activation(test_img)
        
        # Find ROI multiple times - should be consistent
        roi1 = find_lesion_roi(activation, threshold=0.35, min_area=600)
        roi2 = find_lesion_roi(activation, threshold=0.35, min_area=600)
        
        # ROI should be identical for same input
        assert roi1.x == roi2.x, "ROI x coordinate inconsistent"
        assert roi1.y == roi2.y, "ROI y coordinate inconsistent"
        assert roi1.w == roi2.w, "ROI width inconsistent"
        assert roi1.h == roi2.h, "ROI height inconsistent"
        
        # ROI should be within image bounds
        assert 0 <= roi1.x < test_img.shape[1], f"ROI x={roi1.x} outside image width {test_img.shape[1]}"
        assert 0 <= roi1.y < test_img.shape[0], f"ROI y={roi1.y} outside image height {test_img.shape[0]}"
        assert roi1.x + roi1.w <= test_img.shape[1], f"ROI extends beyond image width"
        assert roi1.y + roi1.h <= test_img.shape[0], f"ROI extends beyond image height"


class TestOverlayConsistency:
    """Test overlay generation consistency."""
    
    def test_overlay_artifacts_size_consistency(self, tmp_path):
        """Test that overlay and base image have identical dimensions."""
        test_img = create_test_grid_image(512, 384)
        activation = gradcam_activation(test_img)
        
        # Generate overlays
        artifacts = save_overlays_full_and_zoom(
            test_img,
            activation,
            tmp_path,
            "test_class",
            0.75,
            config={"roi_out_size": 256}
        )
        
        # Verify files exist
        assert artifacts["overlay_full"].exists(), "Full overlay not created"
        assert artifacts["overlay_zoom"].exists(), "Zoom overlay not created"
        
        # Check full overlay dimensions match original
        full_overlay = cv2.imread(str(artifacts["overlay_full"]))
        assert full_overlay.shape[:2] == test_img.shape[:2], f"Full overlay size {full_overlay.shape[:2]} != original {test_img.shape[:2]}"
        
        # Check zoom overlay has target size
        zoom_overlay = cv2.imread(str(artifacts["overlay_zoom"]))
        assert zoom_overlay.shape[:2] == (256, 256), f"Zoom overlay size {zoom_overlay.shape[:2]} != target (256,256)"
    
    def test_overlay_pixel_alignment(self, tmp_path):
        """Test that overlays are properly pixel-aligned without drift."""
        # Create image with known features
        test_img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add a bright square that should be detected
        test_img[75:125, 75:125] = [255, 0, 0]  # Red square
        
        activation = gradcam_activation(test_img)
        
        # The activation should peak near the red square
        peak_y, peak_x = np.unravel_index(np.argmax(activation), activation.shape)
        
        # Peak should be roughly in the center where we placed the square
        center_x, center_y = 100, 100
        distance = np.sqrt((peak_x - center_x)**2 + (peak_y - center_y)**2)
        assert distance < 30, f"Activation peak at ({peak_x},{peak_y}) too far from feature center ({center_x},{center_y}), distance={distance:.1f}px"


class TestFusionSystem:
    """Test the neuro-symbolic fusion system."""
    
    def test_sk_descriptor_boost(self):
        """Test that SK descriptors properly boost seborrheic keratosis probability."""
        cnn_probs = {
            "melanoma": 0.4,
            "bcc": 0.2,
            "scc": 0.15,
            "nevus": 0.2,
            "seborrheic_keratosis": 0.04,
            "benign_other": 0.01
        }
        
        vlm_analysis_sk = {
            "primary_pattern": "seborrheic_keratosis",
            "likelihoods": {
                "melanoma": 0.1,
                "bcc": 0.1,
                "scc": 0.1,
                "nevus": 0.1,
                "seborrheic_keratosis": 0.6
            },
            "descriptors": ["stuck-on appearance", "waxy surface", "milia-like cysts"]
        }
        
        result = fuse_probs(cnn_probs, vlm_analysis_sk)
        
        # SK should be boosted due to strong descriptors
        final_sk_prob = result["final_probs"]["seborrheic_keratosis"]
        original_sk_prob = cnn_probs["seborrheic_keratosis"]
        
        assert final_sk_prob > original_sk_prob, f"SK probability not boosted: {final_sk_prob:.3f} <= {original_sk_prob:.3f}"
        
        # Should be in top predictions
        top_class = result["top3"][0][0]
        assert "seborrheic" in top_class.lower() or top_class == "seborrheic_keratosis", f"SK not in top prediction: {top_class}"
        
        # Should have descriptor adjustments
        adjustments = result["descriptor_adjustments"]["adjustments"]
        assert any("SK boosted" in adj for adj in adjustments), f"No SK boost found in adjustments: {adjustments}"
    
    def test_melanoma_descriptor_boost(self):
        """Test that melanoma descriptors boost melanoma probability."""
        cnn_probs = {
            "melanoma": 0.25,
            "bcc": 0.3,
            "scc": 0.2,
            "nevus": 0.2,
            "seborrheic_keratosis": 0.04,
            "benign_other": 0.01
        }
        
        vlm_analysis_mel = {
            "primary_pattern": "melanoma",
            "likelihoods": {
                "melanoma": 0.5,
                "bcc": 0.2,
                "scc": 0.1,
                "nevus": 0.15,
                "seborrheic_keratosis": 0.05
            },
            "descriptors": ["irregular pigment network", "asymmetric pigmentation"]
        }
        
        result = fuse_probs(cnn_probs, vlm_analysis_mel)
        
        # Melanoma should be boosted
        final_mel_prob = result["final_probs"]["melanoma"]
        original_mel_prob = cnn_probs["melanoma"]
        
        assert final_mel_prob > original_mel_prob, f"Melanoma not boosted: {final_mel_prob:.3f} <= {original_mel_prob:.3f}"
        
        # Should have melanoma descriptor adjustments
        adjustments = result["descriptor_adjustments"]["adjustments"]
        assert any("Melanoma boosted" in adj for adj in adjustments), f"No melanoma boost found: {adjustments}"
    
    def test_tie_breaking_logic(self):
        """Test tie-breaking behavior for close predictions."""
        # Create close CNN probabilities
        cnn_probs = {
            "melanoma": 0.32,
            "bcc": 0.30,
            "scc": 0.18,
            "nevus": 0.15,
            "seborrheic_keratosis": 0.04,
            "benign_other": 0.01
        }
        
        vlm_analysis_uncertain = {
            "primary_pattern": "other",
            "likelihoods": {
                "melanoma": 0.3,
                "bcc": 0.35,
                "scc": 0.2,
                "nevus": 0.1,
                "seborrheic_keratosis": 0.05
            },
            "descriptors": ["unclear borders"]
        }
        
        result = fuse_probs(cnn_probs, vlm_analysis_uncertain)
        
        # Should detect uncertainty due to close margins
        gate = result["gate"]
        assert "UNCERTAIN" in gate.upper() or "clinical" in gate.lower(), f"Should detect uncertainty, got: {gate}"
    
    def test_probability_normalization(self):
        """Test that final probabilities sum to 1.0."""
        cnn_probs = {
            "melanoma": 0.4,
            "bcc": 0.3,
            "scc": 0.2,
            "nevus": 0.1
        }
        
        vlm_analysis = {
            "likelihoods": {
                "melanoma": 0.3,
                "bcc": 0.4,
                "scc": 0.2,
                "nevus": 0.1
            },
            "descriptors": []
        }
        
        result = fuse_probs(cnn_probs, vlm_analysis)
        
        # Probabilities should sum to 1.0 (within floating point tolerance)
        total_prob = sum(result["final_probs"].values())
        assert abs(total_prob - 1.0) < 0.001, f"Probabilities don't sum to 1.0: {total_prob:.6f}"
        
        # All probabilities should be non-negative
        for class_name, prob in result["final_probs"].items():
            assert prob >= 0, f"Negative probability for {class_name}: {prob}"


if __name__ == "__main__":
    # Quick test run
    import sys
    
    print("Running transform accuracy tests...")
    
    # Test coordinate accuracy
    test_transform = TestTransformAccuracy()
    try:
        test_transform.test_round_trip_coordinate_accuracy()
        print("✅ Coordinate accuracy test passed")
    except AssertionError as e:
        print(f"❌ Coordinate accuracy test failed: {e}")
    
    # Test fusion system
    test_fusion = TestFusionSystem()
    try:
        test_fusion.test_sk_descriptor_boost()
        print("✅ SK descriptor boost test passed")
    except AssertionError as e:
        print(f"❌ SK descriptor boost test failed: {e}")
    
    try:
        test_fusion.test_probability_normalization()
        print("✅ Probability normalization test passed")
    except AssertionError as e:
        print(f"❌ Probability normalization test failed: {e}")
    
    print("Unit tests completed.")
