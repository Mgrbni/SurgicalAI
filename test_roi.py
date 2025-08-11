#!/usr/bin/env python3
"""Test ROI-centered overlay generation"""

import numpy as np
from surgicalai_demo.gradcam import find_lesion_roi, save_overlays_full_and_zoom
from PIL import Image

def test_roi_system():
    print("🧪 Testing ROI-centered heatmap system...")
    
    # Load test image
    img_path = 'data/samples/lesion.jpg'
    try:
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        print(f"✓ Loaded image: {image_np.shape}")
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return False

    # Create a mock activation (simulate model output)
    h, w = image_np.shape[:2]
    activation = np.random.rand(h//8, w//8) * 0.5  # Lower resolution activation
    
    # Add high activation in center to simulate lesion
    cy, cx = activation.shape[0]//2, activation.shape[1]//2
    activation[cy-5:cy+5, cx-5:cx+5] = 0.9
    
    print(f"✓ Created mock activation: {activation.shape}")

    # Test ROI detection
    try:
        roi_bbox = find_lesion_roi(activation, threshold=0.3, min_area_px=100)
        print(f"✓ ROI detection: {roi_bbox}")
    except Exception as e:
        print(f"✗ ROI detection failed: {e}")
        return False

    # Test overlay generation
    try:
        save_overlays_full_and_zoom(
            image_np, activation, 
            'test_overlay_full.png', 'test_overlay_zoom.png',
            roi_bbox=roi_bbox
        )
        print('✓ Overlay generation successful')
        print('✓ Generated: test_overlay_full.png, test_overlay_zoom.png')
        return True
    except Exception as e:
        print(f'✗ Overlay generation failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_roi_system()
    if success:
        print("\n🎉 ROI system test PASSED!")
    else:
        print("\n❌ ROI system test FAILED!")
