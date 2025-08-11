from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2
import pytest

from surgicalai_demo.transforms import preprocess_for_model, warp_to_original, TransformMetadata, create_test_grid_image
from surgicalai_demo.fusion import fuse_probs
from surgicalai_demo.gradcam import save_overlays_full_and_zoom, gradcam_activation


def test_warp_round_trip_mae_under_1px():
    img = create_test_grid_image(320, 260)
    tensor, meta = preprocess_for_model(img, model_size=224)
    # Synthetic activation in model space (identity grid)
    grid = np.linspace(0, 1, 224, dtype=np.float32)
    activation_model = np.outer(grid, grid)
    recovered = warp_to_original(activation_model, meta, (img.shape[0], img.shape[1]))
    # Down/up sample forward to original for reference
    forward = cv2.resize(activation_model, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    mae = float(np.mean(np.abs(recovered - forward))) * max(img.shape[:2])
    assert mae < 1.0, f"Mean absolute alignment error {mae:.3f}px exceeds 1px tolerance"


def test_overlay_sizes_match(tmp_path: Path):
    # Create synthetic image & activation
    img = create_test_grid_image(400, 300)
    activation = gradcam_activation(img)
    paths = save_overlays_full_and_zoom(img, activation, tmp_path, top_class="seborrheic_keratosis", probability=0.5)
    full = cv2.imread(str(paths["overlay_full"]))
    zoom = cv2.imread(str(paths["overlay_zoom"]))
    assert full is not None and zoom is not None
    # Full overlay must match original size
    assert full.shape[1] == img.shape[1] and full.shape[0] == img.shape[0]
    # Zoom overlay is square (configured roi_out_size)
    assert zoom.shape[0] == zoom.shape[1]


def test_fusion_sk_descriptor_boost():
    cnn = {"melanoma":0.35,"bcc":0.2,"scc":0.1,"nevus":0.2,"seborrheic_keratosis":0.1,"benign_other":0.05}
    vlm = {
        "primary_pattern":"seborrheic_keratosis",
        "likelihoods":{"melanoma":0.2,"bcc":0.1,"scc":0.1,"nevus":0.1,"seborrheic_keratosis":0.5},
        "descriptors":["stuck-on appearance","waxy surface","sharply demarcated"]
    }
    fused = fuse_probs(cnn, vlm, config={"cnn_weight":0.7,"vlm_weight":0.3,"sk_descriptor_bonus":0.15})
    top1 = fused["top3"][0][0]
    assert top1 in ("seborrheic_keratosis","melanoma","nevus")  # allow variation but expect SK boost
    # Ensure SK probability increased relative to raw CNN
    assert fused["final_probs"]["seborrheic_keratosis"] > cnn["seborrheic_keratosis"], "SK not boosted"
