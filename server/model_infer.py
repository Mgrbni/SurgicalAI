"""Model inference & probability calibration utilities (demo).

This is a lightweight multi-class inference stub replacing earlier random /
binary-like logic. It produces:
- Calibrated probabilities over SETTINGS.class_names using softmax(logits / T)
- Optional per-class bias and weights (log-prior adjustment)
- Returns full probability dict + top-k list
- Generates high-quality ROI-centered Grad-CAM heatmaps

NOT FOR CLINICAL USE.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from .settings import SETTINGS

CLASS2IDX = {c: i for i, c in enumerate(SETTINGS.class_names)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}

@dataclass
class InferenceResult:
    probs: Dict[str, float]
    topk: List[Tuple[str, float]]
    logits: List[float]
    calibrated_logits: List[float]
    temperature: float
    class_weights: Dict[str, float]
    class_bias: Dict[str, float]

# --- Utility softmax with temperature ---
def _softmax(logits: List[float], temp: float) -> List[float]:
    if temp <= 0:
        temp = 1.0
    mx = max(logits)
    exps = [math.exp((z - mx) / temp) for z in logits]
    s = sum(exps)
    if s <= 0:
        return [1.0/len(logits)] * len(logits)
    return [e / s for e in exps]

# --- Placeholder backbone producing raw logits ---
def _mock_backbone(image_path: Path) -> List[float]:
    # Deterministic seed for reproducibility per image (hash of bytes length + name)
    try:
        stat = image_path.stat()
        seed = stat.st_size ^ len(image_path.name)
    except Exception:
        seed = 42
    random.seed(seed)
    n = len(SETTINGS.class_names)
    # Generate moderately separated random logits
    return [random.uniform(-1.2, 2.2) for _ in range(n)]

# --- Enhanced Grad-CAM with ROI centering ---
def _gradcam_enhanced(base_img: Image.Image, out_path: Path, label: str, run_dir: Path):
    """Generate ROI-centered, high-contrast Grad-CAM heatmap with overlays."""
    try:
        # Import the new Grad-CAM module
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from surgicalai_demo.gradcam import (
            gradcam_activation, save_overlays_full_and_zoom, create_unified_heatmap
        )
        
        # Convert PIL to numpy
        img_array = np.array(base_img)
        
        # Generate activation map
        activation = gradcam_activation(img_array)
        
        # Get top class info (simplified for demo)
        top_class = label
        probability = 0.75 + (hash(label) % 100) / 400.0  # Deterministic fake probability
        
        # Generate both overlays
        overlay_paths = save_overlays_full_and_zoom(
            img_array, activation, run_dir, top_class, probability
        )
        
        # Create unified heatmap
        heatmap_path = create_unified_heatmap(img_array, activation, run_dir)
        
        # Copy to expected output path for backwards compatibility
        if out_path != heatmap_path:
            import shutil
            shutil.copy2(heatmap_path, out_path)
            
    except Exception as e:
        # Fallback to simple placeholder if enhanced version fails
        _gradcam_placeholder_simple(base_img, out_path, label)

def _gradcam_placeholder_simple(base_img: Image.Image, out_path: Path, label: str):
    """Simple fallback heatmap generator."""
    w, h = base_img.size
    overlay = Image.new("RGBA", base_img.size, (255,0,0,0))
    draw = ImageDraw.Draw(overlay)
    # simple concentric ellipse pattern with alpha tied to hash(label)
    rnd = (hash(label) & 0xFF) / 255.0
    for i, r in enumerate(range(0, min(w,h)//2, max(8, min(w,h)//10))):
        alpha = int(40 + 90 * (1 - i/6.0))
        draw.ellipse((r, r, w-r, h-r), outline=(255, int(180*rnd), 0, alpha))
    blended = Image.alpha_composite(base_img.convert("RGBA"), overlay)
    blended.save(out_path)

# --- Public inference function ---
def run_inference(image_path: Path, run_dir: Path, topk: int = 3) -> InferenceResult:
    logits = _mock_backbone(image_path)
    # Apply weights/bias before temperature scaling: logit' = (logit * weight) + bias
    adjusted = []
    for i, z in enumerate(logits):
        lbl = IDX2CLASS[i]
        weight = float(SETTINGS.class_weight.get(lbl, 1.0))
        bias = float(SETTINGS.class_bias.get(lbl, 0.0))
        adjusted.append(z * weight + bias)
    probs_list = _softmax(adjusted, SETTINGS.temperature)

    # Build probability dict with clipping + renormalization
    probs = {IDX2CLASS[i]: max(0.001, min(0.999, p)) for i, p in enumerate(probs_list)}
    zsum = sum(probs.values())
    probs = {k: v / zsum for k, v in probs.items()}

    # Top-k sorted
    topk_list = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:topk]

    # Generate enhanced Grad-CAM heatmaps for top-k
    try:
        img = Image.open(image_path).convert("RGB")
        for label, _ in topk_list:
            heatmap_path = run_dir / f"heatmap_{label}.png"
            if not heatmap_path.exists():
                _gradcam_enhanced(img, heatmap_path, label, run_dir)
    except Exception:
        # Fallback to simple version
        try:
            img = Image.open(image_path).convert("RGB")
            for label, _ in topk_list:
                heatmap_path = run_dir / f"heatmap_{label}.png"
                if not heatmap_path.exists():
                    _gradcam_placeholder_simple(img, heatmap_path, label)
        except Exception:
            pass

    return InferenceResult(
        probs=probs,
        topk=topk_list,
        logits=logits,
        calibrated_logits=adjusted,
        temperature=SETTINGS.temperature,
        class_weights=SETTINGS.class_weight,
        class_bias=SETTINGS.class_bias,
    )
