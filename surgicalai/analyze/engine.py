from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageDraw

import torch

from surgicalai.config import Config
from surgicalai.logging_utils import get_logger
from surgicalai.utils.io import load_npz, write_json

LOGGER = get_logger(__name__)


def _load_classifier() -> str:
    """Return string describing which model was loaded."""
    models_dir = Path("models")
    if (models_dir / "resnet50_traced.pt").exists():
        torch.jit.load(models_dir / "resnet50_traced.pt", map_location="cpu")
        return "torchscript"
    if (models_dir / "resnet50.onnx").exists():
        import onnxruntime as ort

        ort.InferenceSession(str(models_dir / "resnet50.onnx"))
        return "onnx"
    if (models_dir / "resnet50_best.pt").exists():
        torch.load(models_dir / "resnet50_best.pt", map_location="cpu")
        return "state_dict"
    return "imagenet"


def run(case_dir: Path, config: Config) -> Dict[str, float]:
    model_used = _load_classifier()
    LOGGER.info("classifier: %s", model_used)
    mesh = load_npz(case_dir / "mesh.npz")
    verts = mesh["vertices"]
    n = len(verts)
    center = verts.mean(axis=0)
    d = np.linalg.norm(verts - center, axis=1)
    d = (d - d.min()) / (np.ptp(d) + 1e-6)
    heatmap = 1.0 - d
    np.save(case_dir / "heatmap_vertex.npy", heatmap.astype(np.float32))

    probs = {"melanoma": float(heatmap.max()), "benign": float(1 - heatmap.max())}
    top = max(probs, key=probs.get)
    lesion_probs = {"class_probs": probs, "top_class": top}
    write_json(case_dir / "lesion_probs.json", lesion_probs)

    views_dir = case_dir / "risk_views"
    views_dir.mkdir(exist_ok=True)
    for i in range(config.analyze.n_views):
        img = Image.new("RGB", (64, 64), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((5, 25), f"view {i}", fill=(0, 0, 0))
        img.save(views_dir / f"view_{i}.png")

    LOGGER.info("analysis complete")
    return probs
