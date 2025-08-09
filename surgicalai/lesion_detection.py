"""Lightweight lesion detection mock using deterministic data."""
from __future__ import annotations

from typing import Dict, Any

import numpy as np

from .utils import seed_everything

CLASSES = ["melanoma", "nevus", "seb_keratosis"]


def detect(mesh_data: Dict[str, Any], device: str = "cpu", weights: str | None = None) -> Dict[str, Any]:
    """Return fake classification probabilities and a heatmap array."""
    seed_everything(0)
    probs = np.array([0.2, 0.5, 0.3])
    class_probs = {name: float(p) for name, p in zip(CLASSES, probs)}

    coords = mesh_data["coordinates"]
    center = coords.mean(axis=0)
    distances = np.linalg.norm(coords - center, axis=1)
    sigma = distances.std() + 1e-6
    heatmap = np.exp(-0.5 * (distances / sigma) ** 2)

    return {"class_probs": class_probs, "heatmap": heatmap}
