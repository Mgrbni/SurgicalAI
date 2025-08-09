"""Project Grad-CAM back to mesh."""
from __future__ import annotations
import numpy as np


def backproject_heatmap(vertices: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Assign heat values to vertices by simple averaging."""
    val = float(heatmap.mean())/255.0
    return np.full((len(vertices),), val, dtype=float)
