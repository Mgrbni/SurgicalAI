"""Rule-based flap planning."""
from __future__ import annotations
import numpy as np
from ..anatomy.atlas import load_atlas


def design_flap(vertices: np.ndarray, heat: np.ndarray) -> dict:
    atlas = load_atlas()
    lesion_idx = int(np.argmax(heat))
    centroid = vertices[lesion_idx]
    langer = np.array(atlas['langer']['lines'][0])
    vec = np.array(langer[1]) - np.array(langer[0])
    if vec.shape[0] == 2:  # pad to 3-D if atlas is 2-D
        vec = np.append(vec, 0.0)
    pivot = centroid + np.array([0.1,0.0,0.0])
    flap = {
        'centroid': centroid.tolist(),
        'pivot': pivot.tolist(),
        'vector': vec.tolist(),
        'score': float(np.clip(1 - np.linalg.norm(vec), 0, 1))
    }
    return flap
