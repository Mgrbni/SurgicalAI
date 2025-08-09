"""Simple strain estimator."""
from __future__ import annotations
import numpy as np
from ..config import CONFIG


def estimate_strain(vertices: np.ndarray, faces: np.ndarray, flap_vec: np.ndarray) -> float:
    """Estimate average displacement magnitude.

    ``flap_vec`` may be a 2-D vector depending on the planning stage.
    Pad to 3-D for addition with the vertex array.
    """

    if flap_vec.shape[0] < 3:
        flap_vec = np.pad(flap_vec, (0, 3 - flap_vec.shape[0]))
    moved = vertices + flap_vec
    strain = np.linalg.norm(moved - vertices, axis=1).mean()
    return float(strain)


def validate_strain(strain: float) -> dict:
    level = 'OK' if strain < CONFIG.strain_thresh else 'HIGH'
    return {'strain': float(strain), 'level': level}
