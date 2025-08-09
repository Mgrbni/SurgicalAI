"""Landmark detection heuristics."""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
from PIL import Image
from ..utils.io import load_json
from ..config import CONFIG


def detect_landmarks(render: np.ndarray | None = None, override: Path | None = None) -> Dict[str, np.ndarray]:
    """Return 3D landmark positions in normalized coordinates."""
    base = CONFIG.data / 'samples'
    override = override or (base / 'manual_landmarks.json')
    if override.exists():
        data = load_json(override)
        return {k: np.array(v, dtype=float) for k, v in data.items()}
    # fallback simple positions
    return {
        'left_eye': np.array([0.3,0.6,0.5]),
        'right_eye': np.array([0.7,0.6,0.5]),
        'nose_tip': np.array([0.5,0.5,0.6]),
        'mouth_left': np.array([0.4,0.3,0.4]),
        'mouth_right': np.array([0.6,0.3,0.4])
    }
