from __future__ import annotations

import os
import numpy as np


def setup_headless() -> None:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("O3D_HEADLESS", "1")


def red_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    colors = np.zeros((len(values), 3))
    colors[:, 0] = values
    return colors
