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


def get_offscreen_renderer(width: int, height: int):
    """Return an Open3D offscreen renderer that works without a GUI."""
    import open3d as o3d

    setup_headless()
    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    except Exception:  # pragma: no cover - fallback
        os.environ["OPEN3D_CPU_RENDERING"] = "1"
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    return renderer
