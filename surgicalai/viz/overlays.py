"""Visualization helpers."""
from __future__ import annotations
import numpy as np
from PIL import Image


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    heat = Image.fromarray(heatmap).resize(image.size)
    heat = heat.convert('RGBA')
    base = image.convert('RGBA')
    blended = Image.blend(base, heat, alpha=0.5)
    return blended
