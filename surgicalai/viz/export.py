"""Export utilities."""
from __future__ import annotations
from pathlib import Path
from PIL import Image
from ..utils.io import ensure_dir


def save_image(img: Image.Image, path: Path):
    ensure_dir(path.parent)
    img.save(path)
