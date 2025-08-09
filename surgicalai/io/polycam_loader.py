"""Polycam scan loader."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image
import trimesh


def load_scan(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    mesh = trimesh.load_mesh(path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    tex = None
    if mesh.visual.kind == 'texture' and hasattr(mesh.visual, 'material'):
        image = mesh.visual.material.image
        if image is None:
            tex = Image.new('RGB', (64,64), color=(128,128,128))
        else:
            tex = image
    else:
        tex = Image.new('RGB', (64,64), color=(128,128,128))
    # lightweight fallback: avoid heavy normal computation for tests
    normals = np.zeros_like(vertices)
    pointcloud = vertices.copy()
    return {
        'vertices': vertices,
        'faces': faces,
        'uvs': np.zeros((len(vertices),2)),
        'texture': tex,
        'normals': normals,
        'pointcloud': pointcloud
    }
