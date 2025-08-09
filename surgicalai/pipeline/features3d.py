"""3D feature extraction."""
from __future__ import annotations
import numpy as np


def compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    tris = vertices[faces]
    n = np.cross(tris[:,1]-tris[:,0], tris[:,2]-tris[:,0])
    n = n / (np.linalg.norm(n, axis=1, keepdims=True)+1e-8)
    for i, face in enumerate(faces):
        for v in face:
            normals[v] += n[i]
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True)+1e-8)
    return normals
