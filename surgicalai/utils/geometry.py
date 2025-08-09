"""Geometry helpers."""
from __future__ import annotations
import numpy as np
from scipy.spatial import KDTree


def build_kdtree(points: np.ndarray) -> KDTree:
    return KDTree(points)


def barycentric_coordinates(p, a, b, c):
    """Compute barycentric coordinates of p with respect to triangle abc."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return 0,0,0
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w
