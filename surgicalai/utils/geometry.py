import numpy as np
from scipy.spatial import KDTree


def build_kdtree(points: np.ndarray) -> KDTree:
    """Build a KDTree from a set of points."""
    return KDTree(points)


def barycentric_weights(triangle: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Compute barycentric weights of a point with respect to a triangle."""
    triangle = np.asarray(triangle)
    T = triangle[1:] - triangle[0]
    v = point - triangle[0]
    sol, _, _, _ = np.linalg.lstsq(T.T, v, rcond=None)
    w1, w2 = sol
    w0 = 1 - w1 - w2
    return np.array([w0, w1, w2])


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform matrix to 3D points."""
    pts = np.c_[points, np.ones(len(points))]
    return (pts @ matrix.T)[:, :3]
