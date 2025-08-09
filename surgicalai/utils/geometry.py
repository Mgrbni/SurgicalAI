# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np


def centroid(points: np.ndarray) -> np.ndarray:
    return points.mean(axis=0)


def principal_curvature(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.zeros(len(verts))
