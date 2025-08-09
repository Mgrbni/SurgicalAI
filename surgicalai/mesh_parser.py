# SPDX-License-Identifier: Apache-2.0
"""Polycam mesh parsing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import trimesh


def load_mesh(path: str | Path) -> Dict[str, Any]:
    """Load a mesh file and extract minimal geometric information.

    Parameters
    ----------
    path:
        Path to an OBJ/PLY/GLB file produced by Polycam or similar tools.

    Returns
    -------
    dict
        Dictionary containing ``coordinates`` (``np.ndarray`` of shape ``(n, 3)``),
        simple ``landmarks`` and ``surface_topology`` statistics.
    """

    mesh = trimesh.load_mesh(str(path), process=False)
    coords = np.asarray(mesh.vertices, dtype=float)

    # Bounding box heuristics for fake landmarks
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = coords.mean(axis=0)
    landmarks = {
        "nasion": [center[0], maxs[1], center[2]],
        "pronasale": [maxs[0], center[1], center[2]],
        "gnathion": [center[0], mins[1], center[2]],
    }

    # Synthetic surface statistic: mean distance from origin
    curvature_mean = float(np.linalg.norm(coords, axis=1).mean())
    surface_topology = {"curvature_mean": curvature_mean}

    return {
        "coordinates": coords,
        "landmarks": landmarks,
        "surface_topology": surface_topology,
    }
