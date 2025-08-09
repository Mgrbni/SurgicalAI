# SPDX-License-Identifier: Apache-2.0
"""Synthetic rotational flap designer."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from .utils import seed_everything


def design(mesh_data: Dict[str, Any], lesion_center) -> Dict[str, Any]:
    """Return a deterministic rotational flap plan."""
    seed_everything(1)
    lesion_center = np.asarray(lesion_center, dtype=float)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])  # unit vectors
    flap_vectors = (vectors + lesion_center[:2]).tolist()
    return {"flap_vectors": flap_vectors, "success_probability": 0.78}
