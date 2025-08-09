"""Contraindication checks against vessels."""
from __future__ import annotations
import numpy as np
from ..anatomy.atlas import load_atlas
from ..config import CONFIG


def assess_risk(point: np.ndarray) -> dict:
    atlas = load_atlas()
    vessels = np.array(atlas['vessels']['vessels'])
    dists = []
    for v in vessels:
        v0, v1 = np.array(v[0]), np.array(v[1])
        # Some atlas entries are 2-D.  Pad with a zero z-coordinate to
        # operate in 3-D space.
        if v0.shape[0] == 2:
            v0 = np.append(v0, 0.0)
            v1 = np.append(v1, 0.0)
        d = np.linalg.norm(np.cross(v1 - v0, v0 - point)) / (np.linalg.norm(v1 - v0) + 1e-8)
        dists.append(d)
    min_dist = min(dists) if dists else 1.0
    level = 'LOW'
    if min_dist < CONFIG.vessel_risk_mm[0]/100:
        level = 'HIGH'
    elif min_dist < CONFIG.vessel_risk_mm[1]/100:
        level = 'MED'
    return {'min_dist': float(min_dist), 'level': level}
