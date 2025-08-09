from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from surgicalai.config import Config
from surgicalai.logging_utils import get_logger
from surgicalai.utils.io import load_npz, write_json

LOGGER = get_logger(__name__)


def run(case_dir: Path, config: Config) -> Dict[str, object]:
    heat = np.load(case_dir / "heatmap_vertex.npy")
    mesh = load_npz(case_dir / "mesh.npz")
    verts = mesh["vertices"]
    lesion = verts[heat > config.analyze.gradcam_threshold]
    if len(lesion) == 0:
        lesion = verts
    centroid = lesion.mean(axis=0)
    diameter = float(np.linalg.norm(lesion.max(axis=0) - lesion.min(axis=0)))
    pivot = centroid + np.array([diameter, 0, 0])
    arc_len = config.plan.arc_multiplier * diameter
    theta = np.linspace(0, np.pi / 2, 5)
    arc = [list(pivot + [0, np.cos(t) * arc_len, np.sin(t) * arc_len]) for t in theta]
    tension_axis = [0.0, 1.0, 0.0]
    success = max(0.0, min(1.0, 1.0 - arc_len / 10.0))
    plan = {
        "type": "rotation",
        "pivot": pivot.tolist(),
        "arc": arc,
        "tension_axis": tension_axis,
        "success_prob": success,
        "notes": "auto-generated"
    }
    write_json(case_dir / "flap_plan.json", plan)
    LOGGER.info("planning complete")
    return plan
