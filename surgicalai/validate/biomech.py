# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from surgicalai.config import Config
from surgicalai.logging_utils import get_logger
from surgicalai.utils.io import read_json, write_json

LOGGER = get_logger(__name__)


def _load_polyline(path: Path) -> np.ndarray:
    if path.exists():
        return np.array(read_json(path))
    pts = np.array([[100, 0, 0], [100, 10, 0]], dtype=float)
    write_json(path, pts.tolist())
    return pts


def _min_distance(points: np.ndarray, poly: np.ndarray) -> float:
    min_d = np.inf
    for p in points:
        d = np.min(np.linalg.norm(poly - p, axis=1))
        min_d = min(min_d, d)
    return float(min_d)


def run(case_dir: Path, config: Config) -> Dict[str, object]:
    plan = read_json(case_dir / "flap_plan.json")
    arc = np.array(plan["arc"])
    anatomy_dir = case_dir / "anatomy"
    anatomy_dir.mkdir(exist_ok=True)
    alerts: List[Dict[str, object]] = []
    for name in ["arteries", "nerves"]:
        poly = _load_polyline(anatomy_dir / f"{name}.json")
        dist = _min_distance(arc, poly)
        if dist < config.validate.danger_margin_mm:
            alerts.append({"type": name, "distance_mm": dist, "details": "proximity"})
    score = float(max(0.0, 1.0 - 0.5 * len(alerts)))
    result = {"alerts": alerts, "score": score}
    write_json(case_dir / "contraindications.json", result)
    LOGGER.info("validation complete")
    return result
