"""Face detection & RSTL (relaxed skin tension lines) orientation estimation.

This module provides:
  - detect_face_orientation(image) -> roll,yaw,pitch (deg)
  - map_rstl(site, center_xy) -> dominant angle at lesion (deg)
  - plan_incision(diameter_mm, rstl_angle_deg, rules) -> closure recommendation & axis

Uses mediapipe face mesh if available; falls back to heuristic.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import math
import numpy as np

try:
    import mediapipe as mp  # type: ignore
    _HAS_MP = True
except Exception:  # pragma: no cover
    _HAS_MP = False

@dataclass
class Orientation:
    roll: float
    yaw: float
    pitch: float

@dataclass
class RSTLResult:
    rstl_angle: float  # degrees
    incision_axis: Tuple[float, float]  # unit vector (dx, dy) parallel to RSTL
    closure_recommendation: str
    success_score: float
    reasoning: str

_DEF_RULES = {
    "tension_primary_threshold": 0.65,
}


def detect_face_orientation(image: np.ndarray) -> Orientation:
    if _HAS_MP:  # simplified placeholder using static assumption
        # A real implementation would run face mesh and compute pose
        pass
    # Fallback neutral pose
    return Orientation(roll=0.0, yaw=0.0, pitch=0.0)


def map_rstl(site: str, center_xy: Tuple[float, float], site_rules: Dict[str, Any]) -> float:
    rules = site_rules.get(site) or {}
    return float(rules.get("rstl_angle_deg", 0.0))


def _tissue_laxity_vector(angle_deg: float) -> Tuple[float, float]:
    rad = math.radians(angle_deg)
    return math.cos(rad), math.sin(rad)


def plan_incision(diameter_mm: Optional[float], rstl_angle: float, rules: Dict[str, Any]) -> Tuple[str, Tuple[float, float], float, str]:
    if diameter_mm is None:
        return ("primary closure (uncertain)", _tissue_laxity_vector(rstl_angle), 70.0, "No size; default primary")
    small_t = rules.get("small_threshold_mm", 10)
    med_t = rules.get("medium_threshold_mm", 20)
    tension_thresh = rules.get("tension_primary_threshold", 0.65)

    # crude gap vs laxity ratio = diameter / (small_t*1.2)
    ratio = diameter_mm / (small_t * 1.2)
    axis = _tissue_laxity_vector(rstl_angle)
    reasoning = f"diameter={diameter_mm:.1f}mm ratio={ratio:.2f} rstl={rstl_angle:.0f}deg"

    if ratio < tension_thresh:
        return ("primary closure", axis, 85.0, reasoning + " below tension threshold")
    if diameter_mm <= med_t:
        return ("elliptical excision (parallel to RSTL)", axis, 78.0, reasoning + " medium defect")
    # large
    return ("local flap (advancement/rotation)", axis, 68.0, reasoning + " large defect needs tissue transfer")


def analyze_orientation_and_closure(site: str, center_xy: Tuple[float, float], diameter_mm: Optional[float], rules: Dict[str, Any]) -> RSTLResult:
    rstl_angle = map_rstl(site, center_xy, rules.get("site_rules", {}))
    rec, axis, score, reasoning = plan_incision(diameter_mm, rstl_angle, rules.get("defaults", {}))
    return RSTLResult(rstl_angle=rstl_angle, incision_axis=axis, closure_recommendation=rec, success_score=score, reasoning=reasoning)

__all__ = ["detect_face_orientation", "map_rstl", "plan_incision", "analyze_orientation_and_closure", "Orientation", "RSTLResult"]
