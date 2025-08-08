"""Facial landmark detection utilities.

The :class:`LandmarkDetector` returns :class:`LandmarkResult` containing both
anatomical points and an optional tension-line representation.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class LandmarkResult:
    """Detected landmarks and tension map."""

    landmarks: Dict[str, Tuple[float, float, float]]
    tension_map: Optional[Any] = None


class LandmarkDetector:
    """Detects anatomical landmarks and optional tension lines."""

    def detect(self, scan: Any) -> LandmarkResult:
        """Detect landmarks on ``scan``.

        Parameters
        ----------
        scan:
            Normalized 3D scan.

        Returns
        -------
        LandmarkResult
            Mapping of landmark names to 3D coordinates and tension map.
        """
        # TODO: integrate with face-alignment or mediapipe
        raise NotImplementedError

    def map_tension_lines(self, scan: Any) -> Any:
        """Estimate facial tension lines (e.g., Langer's lines)."""
        # TODO: implement skin tension estimation
        raise NotImplementedError
