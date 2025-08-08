"""Facial landmark detection utilities."""

from typing import Any


class LandmarkDetector:
    """Detects anatomical landmarks and optional tension lines."""

    def detect(self, scan: Any) -> dict[str, tuple[float, float, float]]:
        """Detect landmarks on ``scan``.

        Parameters
        ----------
        scan: Any
            Normalized 3D scan.

        Returns
        -------
        dict
            Mapping of landmark names to 3D coordinates.
        """
        # TODO: integrate with face-alignment or mediapipe
        raise NotImplementedError

    def map_tension_lines(self, scan: Any) -> Any:
        """Estimate facial tension lines (e.g., Langer's lines)."""
        # TODO: implement skin tension estimation
        raise NotImplementedError
