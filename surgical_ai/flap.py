"""Flap design suggestion engine."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class FlapPlan:
    """Representation of a local flap recommendation."""

    flap_type: str
    axis: Optional[Tuple[float, float, float]] = None
    success_rate: Optional[float] = None
    extra: Dict[str, Any] | None = None


class FlapDesigner:
    """Suggests optimal local flap designs based on lesion data."""

    def suggest(
        self, lesion_location: Tuple[float, float, float], tension_map: Any
    ) -> FlapPlan:
        """Suggest a flap design.

        Parameters
        ----------
        lesion_location:
            3D coordinates of the lesion.
        tension_map:
            Representation of facial tension lines.

        Returns
        -------
        FlapPlan
            Information about the proposed flap, e.g., type and orientation.
        """
        # TODO: implement rule-based or ML-driven flap suggestion
        raise NotImplementedError
