"""3D visualization utilities."""

from typing import Any

from .flap import FlapPlan
from .lesion import LesionResult
from .scan import ScanData


class Visualizer:
    """Renders scans, heatmaps and flap designs."""

    def render_face(self, scan: ScanData) -> None:
        """Render the base 3D facial scan."""
        # TODO: render using open3d, vedo or plotly
        raise NotImplementedError

    def overlay_lesion(self, scan: ScanData, lesion: LesionResult) -> None:
        """Overlay lesion heatmap onto ``scan``."""
        # TODO: map 2D heatmap back to 3D mesh
        raise NotImplementedError

    def show_flap(self, scan: ScanData, flap_info: FlapPlan) -> None:
        """Visualize suggested flap on ``scan``."""
        # TODO: display flap geometry on mesh
        raise NotImplementedError
