"""3D visualization utilities."""

from typing import Any


class Visualizer:
    """Renders scans, heatmaps and flap designs."""

    def render_face(self, scan: Any) -> None:
        """Render the base 3D facial scan."""
        # TODO: render using open3d, vedo or plotly
        raise NotImplementedError

    def overlay_lesion(self, scan: Any, heatmap: Any) -> None:
        """Overlay lesion heatmap onto ``scan``."""
        # TODO: map 2D heatmap back to 3D mesh
        raise NotImplementedError

    def show_flap(self, scan: Any, flap_info: Any) -> None:
        """Visualize suggested flap on ``scan``."""
        # TODO: display flap geometry on mesh
        raise NotImplementedError
