"""Module for 3D scan ingestion and normalization."""

from pathlib import Path
from typing import Any


class ScanProcessor:
    """Handles loading and normalizing 3D facial scans."""

    def load_scan(self, file_path: str | Path) -> Any:
        """Load a 3D scan from ``file_path``.

        Parameters
        ----------
        file_path: str or Path
            Path to a ``.obj`` or ``.ply`` scan.

        Returns
        -------
        Any
            Placeholder for the loaded scan object.
        """
        # TODO: implement loading using open3d or trimesh
        raise NotImplementedError

    def normalize(self, scan: Any) -> Any:
        """Normalize scale, rotation and orientation of ``scan``.

        Parameters
        ----------
        scan: Any
            Loaded scan object.

        Returns
        -------
        Any
            Normalized scan.
        """
        # TODO: implement normalization logic
        raise NotImplementedError
