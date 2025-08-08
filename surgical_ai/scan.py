"""Module for 3D scan ingestion and normalization.

This module exposes :class:`ScanProcessor` for loading Polycam exports and
``ScanData`` which wraps raw and normalized representations.  The functions are
stubs and should be extended with ``open3d`` or ``trimesh`` integration once
real data is available.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ScanData:
    """Container for a 3D scan and related metadata.

    Attributes
    ----------
    raw: Any
        Original scan object as loaded from disk.
    normalized: Any | None
        Scan after size/orientation normalization.  ``None`` until processed.
    """

    raw: Any
    normalized: Optional[Any] = None


class ScanProcessor:
    """Handles loading and normalizing 3D facial scans."""

    def load_scan(self, file_path: str | Path) -> ScanData:
        """Load a 3D scan from ``file_path``.

        Parameters
        ----------
        file_path:
            Path to a ``.obj`` or ``.ply`` scan.

        Returns
        -------
        ScanData
            Wrapper containing the raw scan.
        """
        # TODO: implement loading using open3d or trimesh
        raise NotImplementedError

    def normalize(self, scan: ScanData) -> ScanData:
        """Normalize scale, rotation and orientation of ``scan``.

        Parameters
        ----------
        scan:
            Loaded scan object.

        Returns
        -------
        ScanData
            Updated instance with ``normalized`` field populated.
        """
        # TODO: implement normalization logic
        raise NotImplementedError
