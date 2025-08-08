"""3D facial scan ingestion and normalization utilities."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ScanData:
    """Container for raw and normalized scan representations."""

    raw: Any
    normalized: Any | None = None


class ScanIngestor:
    """Loads and normalizes 3D facial scans."""

    def load_scan(self, path: Path) -> ScanData:
        """Load a 3D scan, normalize it, and return structured data.

        Parameters
        ----------
        path: Path
            File path to `.obj` or `.ply` scan export from Polycam.

        Returns
        -------
        ScanData
            Placeholder for raw and normalized scan representation.
        """
        # Placeholder: in real implementation, load mesh with open3d/trimesh
        # and run `_normalize` on the result.
        mesh = None
        normalized = self._normalize(mesh)
        return ScanData(raw=mesh, normalized=normalized)

    def _normalize(self, mesh: Any) -> Any:
        """Normalize mesh size, rotation, and orientation."""
        # Placeholder: center the mesh and standardize units/axes.
        return None
