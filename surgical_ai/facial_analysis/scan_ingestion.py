"""3D facial scan ingestion and normalization utilities."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ScanData:
    """Normalized scan data placeholder."""

    raw: Any


class ScanIngestor:
    """Loads and normalizes 3D facial scans."""

    def load_scan(self, path: Path) -> ScanData:
        """Load a 3D scan and return normalized data.

        Parameters
        ----------
        path: Path
            File path to .obj or .ply scan.

        Returns
        -------
        ScanData
            Placeholder for normalized scan representation.
        """
        # Placeholder: in real implementation, load mesh and normalize.
        return ScanData(raw=None)
