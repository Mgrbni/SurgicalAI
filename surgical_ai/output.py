"""Reporting and logging utilities."""

from dataclasses import asdict
from typing import Any, Dict


class ReportGenerator:
    """Exports diagnostic and surgical planning results."""

    def save_report(self, data: Dict[str, Any], path: str) -> None:
        """Save a PDF or text report summarizing ``data``."""
        # TODO: implement using reportlab or similar library
        raise NotImplementedError

    def save_metadata(self, data: Dict[str, Any], path: str) -> None:
        """Save ``data`` to JSON or CSV for downstream use."""
        # TODO: implement structured metadata export
        raise NotImplementedError

    def dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Utility to convert dataclasses to dictionaries recursively."""
        try:
            return asdict(obj)
        except TypeError:  # not a dataclass
            return dict(obj)
