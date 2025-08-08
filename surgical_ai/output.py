"""Reporting and logging utilities."""

from typing import Any


class ReportGenerator:
    """Exports diagnostic and surgical planning results."""

    def save_report(self, data: dict[str, Any], path: str) -> None:
        """Save a PDF or text report summarizing ``data``."""
        # TODO: implement using reportlab or similar library
        raise NotImplementedError

    def save_metadata(self, data: dict[str, Any], path: str) -> None:
        """Save ``data`` to JSON or CSV for downstream use."""
        # TODO: implement structured metadata export
        raise NotImplementedError
