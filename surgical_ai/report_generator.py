"""Smart procedural report generation."""
from dataclasses import dataclass
from typing import Any, Dict
import json


@dataclass
class Report:
    summary: str
    data: Dict[str, Any]


class ReportGenerator:
    """Generate human-readable and machine-friendly reports."""

    def generate(self, context: Dict[str, Any]) -> Report:
        """Create a procedural report from context data."""
        summary = (
            f"Flap Status: {context.get('flap_status', 'unknown')}; "
            f"Rotation Angle: {context.get('rotation_angle', 'n/a')}; "
            f"Decision: {context.get('decision', 'n/a')}; "
            f"Contraindication: {context.get('contraindication', 'n/a')}"
        )
        return Report(summary=summary, data=context)

    def to_json(self, report: Report) -> str:
        """Serialize report to JSON."""
        return json.dumps({"summary": report.summary, **report.data}, indent=2)
