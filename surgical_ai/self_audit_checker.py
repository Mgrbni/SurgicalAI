"""Self-audit and consistency checker."""
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AuditLog:
    step: str
    status: str
    details: str


@dataclass
class AuditResult:
    logs: List[AuditLog] = field(default_factory=list)
    consistent: bool = True


class SelfAuditChecker:
    """Ensures pipeline consistency across processing stages."""

    def audit(self, pipeline_state: Dict[str, Any]) -> AuditResult:
        """Check for contradictions in pipeline state."""
        result = AuditResult()
        lesion_zone = pipeline_state.get("lesion_zone")
        heatmap_zone = pipeline_state.get("heatmap_zone")
        if lesion_zone and heatmap_zone and lesion_zone != heatmap_zone:
            result.consistent = False
            result.logs.append(
                AuditLog(
                    step="zone_alignment",
                    status="conflict",
                    details="Lesion and heatmap zones mismatch",
                )
            )
        else:
            result.logs.append(
                AuditLog(step="zone_alignment", status="ok", details="Zones aligned")
            )
        return result
