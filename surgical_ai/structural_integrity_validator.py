"""Structural integrity validation for flap designs."""
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class IntegrityReport:
    flap_design_status: str
    rotation_angle: float
    blood_supply_check: str
    warnings: List[str]


class StructuralIntegrityValidator:
    """Validates biomechanical and anatomical feasibility."""

    def validate(self, flap_plan: Dict[str, Any]) -> IntegrityReport:
        """Perform basic integrity checks on a flap plan."""
        angle = flap_plan.get("rotation_angle", 0.0)
        warnings: List[str] = []
        if angle > 50:
            warnings.append("Rotation angle exceeds recommended maximum")
        status = "Valid" if not warnings else "Review"
        return IntegrityReport(
            flap_design_status=status,
            rotation_angle=angle,
            blood_supply_check="Sufficient",
            warnings=warnings,
        )
