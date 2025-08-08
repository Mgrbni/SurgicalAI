"""Contraindication and risk detection module."""
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ContraindicationResult:
    contraindication: bool
    reason: Optional[str] = None
    suggested_alternative: Optional[str] = None


class ContraindicationDetector:
    """Detects potential contraindications for a surgical flap."""

    def check(self, anatomy_context: Dict[str, Any]) -> ContraindicationResult:
        """Evaluate anatomy for possible contraindications."""
        # Simplified rule: flag if lesion is near critical structure
        if anatomy_context.get("near_nerve", False):
            return ContraindicationResult(
                True,
                reason="Lesion near critical nerve",
                suggested_alternative="Consider transposition flap",
            )
        return ContraindicationResult(False)
