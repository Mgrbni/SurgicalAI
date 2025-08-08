"""Example pipeline integrating cognitive surgical modules."""
from typing import Any, Dict

from .cognitive_decision_engine import CognitiveDecisionEngine
from .contraindication_detector import ContraindicationDetector
from .structural_integrity_validator import StructuralIntegrityValidator
from .self_audit_checker import SelfAuditChecker
from .report_generator import ReportGenerator


def run_pipeline(patient_context: Dict[str, Any]) -> Dict[str, Any]:
    """Run surgical planning pipeline and return report data."""
    decision_engine = CognitiveDecisionEngine()
    contraindication_checker = ContraindicationDetector()
    integrity_validator = StructuralIntegrityValidator()
    auditor = SelfAuditChecker()
    reporter = ReportGenerator()

    decision = decision_engine.evaluate(patient_context)
    contra = contraindication_checker.check(patient_context)
    integrity = integrity_validator.validate(
        {"rotation_angle": patient_context.get("rotation_angle", 0.0)}
    )
    audit = auditor.audit(
        {
            "lesion_zone": patient_context.get("lesion_zone"),
            "heatmap_zone": patient_context.get("heatmap_zone"),
        }
    )
    report = reporter.generate(
        {
            "flap_status": integrity.flap_design_status,
            "rotation_angle": integrity.rotation_angle,
            "decision": decision.message,
            "contraindication": contra.contraindication,
        }
    )
    return {
        "decision": decision,
        "contraindication": contra,
        "integrity": integrity,
        "audit": audit,
        "report": report,
    }
