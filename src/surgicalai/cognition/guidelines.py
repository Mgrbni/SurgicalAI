"""Guidelines engine for surgical rules and recommendations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
import warnings
import math

@dataclass
class SurgicalMargins:
    """Container for surgical margin recommendations."""
    peripheral: str
    deep: str
    source_ids: List[str]
    rationale: str
    defect_calculation: str

@dataclass
class MarginCalculation:
    """Container for calculated margin dimensions."""
    peripheral_mm: float
    defect_diameter_mm: float
    defect_area_mm2: float
    confidence: float
    calculation_method: str

@dataclass
class ReconstructionPolicy:
    """Container for reconstruction policies."""
    defer_conditions: List[Dict[str, str]]
    proceed_condition: str

@dataclass
class FacialSubunitGuidelines:
    """Guidelines for a specific facial subunit."""
    margins: SurgicalMargins
    margin_calculation: Optional[MarginCalculation]
    policy: ReconstructionPolicy
    flap_options: List[str]
    danger_structures: List[Dict[str, str]]
    citations: List[Dict[str, str]]

def load_rules(rules_path: Path = None) -> Dict:
    """Load surgical guidelines from YAML.
    
    Args:
        rules_path: Path to rules YAML. Defaults to package rules/face_rules.yaml
        
    Returns:
        Dict containing parsed rules
    """
    if rules_path is None:
        rules_path = Path(__file__).parent.parent / "rules" / "face_rules.yaml"
    
    with open(rules_path) as f:
        rules = yaml.safe_load(f)
    
    return rules

def validate_sources(rules: Dict) -> bool:
    """Validate that all numeric rules have source citations.
    
    Args:
        rules: Dict containing parsed rules
        
    Returns:
        True if all rules have valid citations
    
    Raises:
        Warning if any rule lacks citations
    """
    valid = True
    missing_citations = []
    
    # Check surgical margins
    for dx in rules["diagnoses"]:
        dx_rules = rules["diagnoses"][dx]
        
        # Check margins
        if "surgical_margins" in dx_rules:
            for thickness, margins in dx_rules["surgical_margins"].items():
                if "source_ids" not in margins:
                    missing_citations.append(f"{dx} margins for {thickness}")
                    valid = False
                elif not margins["source_ids"]:
                    missing_citations.append(f"{dx} margins for {thickness}")
                    valid = False
    
    if not valid:
        warnings.warn(
            f"Missing citations for: {', '.join(missing_citations)}",
            stacklevel=2
        )
    
    return valid

def calculate_margins(
    diagnosis: str, 
    lesion_diameter_mm: float,
    breslow_depth_mm: Optional[float] = None,
    risk_factors: Optional[List[str]] = None,
    rules: Optional[Dict] = None
) -> MarginCalculation:
    """Calculate surgical margins based on diagnosis and lesion characteristics.
    
    Args:
        diagnosis: Primary diagnosis
        lesion_diameter_mm: Lesion diameter in millimeters
        breslow_depth_mm: Breslow depth for melanoma (optional)
        risk_factors: List of risk factors (optional)
        rules: Pre-loaded rules dict (optional)
        
    Returns:
        MarginCalculation with computed dimensions
    """
    if rules is None:
        rules = load_rules()
    
    risk_factors = risk_factors or []
    
    # Get margin requirements for diagnosis
    margin_info = _get_margin_requirements(diagnosis, breslow_depth_mm, risk_factors, rules)
    
    if not margin_info:
        # Fallback conservative margins
        return MarginCalculation(
            peripheral_mm=5.0,
            defect_diameter_mm=lesion_diameter_mm + 10.0,
            defect_area_mm2=math.pi * ((lesion_diameter_mm + 10.0) / 2) ** 2,
            confidence=0.3,
            calculation_method="fallback_conservative"
        )
    
    # Parse margin specification
    peripheral_mm = _parse_margin_spec(margin_info.get("peripheral_mm", margin_info.get("peripheral_cm", "5")))
    
    # Calculate defect dimensions
    defect_diameter = lesion_diameter_mm + (2 * peripheral_mm)
    defect_area = math.pi * (defect_diameter / 2) ** 2
    
    # Determine confidence based on evidence level and specificity
    confidence = _calculate_margin_confidence(margin_info, diagnosis, breslow_depth_mm)
    
    return MarginCalculation(
        peripheral_mm=peripheral_mm,
        defect_diameter_mm=defect_diameter,
        defect_area_mm2=defect_area,
        confidence=confidence,
        calculation_method=f"{diagnosis}_guideline_based"
    )

def _get_margin_requirements(
    diagnosis: str,
    breslow_depth_mm: Optional[float],
    risk_factors: List[str],
    rules: Dict
) -> Optional[Dict]:
    """Get margin requirements for specific diagnosis and characteristics."""
    
    if diagnosis not in rules["diagnoses"]:
        return None
    
    dx_rules = rules["diagnoses"][diagnosis]
    
    # Handle melanoma with Breslow depth
    if diagnosis == "melanoma" and "surgical_margins" in dx_rules:
        if breslow_depth_mm is None:
            return dx_rules["surgical_margins"].get("unknown_depth")
        elif breslow_depth_mm == 0:
            return dx_rules["surgical_margins"].get("in_situ")
        elif breslow_depth_mm <= 1.0:
            return dx_rules["surgical_margins"].get("<=1mm")
        elif breslow_depth_mm <= 2.0:
            return dx_rules["surgical_margins"].get("1.01–2mm")
        else:
            return dx_rules["surgical_margins"].get(">2mm")
    
    # Handle BCC and cSCC with risk stratification
    elif diagnosis in ["bcc", "cscc"]:
        # Check for high-risk factors
        high_risk_flags = dx_rules.get("risk_stratification", {}).get("high_risk_flags", [])
        is_high_risk = any(factor in risk_factors for factor in high_risk_flags)
        
        if is_high_risk:
            return dx_rules.get("surgical_margins_high_risk")
        else:
            return dx_rules.get("surgical_margins_low_risk")
    
    # Handle other diagnoses
    elif "surgical_margins" in dx_rules:
        # For nevus, check for dysplasia
        if diagnosis == "nevus":
            if "dysplastic" in risk_factors or "atypical" in risk_factors:
                return dx_rules["surgical_margins"].get("atypical_dysplastic")
            else:
                return dx_rules["surgical_margins"].get("benign_appearing")
        
        # For seborrheic keratosis
        elif diagnosis == "seb_keratosis":
            return dx_rules["surgical_margins"].get("typical")
    
    return None

def _parse_margin_spec(margin_spec: str) -> float:
    """Parse margin specification string to numeric value in mm."""
    
    # Handle range specifications (take the larger value for safety)
    if "–" in margin_spec or "-" in margin_spec:
        parts = margin_spec.replace("–", "-").split("-")
        try:
            values = [float(part.strip()) for part in parts]
            margin_val = max(values)
        except ValueError:
            margin_val = 5.0  # Default fallback
    else:
        # Handle single values
        margin_str = margin_spec.strip().replace("≥", "").replace("mm", "").replace("cm", "")
        try:
            margin_val = float(margin_str)
            # Convert cm to mm if necessary
            if "cm" in margin_spec:
                margin_val *= 10
        except ValueError:
            margin_val = 5.0  # Default fallback
    
    return margin_val

def _calculate_margin_confidence(
    margin_info: Dict,
    diagnosis: str,
    breslow_depth_mm: Optional[float]
) -> float:
    """Calculate confidence in margin recommendation."""
    
    base_confidence = 0.8
    
    # Higher confidence for well-established guidelines
    if diagnosis == "melanoma" and breslow_depth_mm is not None:
        base_confidence = 0.95  # Well-established evidence
    elif diagnosis in ["bcc", "cscc"]:
        base_confidence = 0.85  # Good evidence base
    
    # Adjust for source quality
    source_ids = margin_info.get("source_ids", [])
    if any("NCCN" in source for source in source_ids):
        base_confidence += 0.05
    if any("AAD" in source for source in source_ids):
        base_confidence += 0.03
    
    return min(base_confidence, 1.0)

def query_rules(
    diagnosis: str,
    subunit: str,
    lesion_diameter_mm: Optional[float] = None,
    breslow_depth_mm: Optional[float] = None,
    risk_factors: Optional[List[str]] = None,
    rules: Optional[Dict] = None
) -> FacialSubunitGuidelines:
    """Get comprehensive guidelines for a specific diagnosis and facial subunit.
    
    Args:
        diagnosis: One of "melanoma", "bcc", "cscc", "nevus", "seb_keratosis"
        subunit: Facial subunit (e.g., "nasal_dorsum")
        lesion_diameter_mm: Lesion diameter for margin calculation
        breslow_depth_mm: Breslow depth for melanoma
        risk_factors: List of risk factors
        rules: Optional pre-loaded rules dict
        
    Returns:
        FacialSubunitGuidelines object containing comprehensive recommendations
        
    Raises:
        KeyError if diagnosis or subunit not found
    """
    if rules is None:
        rules = load_rules()
        
    if diagnosis not in rules["diagnoses"]:
        raise KeyError(f"Unknown diagnosis: {diagnosis}")
        
    dx_rules = rules["diagnoses"][diagnosis]
    risk_factors = risk_factors or []
    
    # Get margins based on diagnosis and characteristics
    margin_info = _get_margin_requirements(diagnosis, breslow_depth_mm, risk_factors, rules)
    
    if margin_info:
        margins = SurgicalMargins(
            peripheral=margin_info.get("peripheral_cm", margin_info.get("peripheral_mm", "Unknown")),
            deep=margin_info.get("deep", "Unknown"),
            source_ids=margin_info.get("source_ids", []),
            rationale=margin_info.get("rationale", ""),
            defect_calculation=margin_info.get("defect_calculation", "")
        )
    else:
        margins = SurgicalMargins(
            peripheral="Unknown",
            deep="Unknown",
            source_ids=[],
            rationale="No specific guidelines available",
            defect_calculation="Conservative estimation required"
        )
    
    # Calculate margins if lesion diameter provided
    margin_calculation = None
    if lesion_diameter_mm:
        margin_calculation = calculate_margins(
            diagnosis, lesion_diameter_mm, breslow_depth_mm, risk_factors, rules
        )
    
    # Get reconstruction policy
    policy = ReconstructionPolicy(
        defer_conditions=dx_rules.get("reconstruction_policy", {}).get("defer_if", []),
        proceed_condition=dx_rules.get("reconstruction_policy", {}).get("proceed_if", "")
    )
    
    # Get flap options for this subunit
    flap_options = dx_rules.get("flap_preferences_by_subunit", {}).get(subunit, [])
    
    # Get danger structures for this subunit
    danger_structs = []
    for struct, info in rules["shared_constraints"]["danger_structures"].items():
        if info["region"] in subunit or subunit in info["region"]:
            danger_structs.append({
                "name": struct,
                "notes": info["notes"]
            })
            
    # Get full citations
    citations = []
    if margin_info:
        for source_id in margin_info.get("source_ids", []):
            if source_id in rules["sources"]:
                citations.append(rules["sources"][source_id])
            
    return FacialSubunitGuidelines(
        margins=margins,
        margin_calculation=margin_calculation,
        policy=policy,
        flap_options=flap_options,
        danger_structures=danger_structs,
        citations=citations
    )
