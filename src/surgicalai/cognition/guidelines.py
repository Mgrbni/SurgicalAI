"""Guidelines engine for surgical rules and recommendations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import warnings

@dataclass
class SurgicalMargins:
    """Container for surgical margin recommendations."""
    peripheral: str
    deep: str
    source_ids: List[str]

@dataclass
class ReconstructionPolicy:
    """Container for reconstruction policies."""
    defer_conditions: List[Dict[str, str]]
    proceed_condition: str

@dataclass
class FacialSubunitGuidelines:
    """Guidelines for a specific facial subunit."""
    margins: SurgicalMargins
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

def query_rules(
    diagnosis: str,
    subunit: str,
    rules: Optional[Dict] = None
) -> FacialSubunitGuidelines:
    """Get guidelines for a specific diagnosis and facial subunit.
    
    Args:
        diagnosis: One of "melanoma", "bcc", "cscc"
        subunit: Facial subunit (e.g., "nasal_dorsum")
        rules: Optional pre-loaded rules dict
        
    Returns:
        FacialSubunitGuidelines object containing recommendations
        
    Raises:
        KeyError if diagnosis or subunit not found
    """
    if rules is None:
        rules = load_rules()
        
    if diagnosis not in rules["diagnoses"]:
        raise KeyError(f"Unknown diagnosis: {diagnosis}")
        
    dx_rules = rules["diagnoses"][diagnosis]
    
    # Get margins based on diagnosis
    if "surgical_margins" in dx_rules:
        margins = next(iter(dx_rules["surgical_margins"].values()))  # Default to first
    else:
        margins = {"peripheral_cm": "Unknown", "deep": "Unknown", "source_ids": []}
        
    # Get reconstruction policy
    policy = ReconstructionPolicy(
        defer_conditions=dx_rules["reconstruction_policy"].get("defer_if", []),
        proceed_condition=dx_rules["reconstruction_policy"].get("proceed_if", "")
    )
    
    # Get flap options for this subunit
    flap_options = dx_rules.get("flap_preferences_by_subunit", {}).get(subunit, [])
    
    # Get danger structures for this subunit
    danger_structs = []
    for struct, info in rules["shared_constraints"]["danger_structures"].items():
        if info["region"] in subunit:
            danger_structs.append({
                "name": struct,
                "notes": info["notes"]
            })
            
    # Get full citations
    citations = []
    for source_id in margins["source_ids"]:
        if source_id in rules["sources"]:
            citations.append(rules["sources"][source_id])
            
    return FacialSubunitGuidelines(
        margins=SurgicalMargins(
            peripheral=margins["peripheral_cm"],
            deep=margins["deep"],
            source_ids=margins["source_ids"]
        ),
        policy=policy,
        flap_options=flap_options,
        danger_structures=danger_structs,
        citations=citations
    )
