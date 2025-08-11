"""Gate checks for surgical contraindications."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

from surgicalai.cognition.guidelines import load_rules

@dataclass
class GateDecision:
    """Container for gate decision and rationale."""
    allow: bool
    reason: str
    citations: List[Dict[str, str]]

class GuidelineGate:
    """Gate checker using guideline rules."""
    
    def __init__(self, rules: Optional[Dict] = None):
        """Initialize with optional pre-loaded rules."""
        self.rules = rules if rules is not None else load_rules()
        
    def check(
        self,
        diagnosis: str,
        subunit: str,
        melanoma_prob: Optional[float] = None,
        high_risk_features: Optional[List[str]] = None
    ) -> GateDecision:
        """Check if reconstruction should proceed based on guidelines.
        
        Args:
            diagnosis: One of "melanoma", "bcc", "cscc"
            subunit: Facial subunit
            melanoma_prob: Optional melanoma probability (0-1)
            high_risk_features: Optional list of high risk features present
            
        Returns:
            GateDecision with allow/deny and rationale
        """
        dx_rules = self.rules["diagnoses"][diagnosis]
        citations = []
        
        # Check if subunit is in H-zone
        is_h_zone = any(
            zone in subunit 
            for zone in self.rules["shared_constraints"]["h_zone"]
        )
        
        # Melanoma specific checks
        if diagnosis == "melanoma":
            melanoma_threshold = 0.30  # From settings
            
            if is_h_zone:
                return GateDecision(
                    allow=False,
                    reason="H-zone location requires tissue-sparing approach",
                    citations=[
                        self.rules["sources"][src_id]
                        for src_id in ["NCCN_MEL", "AAD_MEL"]
                    ]
                )
                
            if melanoma_prob and melanoma_prob >= melanoma_threshold:
                return GateDecision(
                    allow=False,
                    reason=f"Melanoma probability ({melanoma_prob:.2f}) exceeds threshold",
                    citations=[
                        self.rules["sources"][src_id]
                        for src_id in ["NCCN_MEL", "AAD_MEL"]
                    ]
                )
                
        # BCC checks
        elif diagnosis == "bcc":
            high_risk = high_risk_features or []
            
            if is_h_zone and "margin_control" not in high_risk:
                return GateDecision(
                    allow=False,
                    reason="H-zone BCC requires margin-controlled excision",
                    citations=[self.rules["sources"]["NCCN_BCC"]]
                )
                
        # cSCC checks  
        elif diagnosis == "cscc":
            high_risk = high_risk_features or []
            
            if is_h_zone and "margin_control" not in high_risk:
                return GateDecision(
                    allow=False, 
                    reason="H-zone cSCC requires margin-controlled excision",
                    citations=[
                        self.rules["sources"][src_id]
                        for src_id in ["NCCN_cSCC", "AAD_cSCC"]
                    ]
                )
                
        # Default allow if no contraindications found
        return GateDecision(
            allow=True,
            reason="No guideline contraindications",
            citations=[]
        )
