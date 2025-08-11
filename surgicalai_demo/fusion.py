"""
Neuro-symbolic fusion of CNN predictions and Vision-LLM observations.

This module combines traditional CNN-based skin lesion classification with
vision-language model (VLM) observations to improve diagnostic accuracy,
particularly for seborrheic keratosis detection.

Key Features:
- Descriptor-based probability adjustments
- Configurable CNN/VLM weight blending
- Tie-breaking logic for uncertain cases
- Explainable fusion decisions
- Calibrated probability normalization

Fusion Strategy:
1. Extract descriptors from VLM analysis
2. Apply rule-based probability boosts for matching descriptors
3. Weighted combination of CNN and adjusted VLM probabilities
4. Re-ranking and uncertainty detection
5. Generate explanatory notes for clinical interpretation

NOT FOR CLINICAL USE - Research and demonstration purposes only.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import warnings

# Default fusion configuration
DEFAULT_FUSION_CONFIG = {
    "cnn_weight": 0.7,
    "vlm_weight": 0.3,
    "sk_descriptor_bonus": 0.15,
    "melanoma_descriptor_bonus": 0.10,
    "bcc_descriptor_bonus": 0.05,
    "scc_descriptor_bonus": 0.05,
    "tie_margin": 0.06,
    "min_confidence_threshold": 0.30,
    "uncertainty_boost": 0.05
}

# Descriptor patterns for different lesion types
SEBORRHEIC_KERATOSIS_DESCRIPTORS = {
    "stuck-on appearance",
    "waxy surface", 
    "sharply demarcated",
    "milia-like cysts",
    "comedo-like openings",
    "verrucous surface",
    "cerebriform pattern",
    "horn cysts"
}

MELANOMA_DESCRIPTORS = {
    "irregular pigment network",
    "blue-gray ovoid nests",
    "atypical vascular pattern",
    "asymmetric pigmentation",
    "variegated coloring",
    "ulceration",
    "irregular dots/globules",
    "regression structures"
}

BCC_DESCRIPTORS = {
    "pearly rolled border",
    "telangiectasias",
    "ulceration",
    "blue-gray ovoid nests",
    "arborizing vessels",
    "shiny surface",
    "translucent appearance"
}

SCC_DESCRIPTORS = {
    "hyperkeratosis",
    "ulceration", 
    "irregular surface",
    "ill-defined borders",
    "erythematous base",
    "crusting",
    "induration"
}

NEVUS_DESCRIPTORS = {
    "regular pigment network",
    "uniform coloring",
    "symmetric appearance",
    "well-demarcated",
    "stable appearance",
    "uniform dots/globules"
}


class NeuroSymbolicFusion:
    """Fusion engine for CNN and VLM predictions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize fusion engine.
        
        Args:
            config: Fusion configuration parameters
        """
        self.config = {**DEFAULT_FUSION_CONFIG}
        if config:
            self.config.update(config)
        
        # Standard lesion class names
        self.class_names = [
            "melanoma", "bcc", "scc", "nevus", "seborrheic_keratosis", "benign_other"
        ]
    
    def fuse_predictions(
        self, 
        cnn_probs: Dict[str, float], 
        vlm_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fuse CNN probabilities with VLM analysis.
        
        Args:
            cnn_probs: Dictionary of CNN class probabilities
            vlm_analysis: VLM observation dictionary with likelihoods and descriptors
            
        Returns:
            Fused analysis with final probabilities and explanations
        """
        # Normalize and validate CNN probabilities
        cnn_normalized = self._normalize_probabilities(cnn_probs)
        
        # Extract VLM likelihoods and descriptors
        vlm_probs = vlm_analysis.get("likelihoods", {})
        descriptors = vlm_analysis.get("descriptors", [])
        primary_pattern = vlm_analysis.get("primary_pattern", "other")
        
        # Normalize VLM probabilities 
        vlm_normalized = self._normalize_probabilities(vlm_probs)
        
        # Apply descriptor-based adjustments
        vlm_adjusted, descriptor_notes = self._apply_descriptor_adjustments(
            vlm_normalized, descriptors
        )
        
        # Weighted fusion
        final_probs = self._weighted_fusion(cnn_normalized, vlm_adjusted)
        
        # Generate top-k predictions
        top3 = self._get_top_predictions(final_probs, k=3)
        
        # Determine uncertainty and gate logic
        gate_result, uncertainty_notes = self._determine_gate(final_probs, top3)
        
        # Compile fusion notes
        fusion_notes = self._compile_fusion_notes(
            descriptor_notes, primary_pattern, top3, uncertainty_notes
        )
        
        return {
            "final_probs": final_probs,
            "top3": top3,
            "gate": gate_result,
            "vlm_descriptors": descriptors,
            "vlm_primary_pattern": primary_pattern,
            "fusion_notes": fusion_notes,
            "cnn_contribution": self.config["cnn_weight"],
            "vlm_contribution": self.config["vlm_weight"],
            "descriptor_adjustments": descriptor_notes
        }
    
    def _normalize_probabilities(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Normalize probabilities to sum to 1.0."""
        if not probs:
            # Equal probabilities if no data
            n_classes = len(self.class_names)
            return {name: 1.0/n_classes for name in self.class_names}
        
        # Ensure all class names are present
        normalized = {}
        for class_name in self.class_names:
            normalized[class_name] = float(probs.get(class_name, 0.0))
        
        # Normalize to sum to 1
        total = sum(normalized.values())
        if total > 0:
            normalized = {k: v/total for k, v in normalized.items()}
        else:
            # Fallback to uniform distribution
            n_classes = len(self.class_names)
            normalized = {k: 1.0/n_classes for k in normalized.keys()}
        
        return normalized
    
    def _apply_descriptor_adjustments(
        self, 
        vlm_probs: Dict[str, float], 
        descriptors: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Apply descriptor-based probability adjustments."""
        adjusted = vlm_probs.copy()
        notes = {"adjustments": [], "descriptor_matches": {}}
        
        # Convert descriptors to lowercase set for matching
        desc_set = {desc.lower().strip() for desc in descriptors}
        
        # Seborrheic keratosis boost
        sk_matches = len(desc_set & {d.lower() for d in SEBORRHEIC_KERATOSIS_DESCRIPTORS})
        if sk_matches >= 2:
            boost = self.config["sk_descriptor_bonus"]
            adjusted["seborrheic_keratosis"] += boost
            notes["adjustments"].append(f"SK boosted +{boost:.2f} ({sk_matches} descriptors)")
            notes["descriptor_matches"]["seborrheic_keratosis"] = sk_matches
            
            # Reduce melanoma when SK features are strong
            melanoma_reduction = boost * 0.3
            adjusted["melanoma"] = max(0, adjusted["melanoma"] - melanoma_reduction)
            notes["adjustments"].append(f"Melanoma reduced -{melanoma_reduction:.2f} (SK features)")
        
        # Melanoma boost  
        mel_matches = len(desc_set & {d.lower() for d in MELANOMA_DESCRIPTORS})
        if mel_matches >= 1:
            boost = self.config["melanoma_descriptor_bonus"]
            adjusted["melanoma"] += boost
            notes["adjustments"].append(f"Melanoma boosted +{boost:.2f} ({mel_matches} descriptors)")
            notes["descriptor_matches"]["melanoma"] = mel_matches
        
        # BCC boost
        bcc_matches = len(desc_set & {d.lower() for d in BCC_DESCRIPTORS})
        if bcc_matches >= 2:
            boost = self.config["bcc_descriptor_bonus"]
            adjusted["bcc"] += boost
            notes["adjustments"].append(f"BCC boosted +{boost:.2f} ({bcc_matches} descriptors)")
            notes["descriptor_matches"]["bcc"] = bcc_matches
        
        # SCC boost
        scc_matches = len(desc_set & {d.lower() for d in SCC_DESCRIPTORS})
        if scc_matches >= 2:
            boost = self.config["scc_descriptor_bonus"]
            adjusted["scc"] += boost
            notes["adjustments"].append(f"SCC boosted +{boost:.2f} ({scc_matches} descriptors)")
            notes["descriptor_matches"]["scc"] = scc_matches
        
        # Clip probabilities to [0, 1] and renormalize
        for k in adjusted:
            adjusted[k] = max(0.0, min(1.0, adjusted[k]))
        
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v/total for k, v in adjusted.items()}
        
        return adjusted, notes
    
    def _weighted_fusion(
        self, 
        cnn_probs: Dict[str, float], 
        vlm_probs: Dict[str, float]
    ) -> Dict[str, float]:
        """Perform weighted fusion of CNN and VLM probabilities."""
        cnn_weight = self.config["cnn_weight"]
        vlm_weight = self.config["vlm_weight"]
        
        # Ensure weights sum to 1
        total_weight = cnn_weight + vlm_weight
        if total_weight > 0:
            cnn_weight /= total_weight
            vlm_weight /= total_weight
        
        fused = {}
        for class_name in self.class_names:
            cnn_prob = cnn_probs.get(class_name, 0.0)
            vlm_prob = vlm_probs.get(class_name, 0.0)
            fused[class_name] = cnn_weight * cnn_prob + vlm_weight * vlm_prob
        
        return fused
    
    def _get_top_predictions(
        self, 
        probs: Dict[str, float], 
        k: int = 3
    ) -> List[Tuple[str, float]]:
        """Get top-k predictions sorted by probability."""
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:k]
    
    def _determine_gate(
        self, 
        probs: Dict[str, float], 
        top3: List[Tuple[str, float]]
    ) -> Tuple[str, List[str]]:
        """Determine clinical gate decision and uncertainty markers."""
        notes = []
        
        if len(top3) < 2:
            return "UNCERTAIN — insufficient prediction confidence", ["single_prediction"]
        
        top1_class, top1_prob = top3[0]
        top2_class, top2_prob = top3[1]
        
        # Check for tie between top 2 predictions
        prob_diff = top1_prob - top2_prob
        tie_margin = self.config["tie_margin"]
        
        if prob_diff <= tie_margin:
            notes.append(f"close_margin_{prob_diff:.3f}")
            return "UNCERTAIN — clinical correlation/dermoscopy recommended", notes
        
        # Check minimum confidence threshold
        min_conf = self.config["min_confidence_threshold"]
        if top1_prob < min_conf:
            notes.append(f"low_confidence_{top1_prob:.3f}")
            return "UNCERTAIN — low confidence, clinical evaluation recommended", notes
        
        # Generate specific gates based on top prediction
        if top1_class == "melanoma":
            if top1_prob > 0.6:
                return "RED — urgent dermatology referral for suspected melanoma", notes
            else:
                return "YELLOW — dermatology referral for pigmented lesion evaluation", notes
                
        elif top1_class == "bcc":
            if top1_prob > 0.5:
                return "ORANGE — likely BCC, consider dermatology referral", notes
            else:
                return "YELLOW — possible BCC, clinical correlation recommended", notes
                
        elif top1_class == "scc":
            if top1_prob > 0.5:
                return "ORANGE — likely SCC, dermatology referral recommended", notes
            else:
                return "YELLOW — possible SCC, clinical evaluation recommended", notes
                
        elif top1_class == "seborrheic_keratosis":
            if top1_prob > 0.6:
                return "GREEN — likely seborrheic keratosis, routine monitoring", notes
            else:
                return "YELLOW — probable seborrheic keratosis, consider dermatoscopy", notes
                
        elif top1_class == "nevus":
            if top1_prob > 0.7:
                return "GREEN — likely benign nevus, routine monitoring", notes
            else:
                return "YELLOW — atypical nevus, consider dermatoscopy", notes
        
        else:  # benign_other or unknown
            return "GREEN — likely benign, routine monitoring", notes
    
    def _compile_fusion_notes(
        self, 
        descriptor_notes: Dict[str, Any],
        primary_pattern: str,
        top3: List[Tuple[str, float]],
        uncertainty_notes: List[str]
    ) -> str:
        """Compile human-readable fusion explanation."""
        notes_parts = []
        
        # VLM primary assessment
        if primary_pattern and primary_pattern != "other":
            pattern_display = primary_pattern.replace("_", " ").title()
            notes_parts.append(f"VLM assessment: {pattern_display}")
        
        # Descriptor adjustments
        adjustments = descriptor_notes.get("adjustments", [])
        if adjustments:
            adj_summary = "; ".join(adjustments[:2])  # Limit to top 2 adjustments
            notes_parts.append(f"Descriptor adjustments: {adj_summary}")
        
        # Top prediction confidence
        if top3:
            top_class, top_prob = top3[0]
            class_display = top_class.replace("_", " ").title()
            notes_parts.append(f"Final prediction: {class_display} ({top_prob:.1%})")
        
        # Uncertainty factors
        if uncertainty_notes:
            if "close_margin" in str(uncertainty_notes):
                notes_parts.append("Close margin between top predictions")
            if "low_confidence" in str(uncertainty_notes):
                notes_parts.append("Low overall confidence")
        
        return " • ".join(notes_parts) if notes_parts else "Standard CNN-VLM fusion applied"


def fuse_probs(
    cnn: Dict[str, float], 
    vlm: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function for probability fusion.
    
    Args:
        cnn: CNN probability dictionary 
        vlm: VLM observation dictionary
        config: Optional fusion configuration
        
    Returns:
        Fused analysis dictionary
    """
    fusion_engine = NeuroSymbolicFusion(config=config)
    return fusion_engine.fuse_predictions(cnn, vlm)


def test_fusion_system():
    """Test the fusion system with sample data."""
    print("Testing Neuro-Symbolic Fusion System...")
    
    # Test case 1: SK with strong descriptors
    print("\n1. Testing SK with strong descriptors:")
    cnn_probs_sk = {
        "melanoma": 0.35,
        "bcc": 0.20,
        "scc": 0.15,
        "nevus": 0.20,
        "seborrheic_keratosis": 0.08,
        "benign_other": 0.02
    }
    
    vlm_analysis_sk = {
        "primary_pattern": "seborrheic_keratosis",
        "likelihoods": {
            "melanoma": 0.2,
            "bcc": 0.1,
            "scc": 0.1,
            "nevus": 0.1,
            "seborrheic_keratosis": 0.5
        },
        "descriptors": ["stuck-on appearance", "waxy surface", "milia-like cysts", "sharply demarcated"]
    }
    
    result_sk = fuse_probs(cnn_probs_sk, vlm_analysis_sk)
    print(f"CNN top: {max(cnn_probs_sk.items(), key=lambda x: x[1])}")
    print(f"Fused top: {result_sk['top3'][0]}")
    print(f"Gate: {result_sk['gate']}")
    print(f"Notes: {result_sk['fusion_notes']}")
    
    # Test case 2: Melanoma with concerning features
    print("\n2. Testing melanoma with concerning features:")
    cnn_probs_mel = {
        "melanoma": 0.45,
        "bcc": 0.15,
        "scc": 0.10,
        "nevus": 0.25,
        "seborrheic_keratosis": 0.03,
        "benign_other": 0.02
    }
    
    vlm_analysis_mel = {
        "primary_pattern": "melanoma",
        "likelihoods": {
            "melanoma": 0.6,
            "bcc": 0.1,
            "scc": 0.1,
            "nevus": 0.15,
            "seborrheic_keratosis": 0.05
        },
        "descriptors": ["irregular pigment network", "asymmetric pigmentation", "variegated coloring"]
    }
    
    result_mel = fuse_probs(cnn_probs_mel, vlm_analysis_mel)
    print(f"CNN top: {max(cnn_probs_mel.items(), key=lambda x: x[1])}")
    print(f"Fused top: {result_mel['top3'][0]}")
    print(f"Gate: {result_mel['gate']}")
    print(f"Notes: {result_mel['fusion_notes']}")
    
    # Test case 3: Uncertain case with close margins
    print("\n3. Testing uncertain case with close margins:")
    cnn_probs_uncertain = {
        "melanoma": 0.32,
        "bcc": 0.30,
        "scc": 0.18,
        "nevus": 0.15,
        "seborrheic_keratosis": 0.04,
        "benign_other": 0.01
    }
    
    vlm_analysis_uncertain = {
        "primary_pattern": "other",
        "likelihoods": {
            "melanoma": 0.3,
            "bcc": 0.35,
            "scc": 0.2,
            "nevus": 0.1,
            "seborrheic_keratosis": 0.05
        },
        "descriptors": ["irregular surface", "unclear borders"]
    }
    
    result_uncertain = fuse_probs(cnn_probs_uncertain, vlm_analysis_uncertain)
    print(f"CNN top: {max(cnn_probs_uncertain.items(), key=lambda x: x[1])}")
    print(f"Fused top: {result_uncertain['top3'][0]}")
    print(f"Gate: {result_uncertain['gate']}")
    print(f"Notes: {result_uncertain['fusion_notes']}")
    
    print("\nFusion system test completed.")


if __name__ == "__main__":
    test_fusion_system()
