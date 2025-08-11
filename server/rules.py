"""Oncology gate logic for multi-class classifier (demo).

Provides compute_gate(probs, topk) that applies:
- UNCERTAIN if top1 prob < min_conf_for_gate
- DEFER only if melanoma top1 AND prob >= melanoma_defer_threshold
- NMSC status if BCC/SCC top1 above confidence threshold
- OK for benign entities

Returns dict with status, allow_flap, reason, guidance, melanoma_prob.
NOT FOR CLINICAL USE.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from .settings import SETTINGS

NMSC_SET = {"bcc","scc"}

def compute_gate(probs: Dict[str,float], topk: List[Tuple[str,float]]):
    if not probs or not topk:
        return {
            "status": "UNCERTAIN",
            "allow_flap": False,
            "reason": "No probabilities available",
            "guidance": "Re-image / retry",
            "melanoma_prob": float(probs.get("melanoma",0.0)) if probs else 0.0,
        }
    top_label, top_prob = topk[0]
    melanoma_prob = float(probs.get("melanoma", 0.0))
    # Default UNCERTAIN
    status = "UNCERTAIN"
    allow = False
    reason = "Probability mass diffuse; clinical correlation."
    guidance = ""
    if top_prob >= SETTINGS.min_conf_for_gate:
        if top_label == "melanoma" and melanoma_prob >= SETTINGS.melanoma_defer_threshold:
            status = "DEFER"; allow = False
            reason = "High melanoma suspicion"
            guidance = "Biopsy / WLE before complex reconstruction"
        elif top_label in NMSC_SET:
            status = "NMSC"; allow = True
            reason = f"Likely {top_label.upper()}"
            guidance = "Consider margin-controlled excision per guidelines"
        else:
            status = "OK"; allow = True
            reason = "Low melanoma suspicion"
            guidance = "Reconstructive planning allowed"
    return {
        "status": status,
        "allow_flap": allow,
        "reason": reason,
        "guidance": guidance,
        "melanoma_prob": melanoma_prob,
    }
