from __future__ import annotations
from typing import Dict, Any

def fallback_summarize_case(m: Dict[str, Any]) -> Dict[str, Any]:
    # Read metrics
    A = float(m.get("asymmetry", 0))
    B = float(m.get("border_irregularity", 1.0))
    C = float(m.get("color_variegation", 0))
    D = float(m.get("diameter_px", 0))
    E = float(m.get("elevation_satellite", 0))
    age = m.get("age")
    site = (m.get("body_site") or "").lower()

    # crude risk score (mirrors rules.py)
    score = 1.4*A + 1.2*B + 1.0*C + 0.8*(D/100.0) + 0.6*E
    if age and age >= 60: score *= 1.1
    if site in {"cheek","face","scalp"}: score *= 1.1

    # probability allocation (simple calibrated softmax-ish)
    # candidates limited for demo; extend later
    bases = {
        "Melanoma": 0.5*score + 0.2*(C>=2) + 0.2*(D>=60) + max(0.0, B-1.2),
        "Nevus": 0.8 - 0.4*(score>1.6),
        "Seborrheic keratosis": 0.6 - 0.2*(C>=2),
        "BCC": 0.4 + 0.2*(B>=1.4) + 0.2*(D>=50)
    }
    # clamp negatives, normalize to 100
    vals = {k:max(0.01,v) for k,v in bases.items()}
    s = sum(vals.values())
    probs = {k: round(100.0*v/s, 1) for k,v in vals.items()}

    # triage label
    tri = "BENIGN-LEANING"
    if score >= 2.2: tri = "RED FLAG"
    elif score >= 1.1: tri = "ATYPICAL"

    # flap suggestion (demo logic)
    flap = {
        "indicated": False,
        "type": "None",
        "design_summary": "Observation / biopsy-first; no flap indicated for demo fallback.",
        "tension_lines": "Align closure along local Langer’s lines.",
        "rotation_vector": "N/A",
        "predicted_success_pct": 85.0,
        "key_risks": "Pigment recurrence; margin positivity; tension causing edge necrosis."
    }
    if tri == "RED FLAG" and site in {"cheek","temple","nose","perioral","face"}:
        flap = {
            "indicated": True,
            "type": "Rotation flap",
            "design_summary": "After margin clearance, curvilinear rotation from lateral cheek; broad base to preserve perfusion.",
            "tension_lines": "Parallel to relaxed skin tension lines of midface.",
            "rotation_vector": "Lateral → medial arc (90–120°)",
            "predicted_success_pct": 88.0 if D < 120 else 78.0,
            "key_risks": "Distal epidermolysis; dog‑ear at pivot; injury to facial artery/nerve branches—plan undermining in sub‑SMAS plane."
        }
    elif tri in {"ATYPICAL"} and site in {"nose","perinasal"}:
        flap = {
            "indicated": True,
            "type": "Bilobed flap",
            "design_summary": "Small bilobed transposition respecting nasal subunits for contour match.",
            "tension_lines": "First lobe along alar crease.",
            "rotation_vector": "Lateral‑inferior to defect",
            "predicted_success_pct": 85.0,
            "key_risks": "Trapdoor deformity; alar retraction."
        }

    return {
        "triage_label": tri,
        "diagnosis_candidates": [{"name":k, "probability_pct":v} for k,v in sorted(probs.items(), key=lambda x:-x[1])],
        "flap_plan": flap,
        "notes": "Fallback generator used (offline). Biopsy remains required before reconstruction planning."
    }
