# rules.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any

# ----------------------------- Constants ---------------------------------

MELANOMA_DEFER_THRESHOLD = 0.30
CRITICAL_FACE_ZONES = {"nose", "nasal_tip", "nasal_ala", "eyelid",
                       "medial_canthal", "lateral_canthal", "ear", "lip"}

# ------------------------------ Data types --------------------------------

@dataclass
class GateDecision:
    allow_flap: bool
    reason: str
    guidance: str
    citations: List[str]

@dataclass
class FlapOption:
    name: str
    rationale: str
    success_hint: str
    cautions: List[str]
    priority: int  # lower = higher rank

# ---------------------------- Gatekeeper ----------------------------------

def oncology_gate(
    probs: Dict[str, float],
    location: str,
    ill_defined_borders: bool = False,
    recurrent_tumor: bool = False,
    margin_status: Optional[str] = None,
) -> GateDecision:
    p_meln = float(probs.get("melanoma", 0.0))
    p_bcc  = float(probs.get("bcc", 0.0))
    p_scc  = float(probs.get("scc", 0.0))
    nmsc_prob = max(p_bcc, p_scc)
    high_risk_site = location in CRITICAL_FACE_ZONES

    if p_meln >= MELANOMA_DEFER_THRESHOLD and margin_status != "clear":
        return GateDecision(
            allow_flap=False,
            reason="High melanoma suspicion.",
            guidance=("Perform diagnostic excisional biopsy (1–3 mm margins). "
                      "Plan WLE by Breslow and reconstruct after margin control."),
            citations=["NCCN Melanoma: biopsy first; WLE margins by Breslow"]
        )

    if (nmsc_prob >= 0.30) and (high_risk_site or ill_defined_borders or recurrent_tumor) and margin_status != "clear":
        return GateDecision(
            allow_flap=False,
            reason="NMSC in high‑risk context.",
            guidance=("Prefer Mohs (margin‑controlled). Reconstruct after clear margins."),
            citations=["NCCN BCC/SCC: Mohs for H‑zone / ill‑defined / recurrent"]
        )

    if (nmsc_prob >= 0.30 or p_meln >= 0.10) and margin_status != "clear":
        return GateDecision(
            allow_flap=False,
            reason="Margins not confirmed.",
            guidance=("Defer complex flaps until margin status is confirmed; "
                      "temporize with linear closure or graft."),
            citations=["Oncologic sequencing: margin control before flap"]
        )

    return GateDecision(True, "Oncologic gates cleared for demo.",
                        "Proceed to local reconstruction planning.", [])

# --------------------------- Flap selection -------------------------------

def flap_selector(
    location: str,
    defect_equiv_diam_mm: Optional[float],
    depth: str = "dermal",
    laxity: str = "moderate",
) -> List[FlapOption]:
    d = float(defect_equiv_diam_mm or 0.0)
    L: List[FlapOption] = []

    if location == "cheek":
        if d <= 15 and laxity in {"moderate", "high"}:
            L.append(FlapOption(
                "V‑Y advancement (cheek)",
                "Small defect with cheek laxity; preserves subunit.",
                "Good for ≤1.5 cm; scar in RSTLs.",
                ["Avoid vertical tension near lower lid", "Protect facial artery branches"], 1))
        if 15 < d <= 30 and laxity != "low":
            L.append(FlapOption(
                "Rotation flap (lateral cheek/temple)",
                "Moderate defect; arc rotation with lateral reservoir.",
                "Generous back‑cut; wide undermining.",
                ["Vector tension laterally to avoid ectropion", "Preserve SMAS/nerve branches"], 2))
        if d > 30:
            L.append(FlapOption(
                "Cervicofacial advancement",
                "Large cheek/infraorbital defects using cervical/SMAS laxity.",
                "Sub‑SMAS; support lower lid.",
                ["Distal edge ischemia risk", "Meticulous hemostasis"], 3))

    elif location in {"nose", "nasal_tip", "nasal_dorsum"}:
        if location == "nasal_tip" and d <= 12 and depth in {"dermal", "full-thickness skin"}:
            L.append(FlapOption(
                "Bilobed flap (tip)",
                "Small–moderate tip defects; redirects tension.",
                "1st lobe ~50–60% of defect; correct pivot.",
                ["Trapdoor risk", "Respect alar support"], 1))
        elif d <= 20:
            L.append(FlapOption(
                "Dorsal nasal (Rieger) flap",
                "Dorsum/tip medium defects; good match.",
                "Pivot near medial canthus; sub‑SMAS undermining.",
                ["Glabellar fullness", "Pedicle congestion"], 2))
        else:
            L.append(FlapOption(
                "Paramedian forehead flap (staged)",
                "Large tip/ala/subunit losses; robust vascularity.",
                "Supratrochlear pedicle; staged thinning.",
                ["Multiple stages", "Forehead donor morbidity"], 3))

    elif location == "lower_eyelid":
        if d <= 10:
            L.append(FlapOption(
                "Primary closure (horizontal)",
                "Small lid defects preserving margin.",
                "Layered closure; lateral cantholysis if needed.",
                ["Ectropion if vertical tension", "Frost suture support"], 1))
        elif 10 < d <= 20:
            L.append(FlapOption(
                "Tenzel semicircular flap",
                "≈25–50% lid length defects.",
                "Lateral canthotomy/cantholysis for mobilization.",
                ["Lid malposition risk", "Precise canthal reattachment"], 2))

    elif location == "nasolabial":
        if d <= 20:
            L.append(FlapOption(
                "Nasolabial advancement/rotation",
                "Subunit match; hides scar in fold.",
                "Design along fold; deep SMAS anchoring.",
                ["Bulky if over‑advanced", "Beware alar distortion"], 1))

    elif location == "forehead":
        if d <= 25:
            L.append(FlapOption(
                "Rotation/advancement (forehead)",
                "Good laxity and RSTL alignment.",
                "Curvilinear incision; galeal release if needed.",
                ["Protect frontal branch", "Mind hairline shift"], 1))

    if not L:
        L.append(FlapOption(
            "Consider full‑thickness skin graft or linear closure",
            "No strong local flap match for given metrics.",
            "Postauricular/supraclavicular donor for match.",
            ["Contour mismatch", "Contraction risk"], 99))

    L.sort(key=lambda x: x.priority)
    return L

# ------------------------- High-level planner -----------------------------

def plan_reconstruction(
    probs: Dict[str, float],
    location: str,
    defect_equiv_diam_mm: Optional[float],
    depth: str,
    laxity: str,
    ill_defined_borders: bool = False,
    recurrent_tumor: bool = False,
    margin_status: Optional[str] = None,
) -> Dict:
    gate = oncology_gate(
        probs=probs,
        location=location,
        ill_defined_borders=ill_defined_borders,
        recurrent_tumor=recurrent_tumor,
        margin_status=margin_status,
    )
    if not gate.allow_flap:
        return {"indicated": False, "reason": gate.reason, "guidance": gate.guidance,
                "citations": gate.citations, "options": []}

    options = flap_selector(location, defect_equiv_diam_mm, depth, laxity)
    return {"indicated": True, "reason": gate.reason, "guidance": gate.guidance,
            "citations": gate.citations, "options": [o.__dict__ for o in options]}

# ----------------------------- Triage (Tier‑0) ----------------------------
# Accepts EITHER probs dict OR ABCDE dataclass/dict plus optional age/body_site.

def triage_tier0(abcde_or_probs: Any, age: Optional[int] = None, body_site: Optional[str] = None) -> str:
    # Path 1: probability-based (backward compatible)
    if isinstance(abcde_or_probs, dict) and {"melanoma", "bcc", "scc"} & set(abcde_or_probs.keys()):
        p_meln = float(abcde_or_probs.get("melanoma", 0.0))
        p_bcc  = float(abcde_or_probs.get("bcc", 0.0))
        p_scc  = float(abcde_or_probs.get("scc", 0.0))
        nmsc   = max(p_bcc, p_scc)
        if p_meln >= 0.30 or nmsc >= 0.60:
            return "RED FLAG"
        if p_meln >= 0.15 or nmsc >= 0.30:
            return "YELLOW (evaluate/biopsy)"
        return "GREEN (low risk by model)"

    # Path 2: ABCDE-based heuristic (for demo safety)
    # Grab attributes whether it's a dataclass or dict
    getv = (lambda k: getattr(abcde_or_probs, k) if hasattr(abcde_or_probs, k)
            else (abcde_or_probs.get(k) if isinstance(abcde_or_probs, dict) else None))

    A = float(getv("asymmetry") or 0.0)
    B = float(getv("border_irregularity") or 0.0)
    C = float(getv("color_variegation") or 0.0)
    E = float(getv("elevation_satellite") or 0.0)

    score = 0.0
    if A > 0.35: score += 1.0
    if B > 1.80: score += 1.0
    if C >= 2.0: score += 1.0
    if E > 2.0:  score += 0.5
    if (age or 0) >= 60: score += 0.5
    if (body_site or "") in CRITICAL_FACE_ZONES: score += 0.5

    if score >= 2.5:
        return "RED FLAG"
    if score >= 1.5:
        return "YELLOW (evaluate/biopsy)"
    return "GREEN (low risk by ABCDE)"
