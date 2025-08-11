# rules.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any

# ----------------------------- Constants ---------------------------------

MELANOMA_DEFER_THRESHOLD: float = 0.30  # defer complex flap if melanoma ≥ 30% and margins not clear

# “H‑zone” + other critical cosmetic/functional regions
CRITICAL_FACE_ZONES = {
    "nose", "nasal_tip", "nasal_dorsum", "nasal_ala",
    "eyelid", "lower_eyelid", "upper_eyelid",
    "medial_canthal", "lateral_canthal",
    "ear", "lip", "vermilion"
}

# Location aliases to keep inputs forgiving
_LOCATION_NORMALIZER = {
    "tip": "nasal_tip",
    "dorsum": "nasal_dorsum",
    "ala": "nasal_ala",
    "lowerlid": "lower_eyelid",
    "upperlid": "upper_eyelid",
    "canthus_medial": "medial_canthal",
    "canthus_lateral": "lateral_canthal",
    "nasolabial_fold": "nasolabial",
}

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

# ------------------------------ Helpers -----------------------------------

def _norm_loc(location: Optional[str]) -> str:
    """Normalize free‑text locations to a canonical key."""
    if not location:
        return ""
    key = location.strip().lower().replace(" ", "_")
    return _LOCATION_NORMALIZER.get(key, key)

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

# ---------------------------- Gatekeeper ----------------------------------

def oncology_gate(
    probs: Dict[str, float],
    location: str,
    ill_defined_borders: bool = False,
    recurrent_tumor: bool = False,
    margin_status: Optional[str] = None,  # "clear", "unknown", "positive"
) -> GateDecision:
    """
    Conservative oncologic gates before planning complex local flaps.
    Returns a GateDecision. No attribute access on strings anywhere.
    """
    loc = _norm_loc(location)

    p_meln = _safe_float(probs.get("melanoma"))
    p_bcc  = _safe_float(probs.get("bcc"))
    p_scc  = _safe_float(probs.get("scc"))
    nmsc_prob = max(p_bcc, p_scc)
    high_risk_site = loc in CRITICAL_FACE_ZONES

    # 1) Melanoma suspicion: biopsy / WLE first unless margins are already clear.
    if p_meln >= MELANOMA_DEFER_THRESHOLD and margin_status != "clear":
        return GateDecision(
            allow_flap=False,
            reason="High melanoma suspicion.",
            guidance=(
                "Perform diagnostic excisional biopsy (1–3 mm margins). "
                "Plan wide local excision per Breslow and reconstruct after margin control."
            ),
            citations=["NCCN Melanoma—biopsy then WLE; reconstruct after clear margins"],
        )

    # 2) NMSC in high‑risk context → Mohs/CCPDMA preferred first.
    if (nmsc_prob >= 0.30) and (high_risk_site or ill_defined_borders or recurrent_tumor) and margin_status != "clear":
        return GateDecision(
            allow_flap=False,
            reason="Non‑melanoma skin cancer with high‑risk features/site.",
            guidance=("Prefer Mohs (margin‑controlled) or staged excision. Reconstruct after clear margins."),
            citations=["NCCN BCC/SCC—H‑zone / ill‑defined / recurrent → Mohs/CCPDMA"],
        )

    # 3) Any meaningful malignancy probability without known clear margins → defer.
    if (nmsc_prob >= 0.30 or p_meln >= 0.10) and margin_status != "clear":
        return GateDecision(
            allow_flap=False,
            reason="Margins not confirmed.",
            guidance=("Temporize (linear/graft) and finalize reconstruction after margin control."),
            citations=["Oncologic reconstruction sequencing—margin control first"],
        )

    return GateDecision(
        allow_flap=True,
        reason="Oncologic gates cleared for demo.",
        guidance="Proceed to local reconstruction planning.",
        citations=[],
    )

# --------------------------- Flap selection -------------------------------

def flap_selector(
    location: str,
    defect_equiv_diam_mm: Optional[float],
    depth: str = "dermal",
    laxity: str = "moderate",
) -> List[FlapOption]:
    """
    Heuristic flap suggestions by site/size/laxity for demo.
    """
    loc = _norm_loc(location)
    d = _safe_float(defect_equiv_diam_mm)
    L: List[FlapOption] = []

    if loc == "cheek":
        if d <= 15 and laxity in {"moderate", "high"}:
            L.append(FlapOption(
                "V‑Y advancement (cheek)",
                "Small defect with cheek laxity; preserves subunit.",
                "Good for ≤1.5 cm; align along RSTLs.",
                ["Avoid vertical tension near lower lid", "Protect facial artery branches"],
                1,
            ))
        if 15 < d <= 30 and laxity != "low":
            L.append(FlapOption(
                "Rotation flap (lateral cheek/temple)",
                "Moderate defect; arc rotation with lateral tissue reservoir.",
                "Use generous back‑cut and wide undermining.",
                ["Vector tension laterally to avoid ectropion", "Preserve SMAS/nerve branches"],
                2,
            ))
        if d > 30:
            L.append(FlapOption(
                "Cervicofacial advancement",
                "Large cheek/infraorbital defects using cervical/SMAS laxity.",
                "Sub‑SMAS plane; support lower lid.",
                ["Distal edge ischemia risk", "Meticulous hemostasis"],
                3,
            ))

    elif loc in {"nose", "nasal_tip", "nasal_dorsum"}:
        if loc == "nasal_tip" and d <= 12 and depth in {"dermal", "full-thickness_skin", "full-thickness skin"}:
            L.append(FlapOption(
                "Bilobed flap (tip)",
                "Small–moderate tip defects; redirects tension.",
                "First lobe ~50–60% of defect; correct pivot distance.",
                ["Trapdoor risk", "Respect alar support"],
                1,
            ))
        elif d <= 20:
            L.append(FlapOption(
                "Dorsal nasal (Rieger) flap",
                "Dorsum/tip medium defects; good color/texture match.",
                "Pivot near medial canthus; sub‑SMAS undermining.",
                ["Glabellar fullness", "Pedicle congestion"],
                2,
            ))
        else:
            L.append(FlapOption(
                "Paramedian forehead flap (staged)",
                "Large tip/ala/subunit losses; robust vascularity.",
                "Supratrochlear pedicle; staged thinning.",
                ["Multiple stages", "Forehead donor morbidity"],
                3,
            ))

    elif loc == "lower_eyelid":
        if d <= 10:
            L.append(FlapOption(
                "Primary closure (horizontal)",
                "Small lid defects preserving margin.",
                "Layered closure; consider lateral cantholysis.",
                ["Ectropion with vertical tension", "Frost suture support"],
                1,
            ))
        elif 10 < d <= 20:
            L.append(FlapOption(
                "Tenzel semicircular flap",
                "≈25–50% lid length defects.",
                "Lateral canthotomy/cantholysis for mobilization.",
                ["Lid malposition risk", "Precise canthal reattachment"],
                2,
            ))

    elif loc == "nasolabial":
        if d <= 20:
            L.append(FlapOption(
                "Nasolabial advancement/rotation",
                "Excellent subunit match; hides scar in fold.",
                "Design along fold; deep SMAS anchoring.",
                ["Bulky if over‑advanced", "Beware alar distortion"],
                1,
            ))

    elif loc == "forehead":
        if d <= 25:
            L.append(FlapOption(
                "Rotation/advancement (forehead)",
                "Good laxity and RSTL alignment.",
                "Curvilinear incision; galeal release if needed.",
                ["Protect frontal branch", "Mind hairline shift"],
                1,
            ))

    if not L:
        L.append(FlapOption(
            "Consider full‑thickness skin graft or linear closure",
            "No strong local flap match for given metrics.",
            "Postauricular/supraclavicular donor for color/texture match.",
            ["Contour mismatch", "Contraction risk"],
            99,
        ))

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
) -> Dict[str, Any]:
    """
    Full planning wrapper: runs oncologic gates then flap selection.
    Returns only JSON‑serializable primitives (no dataclass objects).
    """
    gate = oncology_gate(
        probs=probs,
        location=location,
        ill_defined_borders=ill_defined_borders,
        recurrent_tumor=recurrent_tumor,
        margin_status=margin_status,
    )

    if not gate.allow_flap:
        return {
            "indicated": False,
            "reason": gate.reason,
            "guidance": gate.guidance,
            "citations": list(gate.citations),
            "options": [],
        }

    options = [asdict(o) for o in flap_selector(location, defect_equiv_diam_mm, depth, laxity)]
    return {
        "indicated": True,
        "reason": gate.reason,
        "guidance": gate.guidance,
        "citations": list(gate.citations),
        "options": options,
    }

# ----------------------------- Triage (Tier‑0) ----------------------------
# Accepts EITHER a {melanoma,bcc,scc} probs dict OR an ABCDE‑style object/dict.
# Returns a simple string label to avoid the previous `.label` attribute error.

def triage_tier0(abcde_or_probs: Any, age: Optional[int] = None, body_site: Optional[str] = None) -> str:
    """
    Tier‑0 safety triage:
    - If given probs dict → thresholding.
    - Else if given ABCDE fields → heuristic.
    Always returns a plain STRING label ("RED FLAG" / "YELLOW ..." / "GREEN ...").
    """
    # Path 1: probability‑based
    if isinstance(abcde_or_probs, dict) and {"melanoma", "bcc", "scc"} & set(abcde_or_probs.keys()):
        p_meln = _safe_float(abcde_or_probs.get("melanoma"))
        p_bcc  = _safe_float(abcde_or_probs.get("bcc"))
        p_scc  = _safe_float(abcde_or_probs.get("scc"))
        nmsc   = max(p_bcc, p_scc)

        if p_meln >= 0.30 or nmsc >= 0.60:
            return "RED FLAG"
        if p_meln >= 0.15 or nmsc >= 0.30:
            return "YELLOW (evaluate/biopsy)"
        return "GREEN (low risk by model)"

    # Path 2: ABCDE‑based heuristic
    def getv(k: str) -> Any:
        if hasattr(abcde_or_probs, k):
            return getattr(abcde_or_probs, k)
        if isinstance(abcde_or_probs, dict):
            return abcde_or_probs.get(k)
        return None

    A = _safe_float(getv("asymmetry"))
    B = _safe_float(getv("border_irregularity"))
    C = _safe_float(getv("color_variegation"))
    E = _safe_float(getv("elevation_satellite"))

    score = 0.0
    if A > 0.35: score += 1.0
    if B > 1.80: score += 1.0
    if C >= 2.0: score += 1.0
    if E > 2.0:  score += 0.5
    if (age or 0) >= 60: score += 0.5
    if _norm_loc(body_site) in CRITICAL_FACE_ZONES: score += 0.5

    if score >= 2.5:
        return "RED FLAG"
    if score >= 1.5:
        return "YELLOW (evaluate/biopsy)"
    return "GREEN (low risk by ABCDE)"
