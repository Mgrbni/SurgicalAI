from __future__ import annotations
from dataclasses import dataclass
from .features import ABCDE

@dataclass
class TriageResult:
    label: str          # BENIGN-LEANING / ATYPICAL / RED FLAG
    risk_pct: float     # 0-100 (heuristic for Tier-0)
    rationale: str

def triage_tier0(abcde: ABCDE, age: int|None, body_site: str|None) -> TriageResult:
    # Baseline thresholds (tweakable)
    A, B, C, D, E = abcde.asymmetry, abcde.border_irregularity, abcde.color_variegation, abcde.diameter_px, abcde.elevation_satellite
    site = (body_site or "").lower()
    # Site/age priors (very crude Tier‑0 nudges)
    b_mult = 1.0 + (0.1 if site in {"cheek","face","scalp"} else 0.0)
    a_mult = 1.0 + (0.1 if (age or 0) >= 60 else 0.0)

    score = (
        1.4*A +
        1.2*B*b_mult +
        1.0*C +
        0.8*(D/100.0) +   # scale diameter to ~0-2 range
        0.6*E
    ) * a_mult

    # Map to categories
    if score >= 2.2:
        label = "RED FLAG"
    elif score >= 1.1:
        label = "ATYPICAL"
    else:
        label = "BENIGN-LEANING"

    # Heuristic risk %
    risk = max(0.0, min(100.0, 20 + 30*A + 20*(B-1.0) + 10*C + 0.2*D + 5*E))
    rationale = f"A={A:.2f}, B={B:.2f}, C={C:.2f}, D={D:.1f}px, E={E:.2f}, age={age}, site={body_site}"
    return TriageResult(label, float(risk), rationale)
