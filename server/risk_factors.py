"""Risk factor intake + Bayesian probability adjustment.

We model risk factors as multiplicative likelihood ratios (simplified) applied to base class probabilities,
then renormalize. Coefficients are heuristic placeholders for demo.

NOT FOR CLINICAL USE.
"""
from __future__ import annotations
from typing import Dict, Any

# Heuristic likelihood ratios per factor per malignant / benign grouping
FACTOR_LR = {
    'smoking': 1.15,            # slight ↑ SCC risk
    'uv_extreme': 1.25,         # ↑ melanoma + NMSC
    'immunosuppressed': 1.6,    # ↑ SCC esp.
    'family_history_melanoma': 1.4,
    'prior_radiation': 1.3,
    'fitzpatrick_I_II': 1.2,
}

BENIGN_DAMPEN = 0.9  # scale benign classes when malignant LRs applied
MALIGNANT_LABELS = {'melanoma','bcc','scc'}

# Map UI form keys to internal factor codes
FORM_MAP = {
    'smoking': 'smoking',
    'uv': 'uv_extreme',
    'immunosuppression': 'immunosuppressed',
    'family_history': 'family_history_melanoma',
    'prior_radiation': 'prior_radiation',
    'fitzpatrick': 'fitzpatrick_I_II',  # only if type I/II else ignore
}

def adjust_probabilities(base: Dict[str, float], factors: Dict[str, Any]) -> Dict[str, float]:
    """Apply Bayesian-like scaling to base probabilities based on risk factors.

    Args:
        base: mapping label->prob (sums to ~1)
        factors: raw input values (booleans / categorical)
    Returns new mapping with same keys summing to 1.
    """
    scaled = base.copy()
    malignant_scale = 1.0
    benign_scale = 1.0
    # Translate factors
    for k,v in factors.items():
        if k not in FORM_MAP:
            continue
        code = FORM_MAP[k]
        if code == 'fitzpatrick_I_II':
            # Accept numeric 1/2 or string 'I','II'
            if str(v).strip().upper() not in {'1','2','I','II'}:
                continue
        if v in (False, None, '', 0):
            continue
        lr = FACTOR_LR.get(code, 1.0)
        malignant_scale *= lr
        benign_scale *= BENIGN_DAMPEN
    if malignant_scale == 1.0 and benign_scale == 1.0:
        return base
    for lbl in scaled:
        if lbl in MALIGNANT_LABELS:
            scaled[lbl] *= malignant_scale
        else:
            scaled[lbl] *= benign_scale
    # Renormalize
    total = sum(scaled.values()) or 1.0
    for lbl in scaled:
        scaled[lbl] /= total
    return scaled
