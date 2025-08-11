"""Risk factor Bayesian adjustment module.

adjust_probs(priors, factors) -> dict with posterior probabilities and per-factor deltas.
All probabilities assumed over set of diagnosis keys (e.g., melanoma, bcc, scc, nevus, seborrheic_keratosis, benign_other).
Factors is a dict of structured intake answers.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import math

# Simple likelihood ratio table (placeholder values)
# Format: factor -> value -> {diagnosis -> LR}
LR_TABLE = {
    "uv_exposure": {
        "high": {"melanoma": 1.8, "bcc": 1.6, "scc": 2.0},
        "moderate": {"melanoma": 1.2, "bcc": 1.1, "scc": 1.1},
        "low": {"melanoma": 0.8, "bcc": 0.9, "scc": 0.9},
    },
    "immunosuppression": {
        True: {"scc": 2.5, "melanoma": 1.3},
        False: {"scc": 0.9},
    },
    "smoking": {
        True: {"scc": 1.2},
    },
    "family_history_melanoma": {
        True: {"melanoma": 2.0},
    },
    "family_history_nmsc": {
        True: {"bcc": 1.4, "scc": 1.4},
    },
    "fitzpatrick": {
        1: {"melanoma": 1.3},
        2: {"melanoma": 1.2},
        5: {"melanoma": 0.8},
        6: {"melanoma": 0.6},
    },
    "evolution_change": {
        True: {"melanoma": 2.2},
        False: {"melanoma": 0.8},
    },
    "prior_biopsy": {
        True: {"melanoma": 0.7, "bcc": 0.8, "scc": 0.8},  # if already benign biopsy lowers malignant probability
    },
}

@dataclass
class AdjustmentResult:
    posteriors: Dict[str, float]
    deltas: Dict[str, Dict[str, float]]  # factor -> dx -> delta probability (posterior - prior contribution)


def _normalize(probs: Dict[str, float]) -> Dict[str, float]:
    s = sum(probs.values())
    if s <= 0:
        return {k: 0.0 for k in probs}
    return {k: v / s for k, v in probs.items()}


def adjust_probs(priors: Dict[str, float], factors: Dict[str, Any]) -> AdjustmentResult:
    """Apply naive Bayes style adjustment with multiplicative LRs.
    Args:
        priors: dict of prior probabilities summing to 1.
        factors: intake answers.
    Returns:
        AdjustmentResult with normalized posteriors and per-factor deltas.
    """
    priors = _normalize(priors)
    odds = {k: (v / (1 - v)) if 0 < v < 1 else (1e6 if v >= 1 else 1e-6) for k, v in priors.items()}
    original_odds = odds.copy()
    deltas: Dict[str, Dict[str, float]] = {}

    for factor, value in factors.items():
        table = LR_TABLE.get(factor)
        if not table:
            continue
        # pick matching value
        lr_map = table.get(value)
        if not lr_map:
            continue
        before = odds.copy()
        for dx, lr in lr_map.items():
            if dx in odds:
                odds[dx] *= lr
        # compute delta in probability space after this factor alone
        temp_probs = _normalize({k: o/(1+o) for k, o in odds.items()})
        prev_probs = _normalize({k: b/(1+b) for k, b in before.items()})
        deltas[factor] = {k: temp_probs[k] - prev_probs[k] for k in temp_probs}

    # Final conversion back to probabilities
    post_probs = {k: o/(1+o) for k, o in odds.items()}
    post_probs = _normalize(post_probs)
    return AdjustmentResult(posteriors=post_probs, deltas=deltas)

__all__ = ["adjust_probs", "AdjustmentResult", "LR_TABLE"]
