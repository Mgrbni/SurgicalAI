"""Guideline ingestion & lookup utilities.

Loads structured evidence-based surgical guidelines from YAML for margin and reconstruction planning.
This module is intentionally verbose with comments for maintainability.

NOT FOR CLINICAL USE.
"""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, Optional
import yaml

DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'guidelines.yaml'

class GuidelineNotFound(Exception):
    pass

@lru_cache(maxsize=1)
def _load() -> Dict[str, Any]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Guidelines file missing: {DATA_PATH}")
    with DATA_PATH.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data

def list_diagnoses() -> list[str]:
    return sorted(_load().keys())

def get_guideline(dx: str) -> Dict[str, Any]:
    dx_key = dx.lower().replace(' ', '_')
    data = _load()
    if dx_key not in data:
        raise GuidelineNotFound(dx_key)
    return data[dx_key]

def summarize_for_report(prob_order: list[tuple[str,float]], top_n: int = 3) -> list[Dict[str, Any]]:
    """Return top_n guidelines joined with probabilities.

    Args:
        prob_order: list of (label, probability) already sorted desc.
    """
    out: list[Dict[str, Any]] = []
    for i,(label,p) in enumerate(prob_order[:top_n]):
        g: Dict[str, Any] | None = None
        try:
            g = get_guideline(label)
        except GuidelineNotFound:
            g = None
        out.append({
            'rank': i+1,
            'diagnosis': label,
            'probability': p,
            'summary': g.get('summary') if g else None,
            'margins': g.get('margins') if g else None,
            'reconstruction': g.get('reconstruction') if g else None,
            'notes': g.get('notes') if g else None,
        })
    return out

def choose_reconstruction(dx: str, defect_diameter_mm: Optional[float] = None, site: Optional[str] = None) -> str:
    """Heuristic selection of reconstruction line for quick suggestions.

    Prefers flap for malignant moderate-size lesions on cosmetic units.
    """
    g = get_guideline(dx)
    recon = g.get('reconstruction', {})
    if not recon:
        return 'primary_closure'
    # simple heuristics
    if dx in {'melanoma','bcc','scc'} and defect_diameter_mm and defect_diameter_mm > 8:
        if 'rotational_flap' in recon and (site and any(k in site for k in ['cheek','temple'])):
            return 'rotational_flap'
        if 'advancement_flap' in recon:
            return 'advancement_flap'
    if 'primary_closure' in recon:
        return 'primary_closure'
    # fallback to first key
    return next(iter(recon.keys()))
