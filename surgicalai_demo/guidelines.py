"""Oncologic margins & treatment guideline engine.

Loads YAML rules from data/oncology_rules.yaml and exposes a plan() function.
All outputs are transparent dicts with citations placeholders.
Not for clinical use.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import math
import yaml

RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "oncology_rules.yaml"

@dataclass
class GuidelineResult:
    margins_mm: Optional[int | float | str]
    treatment_modality: str
    reconstruction_options: List[str]
    citations: List[str]
    defer_flap: bool = False
    notes: List[str] | None = None

_cache: Dict[str, Any] | None = None

def _load_rules() -> Dict[str, Any]:
    global _cache
    if _cache is None:
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            _cache = yaml.safe_load(f)
    return _cache


def _breslow_margin_mm(breslow_mm: Optional[float], margins: Dict[str, Any]) -> Optional[int]:
    if breslow_mm is None:
        return None
    if breslow_mm == 0:
        return int(margins.get("in_situ_mm", 5))
    if breslow_mm < 1.0:
        return int(margins.get("breslow_lt_1_0_cm", 10))
    if 1.0 <= breslow_mm <= 2.0:
        return int(margins.get("breslow_1_01_2_0_cm", 15))
    return int(margins.get("breslow_gt_2_0_cm", 20))


def _diagnosis_margin(diagnosis: str, breslow_mm: Optional[float], rules: Dict[str, Any]) -> Optional[int | str]:
    dx = rules["diagnoses"].get(diagnosis)
    if not dx:
        return None
    margins = dx.get("margins")
    if isinstance(margins, (str, int)):
        return margins
    if diagnosis == "melanoma":
        return _breslow_margin_mm(breslow_mm, margins or {})
    if diagnosis == "bcc":
        return margins.get("low_risk_mm", 4)
    if diagnosis == "scc":
        return margins.get("low_risk_mm", 6)
    return None


def _site_recon_options(site: str, diameter_mm: Optional[float], rules: Dict[str, Any]) -> List[str]:
    s = rules.get("site_rules", {}).get(site) or {}
    recon = s.get("reconstruction", {})
    if diameter_mm is None:
        return recon.get("small_defect_mm", [])
    small_t = rules.get("defaults", {}).get("small_threshold_mm", 10)
    med_t = rules.get("defaults", {}).get("medium_threshold_mm", 20)
    if diameter_mm <= small_t:
        return recon.get("small_defect_mm", [])
    if diameter_mm <= med_t:
        return recon.get("medium_defect_mm", [])
    return recon.get("large_defect_mm", [])


def plan(diagnosis: str, site: str, diameter_mm: Optional[float], borders_clear: bool, breslow_mm: Optional[float] = None) -> GuidelineResult:
    rules = _load_rules()
    dx = rules["diagnoses"].get(diagnosis, {})
    margins_val = _diagnosis_margin(diagnosis, breslow_mm, rules)
    treatment_modality = dx.get("treatment", {}).get("modality", "NONE")
    defer_flap = bool(dx.get("treatment", {}).get("defer_flap", False))
    options = _site_recon_options(site, diameter_mm, rules)
    citations = list(dict.fromkeys(dx.get("citations", []) + rules.get("defaults", {}).get("citations_placeholder", [])))
    notes = dx.get("notes")
    # melanoma margin clearance dependency
    if diagnosis == "melanoma" and not borders_clear:
        # emphasize margin clearance
        options = [f"Defer definitive reconstruction until clear margins (planned options: {', '.join(options)})"]
    return GuidelineResult(
        margins_mm=margins_val,
        treatment_modality=treatment_modality,
        reconstruction_options=options,
        citations=citations,
        defer_flap=defer_flap,
        notes=notes,
    )

__all__ = ["plan", "GuidelineResult"]
