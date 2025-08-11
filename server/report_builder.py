"""Enhanced report builder stacking on existing server.report.

Generates rich context data for PDF including:
- Risk factors
- Top differential guidelines
- Reconstruction recommendation
- Langer line overlay & incision suggestion

It delegates final PDF rendering to server.report.build_report for backward compatibility.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from . import report as base_report
from . import guidelines


def assemble_context(run_id: str,
                     run_dir: Path,
                     probs: Dict[str,float],
                     gate: Dict[str,Any],
                     risk_factors: Dict[str,Any],
                     lesion_center: Optional[Tuple[int,int]] = None,
                     incision_angle: Optional[float] = None) -> Dict[str, Any]:
    # Sort probabilities
    ordered = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top_guidelines = guidelines.summarize_for_report(ordered, top_n=3)
    primary_dx = ordered[0][0] if ordered else None
    recon_choice_key = guidelines.choose_reconstruction(primary_dx, site=risk_factors.get('site')) if primary_dx else None
    recon_text = None
    if primary_dx:
        try:
            g = guidelines.get_guideline(primary_dx)
            recon_text = g.get('reconstruction', {}).get(recon_choice_key)
        except Exception:
            pass
    return {
        'run_id': run_id,
        'primary_diagnosis': primary_dx,
        'ordered_probabilities': ordered,
        'guidelines': top_guidelines,
        'reconstruction_choice': recon_choice_key,
        'reconstruction_text': recon_text,
        'risk_factors': risk_factors,
        'gate': gate,
        'lesion_center': lesion_center,
        'incision_angle': incision_angle,
    }


def build_enhanced_pdf(run_id: str,
                       run_dir: Path,
                       probs: Dict[str,float],
                       gate: Dict[str,Any],
                       original_path: Path,
                       heatmap_path: Path,
                       overlay_path: Path,
                       risk_factors: Dict[str,Any],
                       lesion_center: Optional[Tuple[int,int]] = None,
                       incision_angle: Optional[float] = None) -> Path:
    ctx = assemble_context(run_id, run_dir, probs, gate, risk_factors, lesion_center, incision_angle)
    # Persist structured context for downstream usage (docx export etc.)
    (run_dir / 'context_extended.json').write_text(__import__('json').dumps(ctx, indent=2))
    return base_report.build_report(run_id, run_dir, probs, gate, original_path, heatmap_path, overlay_path, metrics=ctx, produce_pdf=True)  # reuse existing builder
