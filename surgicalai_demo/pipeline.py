from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

from .config import SETTINGS
from .features import (
    _read_rgb,
    compute_abcde,
    save_overlay,
    dump_metrics,
    estimate_scale,
    defect_metrics_px_to_mm,
)
from .rules import plan_reconstruction, triage_tier0
from .retrieval import SimpleAnn, feature_vector_from_abcde, save_thumbnails
from .report import make_pdf

# Import new ensemble classifier
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from surgicalai.vision.ensemble import EnsembleClassifier
    from surgicalai.vision.landmarks import FacialLandmarkDetector
    from surgicalai.geometry.rstl import RSTLAnalyzer
    from surgicalai.viz.rstl_overlay import RSTLOverlayRenderer
    from surgicalai.cognition.guidelines import query_rules, calculate_margins
    from surgicalai.geometry.flap_planning import AdvancedFlapPlanner
    ENSEMBLE_AVAILABLE = True
except ImportError as e:
    print(f"Advanced modules not available: {e}")
    ENSEMBLE_AVAILABLE = False


# -------------------------- utility helpers --------------------------

def _coalesce_lesion_image(provided: Optional[Path]) -> Path:
    """Pick a lesion image: prefer provided; else try SETTINGS or ./data."""
    if provided and provided.exists():
        return provided

    # Try a sample from SETTINGS if present
    sample = None
    try:
        sample = Path(getattr(SETTINGS, "sample_image", "")) if getattr(SETTINGS, "sample_image", "") else None
    except Exception:
        sample = None

    if sample and sample.exists():
        return sample

    # Fallback: first jpg/png under data/
    for root in ("data", "assets", "."):
        p = Path(root)
        if p.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                picks = list(p.rglob(ext))
                if picks:
                    return picks[0]

    raise FileNotFoundError("No lesion image found. Pass --image PATH or place a jpg/png under ./data")


def _prompt_optional(prompt: str) -> str:
    try:
        return input(prompt) or ""
    except Exception:
        return ""


def _safe_int(x: str, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


# ------------------------------- core --------------------------------

def run_demo(
    face_img_path: Optional[Path],
    lesion_img_path: Optional[Path],
    out_dir: Path,
    ask: bool = False,
) -> Dict[str, Any]:
    """
    End‑to‑end offline demo:
      - load lesion image
      - compute ABCDE features (+ simple heatmap if available)
      - Tier‑0 triage (string label)
      - optional retrieval of neighbors
      - oncologic gate + flap suggestions
      - write overlay.png, ai_summary.json, report.pdf
    Returns the JSON summary as a dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load lesion image
    lesion_path = _coalesce_lesion_image(lesion_img_path)
    img_rgb = _read_rgb(lesion_path)

    # 2) Optional metadata from user
    meta: Dict[str, Any] = {
        "age": None,
        "sex": "",
        "body_site": "",
        "photo_type": "",
        "known_scale_mm_per_px": None,
    }

    if ask:
        print("\nEnter optional patient/photo info (press Enter to skip):")
        meta["age"] = _safe_int(_prompt_optional("  Age (years): ").strip(), None)
        meta["sex"] = (_prompt_optional("  Sex (male/female/other): ").strip() or "").lower()
        meta["body_site"] = (_prompt_optional("  Body site (e.g., cheek, scalp, trunk): ").strip() or "").lower().replace(" ", "_")
        meta["photo_type"] = (_prompt_optional("  Photo type (dermoscopy / clinical): ").strip() or "").lower()
        meta["known_scale_mm_per_px"] = _safe_float(_prompt_optional("  Known scale (mm per pixel, e.g., 0.02) or blank: ").strip(), None)

    # 3) ABCDE features + basic lesion metrics (pixels)
    abcde = compute_abcde(img_rgb)  # now returns dict
    # Ensure stable keys exist
    abcde.setdefault("diameter_px", abcde.get("diameter_px", None))
    abcde.setdefault("center_xy", abcde.get("center_xy", None))
    heatmap = abcde.get("heatmap")  # may be None

    # 4) Scale estimation → mm conversion (best effort)
    mm_per_px = meta["known_scale_mm_per_px"]
    if mm_per_px is None:
        try:
            mm_per_px = estimate_scale(img_rgb)
        except Exception:
            mm_per_px = None

    metrics_px = {
        "diameter_px": abcde.get("diameter_px"),
        "center_xy": abcde.get("center_xy"),
    }
    metrics_mm = defect_metrics_px_to_mm(metrics_px, mm_per_px) if mm_per_px else None

    # 4.5) RSTL Analysis and Facial Landmark Detection (if available)
    rstl_analysis = None
    if ENSEMBLE_AVAILABLE:
        try:
            # Detect facial landmarks
            landmark_detector = FacialLandmarkDetector()
            landmarks = landmark_detector.detect_landmarks(img_rgb)
            
            if landmarks:
                # Estimate head pose
                head_pose = landmark_detector.estimate_head_pose(landmarks, img_rgb.shape)
                
                # Analyze RSTL
                rstl_analyzer = RSTLAnalyzer()
                aligned_rstl_lines = rstl_analyzer.align_rstl_to_face(landmarks, img_rgb.shape, head_pose)
                
                # Calculate optimal excision alignment if lesion detected
                lesion_center = abcde.get("center_xy", (img_rgb.shape[1]//2, img_rgb.shape[0]//2))
                lesion_size = (abcde.get("diameter_px", 20), abcde.get("diameter_px", 20))
                
                excision_alignment = rstl_analyzer.calculate_optimal_excision_alignment(
                    lesion_center, aligned_rstl_lines, lesion_size
                )
                
                rstl_analysis = {
                    "landmarks": landmarks,
                    "head_pose": head_pose,
                    "rstl_lines": aligned_rstl_lines,
                    "excision_alignment": excision_alignment
                }
        except Exception as e:
            print(f"RSTL analysis failed: {e}")
            rstl_analysis = None

    # 5) Enhanced diagnosis using ensemble classifier
    if ENSEMBLE_AVAILABLE:
        try:
            ensemble = EnsembleClassifier()
            # Prepare features for ensemble
            ensemble_features = {
                "asymmetry": abcde.get("asymmetry", 0.0),
                "border_irregularity": abcde.get("border_irregularity", 0.0),
                "color_variegation": abcde.get("color_variegation", 0.0),
                "diameter_mm": (metrics_mm or {}).get("diameter_mm", 0.0),
                "elevation_satellite": abcde.get("elevation_satellite", 0.0),
                **{k: v for k, v in abcde.items() if isinstance(v, (int, float))}
            }
            
            ensemble_result = ensemble.predict(ensemble_features)
            
            # Extract top diagnoses probabilities for planning
            probs = {}
            for diag_result in ensemble_result.top_diagnoses:
                if diag_result.diagnosis in ["melanoma", "bcc", "scc"]:
                    probs[diag_result.diagnosis] = diag_result.probability
            
            # Use ensemble triage if available
            top_diagnosis = ensemble_result.top_diagnoses[0].diagnosis if ensemble_result.top_diagnoses else "unknown"
            tri_label = f"ENSEMBLE: {top_diagnosis} ({ensemble_result.top_diagnoses[0].probability:.2f})" if ensemble_result.top_diagnoses else "ENSEMBLE: uncertain"
            
            if ensemble_result.high_risk_flag:
                tri_label = "RED FLAG (ENSEMBLE)"
                
        except Exception as e:
            print(f"Ensemble classifier failed: {e}")
            # Fallback to original triage
            ensemble_result = None
            tri_label = triage_tier0(abcde, age=meta["age"], body_site=meta["body_site"])
            probs = {
                "melanoma": float(abcde.get("melanoma_prob", 0.0) or 0.0),
                "bcc": float(abcde.get("bcc_prob", 0.0) or 0.0),
                "scc": float(abcde.get("scc_prob", 0.0) or 0.0),
            }
    else:
        # Fallback to original triage system
        ensemble_result = None
        tri_label = triage_tier0(abcde, age=meta["age"], body_site=meta["body_site"])
        probs = {
            "melanoma": float(abcde.get("melanoma_prob", 0.0) or 0.0),
            "bcc": float(abcde.get("bcc_prob", 0.0) or 0.0),
            "scc": float(abcde.get("scc_prob", 0.0) or 0.0),
        }

    # 7) Enhanced Planning with Guideline-based Margins and Advanced Flap Planning
    enhanced_plan = None
    if ENSEMBLE_AVAILABLE and ensemble_result:
        try:
            # Get top diagnosis for guideline lookup
            top_diagnosis = ensemble_result.top_diagnoses[0].diagnosis if ensemble_result.top_diagnoses else "unknown"
            
            # Query comprehensive guidelines
            if top_diagnosis in ["melanoma", "bcc", "cscc", "nevus", "seb_keratosis"]:
                lesion_diameter_mm = (metrics_mm or {}).get("diameter_mm", 10.0)
                
                guidelines = query_rules(
                    diagnosis=top_diagnosis,
                    subunit=meta["body_site"] or "cheek",
                    lesion_diameter_mm=lesion_diameter_mm,
                    breslow_depth_mm=None,  # Would need histology
                    risk_factors=["ill_defined_borders"] if abcde.get("border_irregularity", 0) > 1.8 else []
                )
                
                # Advanced flap planning
                flap_planner = AdvancedFlapPlanner()
                
                lesion_center = abcde.get("center_xy", (img_rgb.shape[1]//2, img_rgb.shape[0]//2))
                lesion_dims = (lesion_diameter_mm or 10.0, lesion_diameter_mm or 10.0)
                
                defect_analysis = flap_planner.analyze_defect(
                    center=lesion_center,
                    dimensions=lesion_dims,
                    location=meta["body_site"] or "cheek",
                    depth="full_thickness"
                )
                
                flap_plan = flap_planner.plan_reconstruction(
                    defect=defect_analysis,
                    tissue_laxity="moderate",
                    patient_age=meta.get("age"),
                    smoking_history=False  # Would need patient history
                )
                
                enhanced_plan = {
                    "guidelines": guidelines,
                    "flap_plan": flap_plan,
                    "margin_calculation": guidelines.margin_calculation,
                    "citations": guidelines.citations
                }
        except Exception as e:
            print(f"Enhanced planning failed: {e}")
            enhanced_plan = None
    
    # 8) Original planning (fallback)
    plan = plan_reconstruction(
        probs=probs,
        location=meta["body_site"] or "cheek",
        defect_equiv_diam_mm=(metrics_mm or {}).get("diameter_mm"),
        depth="dermal",
        laxity="moderate",
        ill_defined_borders=bool(abcde.get("border_irregularity", 0) and abcde.get("border_irregularity", 0) > 1.8),
        recurrent_tumor=False,
        margin_status=None,
    )

    # 8) Retrieval (optional; non‑fatal if assets missing)
    # 8) Retrieval (optional; non‑fatal if assets missing)
    neighbors_info: Dict[str, Any] = {}
    try:
        ann = SimpleAnn()
        vec = feature_vector_from_abcde(abcde)
        hits = ann.search(vec, k=6)
        thumbs = save_thumbnails(hits, out_dir)
        neighbors_info = {"neighbors": hits, "thumbnails": thumbs}
    except Exception:
        neighbors_info = {"neighbors": [], "thumbnails": []}

    # 9) Enhanced Visual overlay with RSTL if available
    try:
        overlay_path = out_dir / "overlay.png"
        
        if ENSEMBLE_AVAILABLE and rstl_analysis:
            # Create enhanced overlay with RSTL
            renderer = RSTLOverlayRenderer()
            
            # First create standard lesion overlay
            lesion_overlay = save_overlay(img_rgb, heatmap, overlay_path)
            
            # Then add RSTL overlay
            rstl_overlay = renderer.render_rstl_overlay_raster(
                img_rgb, rstl_analysis["rstl_lines"], opacity=0.4
            )
            
            # Add excision alignment if available
            if rstl_analysis.get("excision_alignment"):
                rstl_overlay = renderer.render_excision_alignment(
                    rstl_overlay, rstl_analysis["excision_alignment"]
                )
            
            # Save enhanced overlay
            from PIL import Image
            Image.fromarray(rstl_overlay).save(overlay_path)
            
            # Also create SVG overlay
            svg_path = out_dir / "rstl_overlay.svg"
            renderer.render_rstl_overlay_svg(
                img_rgb.shape, rstl_analysis["rstl_lines"], svg_path
            )
        else:
            # Standard overlay
            save_overlay(img_rgb, heatmap, overlay_path)
            
    except Exception as e:
        print(f"Overlay generation failed: {e}")
        overlay_path = None

    # 10) Dump metrics + AI summary JSON
    dump_metrics(
        out_dir / "metrics.json",
        img_path=str(lesion_path),
        mm_per_px=mm_per_px,
        metrics_px=metrics_px,
        metrics_mm=metrics_mm,
        abcde=abcde,
    )

    summary: Dict[str, Any] = {
        "triage": {
            "label": str(tri_label),
        },
        "probs": probs,
        "plan": plan,
        "meta": meta,
        "paths": {
            "overlay_png": str(overlay_path) if overlay_path else None,
            "lesion_image": str(lesion_path),
        },
        **neighbors_info,
    }
    
    # Add ensemble results to summary if available
    if ensemble_result:
        summary["ensemble_diagnosis"] = {
            "top_diagnoses": [
                {
                    "diagnosis": d.diagnosis,
                    "probability": d.probability,
                    "confidence_interval": d.confidence_interval,
                    "confidence_score": d.confidence_score
                }
                for d in ensemble_result.top_diagnoses
            ],
            "uncertainty_score": ensemble_result.uncertainty_score,
            "high_risk_flag": ensemble_result.high_risk_flag,
            "feature_explanations": ensemble_result.feature_explanations
        }
    
    # Add RSTL analysis results if available
    if rstl_analysis:
        summary["rstl_analysis"] = {
            "landmarks_detected": True,
            "landmark_confidence": rstl_analysis["landmarks"].confidence,
            "head_pose": {
                "yaw": rstl_analysis["head_pose"].yaw,
                "pitch": rstl_analysis["head_pose"].pitch,
                "roll": rstl_analysis["head_pose"].roll,
                "confidence": rstl_analysis["head_pose"].confidence
            },
            "excision_alignment": rstl_analysis.get("excision_alignment", {}),
            "rstl_line_count": len(rstl_analysis["rstl_lines"])
        }
    
    # Add enhanced planning results if available
    if enhanced_plan:
        summary["enhanced_planning"] = {
            "guideline_based_margins": True,
            "margin_calculation": {
                "peripheral_mm": enhanced_plan["margin_calculation"].peripheral_mm if enhanced_plan["margin_calculation"] else None,
                "defect_diameter_mm": enhanced_plan["margin_calculation"].defect_diameter_mm if enhanced_plan["margin_calculation"] else None,
                "confidence": enhanced_plan["margin_calculation"].confidence if enhanced_plan["margin_calculation"] else None
            } if enhanced_plan.get("margin_calculation") else None,
            "flap_recommendation": {
                "flap_type": enhanced_plan["flap_plan"].recommended_flap.flap_type.value if enhanced_plan["flap_plan"].recommended_flap else None,
                "success_rate": enhanced_plan["flap_plan"].recommended_flap.estimated_success_rate if enhanced_plan["flap_plan"].recommended_flap else None,
                "warnings": enhanced_plan["flap_plan"].warnings,
                "technical_notes": enhanced_plan["flap_plan"].technical_notes
            } if enhanced_plan.get("flap_plan") else None,
            "citations": [
                {
                    "title": c.get("title", ""),
                    "year": c.get("year", ""),
                    "level_of_evidence": c.get("level_of_evidence", "")
                }
                for c in enhanced_plan.get("citations", [])
            ]
        }

    with open(out_dir / "ai_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # 11) PDF report (best effort)
    try:
        make_pdf(
            out_path=out_dir / "report.pdf",
            summary=summary,
            overlay_path=(out_dir / "overlay.png") if overlay_path else None,
        )
    except Exception as e:
        # PDF is optional in the demo; continue silently but log error
        print(f"PDF generation failed: {e}")

    print(f"\n✅ Demo complete. Outputs in: {out_dir}")
    return summary
