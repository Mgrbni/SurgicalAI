# surgicalai_demo/pipeline.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import cv2

from .config import SETTINGS
from .features import (
    _read_rgb,
    compute_abcde,
    save_overlay,
    dump_metrics,
    estimate_scale,
    defect_metrics_px_to_mm,
    image_to_bytes,
)
from .rules import plan_reconstruction, triage_tier0
from .rules.flap_planner import plan_flap
from .retrieval import SimpleAnn, feature_vector_from_abcde, save_thumbnails
from .report import make_pdf, make_pdf_legacy
from .transforms import preprocess_for_model, warp_to_original, gradcam, test_inverse_transform_accuracy
from .gradcam import gradcam_activation, save_overlays_full_and_zoom
from .vlm_observer import describe_lesion_roi
from .fusion import fuse_probs


# -------------------------- utility helpers --------------------------

def _extract_roi_for_vlm(img_rgb: np.ndarray, abcde: Dict[str, Any], heatmap: Optional[np.ndarray]) -> np.ndarray:
    """Extract ROI crop for VLM analysis."""
    h, w = img_rgb.shape[:2]
    
    # Use center from ABCDE or heatmap peak
    center_xy = abcde.get("center_xy")
    if center_xy:
        center_x, center_y = center_xy
    elif heatmap is not None:
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        center_x, center_y = float(peak_x), float(peak_y)
    else:
        center_x, center_y = w/2, h/2
    
    # Determine ROI size based on lesion diameter or default
    diameter_px = abcde.get("diameter_px", min(w, h) // 4)
    roi_size = max(int(diameter_px * 1.5), 128)  # At least 128px, 1.5x lesion size
    roi_size = min(roi_size, min(w, h) // 2)  # Don't exceed half image size
    
    # Calculate crop bounds
    half_size = roi_size // 2
    x1 = max(0, int(center_x - half_size))
    y1 = max(0, int(center_y - half_size))
    x2 = min(w, x1 + roi_size)
    y2 = min(h, y1 + roi_size)
    
    # Adjust if crop goes out of bounds
    if x2 - x1 < roi_size:
        x1 = max(0, x2 - roi_size)
    if y2 - y1 < roi_size:
        y1 = max(0, y2 - roi_size)
    
    return img_rgb[y1:y2, x1:x2]


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
    
    # 3b) Advanced Grad-CAM activation (if model available)
    try:
        # For demo, use synthetic gradcam - in production would use real model
        heatmap_settings = getattr(SETTINGS, 'heatmap', {})
        activation = gradcam_activation(img_rgb)
        
        # Test inverse transform accuracy for debugging
        if os.getenv("DEBUG_TRANSFORMS", "").lower() == "true":
            accuracy_metrics = test_inverse_transform_accuracy(img_rgb)
            print(f"Transform accuracy: {accuracy_metrics}")
            
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}, using fallback")
        activation = None
    
    heatmap = activation

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

    # 5) Tier‑0 triage — returns STRING label; use directly without attribute access
    tri_label = triage_tier0(abcde, age=meta["age"], body_site=meta["body_site"])
    
    # 6) CNN class probabilities (demo/synthetic for now)
    # In production, this would use actual CNN inference
    probs: Dict[str, float] = {
        "melanoma": float(abcde.get("melanoma_prob", 0.15) or 0.15),
        "bcc": float(abcde.get("bcc_prob", 0.20) or 0.20),
        "scc": float(abcde.get("scc_prob", 0.15) or 0.15),
        "nevus": float(abcde.get("nevus_prob", 0.25) or 0.25),
        "seborrheic_keratosis": float(abcde.get("sk_prob", 0.20) or 0.20),
        "benign_other": float(abcde.get("benign_prob", 0.05) or 0.05)
    }
    
    # 6b) Vision-LLM Observer Analysis (if enabled)
    vlm_analysis = None
    fusion_result = None
    
    fusion_config = getattr(SETTINGS, 'fusion', {})
    use_vlm = fusion_config.get('use_vlm', False)
    
    if use_vlm:
        try:
            # Extract ROI for VLM analysis
            roi_crop = _extract_roi_for_vlm(img_rgb, abcde, heatmap)
            roi_bytes = image_to_bytes(roi_crop)
            
            # Get VLM analysis
            vlm_provider = fusion_config.get('provider', 'openai')
            vlm_analysis = describe_lesion_roi(roi_bytes, provider=vlm_provider)
            
            # Fuse CNN and VLM predictions
            fusion_result = fuse_probs(probs, vlm_analysis, config=fusion_config)
            
            # Update final probabilities with fused results
            if not fusion_result.get('error'):
                probs = fusion_result['final_probs']
                
        except Exception as e:
            print(f"VLM analysis failed: {e}, continuing with CNN only")
            vlm_analysis = {"error": str(e), "fallback": True}

    # 7) Planning (oncology gates + flap suggestions)
    # Legacy plan for compatibility
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
    
    # 7b) New detailed flap planning
    try:
        # Extract top diagnosis from probabilities
        top_dx = max(probs.items(), key=lambda x: x[1])[0] if probs else 'unknown'
        
        detailed_plan = plan_flap(
            diagnosis=top_dx,
            site=meta["body_site"] or "cheek",
            diameter_mm=(metrics_mm or {}).get("diameter_mm"),
            borders_well_defined=not bool(abcde.get("border_irregularity", 0) and abcde.get("border_irregularity", 0) > 1.8),
            high_risk=False
        )
        
        # Convert to dict for JSON serialization
        plan["detailed"] = {
            "incision_orientation_deg": detailed_plan.incision_orientation_deg,
            "rstl_unit": detailed_plan.rstl_unit,
            "oncology": detailed_plan.oncology,
            "candidates": detailed_plan.candidates,
            "danger": detailed_plan.danger,
            "citations": detailed_plan.citations,
            "defer": detailed_plan.defer
        }
        
    except Exception as e:
        # Fallback if detailed planning fails
        plan["detailed"] = {
            "error": f"Detailed planning failed: {str(e)}",
            "incision_orientation_deg": None,
            "rstl_unit": "",
            "oncology": {"action": "manual consultation", "reason": "planning error"},
            "candidates": [],
            "danger": [],
            "citations": [],
            "defer": True
        }

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

    # 9) Visual overlay generation with advanced heatmaps
    overlay_paths = {}
    try:
        # Get top prediction for overlay annotation
        top_class = max(probs.items(), key=lambda x: x[1])
        top_class_name, top_prob = top_class
        
        if heatmap is not None:
            # Use advanced gradcam overlay system
            overlay_paths = save_overlays_full_and_zoom(
                image=img_rgb,
                activation=heatmap,
                run_dir=out_dir,
                top_class=top_class_name,
                probability=top_prob,
                config=getattr(SETTINGS, 'heatmap', {})
            )
        else:
            # Fallback to basic overlay
            overlay_path = out_dir / "overlay.png"
            save_overlay(img_rgb, heatmap, overlay_path, plan_data=plan)
            overlay_paths = {"overlay": overlay_path}
            
    except Exception as e:
        print(f"Overlay generation failed: {e}")
        overlay_paths = {}

    # 10) Dump metrics + AI summary JSON
    dump_metrics(
        out_dir / "metrics.json",
        img_path=str(lesion_path),
        mm_per_px=mm_per_px,
        metrics_px=metrics_px,
        metrics_mm=metrics_mm,
        abcde=abcde,
    )
    
    # 10b) Save detailed flap plan as separate JSON
    if "detailed" in plan and not plan["detailed"].get("error"):
        with open(out_dir / "plan.json", "w", encoding="utf-8") as f:
            json.dump(plan["detailed"], f, ensure_ascii=False, indent=2)

    summary: Dict[str, Any] = {
        "triage": {
            "label": str(tri_label),
        },
        "probs": probs,
        "plan": plan,
        "meta": meta,
        "paths": {
            "overlay_png": str(overlay_paths.get("overlay", "")),
            "overlay_full_png": str(overlay_paths.get("overlay_full", "")),
            "overlay_zoom_png": str(overlay_paths.get("overlay_zoom", "")),
            "lesion_image": str(lesion_path),
        },
        **neighbors_info,
    }
    
    # Add VLM and fusion results if available
    if vlm_analysis:
        summary["observer"] = vlm_analysis
    
    if fusion_result:
        summary["fusion"] = {
            "top3": fusion_result.get("top3", []),
            "gate": fusion_result.get("gate", ""),
            "notes": fusion_result.get("fusion_notes", ""),
            "vlm_descriptors": fusion_result.get("vlm_descriptors", []),
            "descriptor_adjustments": fusion_result.get("descriptor_adjustments", {})
        }

    with open(out_dir / "ai_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 11) PDF report (best effort)
    try:
        # Build metrics subset for new PDF API
        metrics_for_pdf = {
            "label": summary.get("triage", {}).get("label"),
            "age": summary.get("meta", {}).get("age"),
            "sex": summary.get("meta", {}).get("sex"),
            "body_site": summary.get("meta", {}).get("body_site"),
            "photo_type": summary.get("meta", {}).get("photo_type"),
            "mm_per_px": summary.get("meta", {}).get("known_scale_mm_per_px"),
        }

        make_pdf(
            pdf_path=out_dir / "report.pdf",
            title="SurgicalAI Analysis Report",
            version="Demo v1.0",
            metrics=metrics_for_pdf,
            overlay_png=overlay_paths.get("overlay") or overlay_paths.get("overlay_zoom") or "",
            overlay_zoom_png=str(overlay_paths.get("overlay_zoom")) if overlay_paths.get("overlay_zoom") else None,
            overlay_full_png=str(overlay_paths.get("overlay_full")) if overlay_paths.get("overlay_full") else None,
            neighbor_paths=[n.get("path") for n in summary.get("neighbors", []) if isinstance(n, dict) and n.get("path")],
            observer=summary.get("observer"),
            fusion=summary.get("fusion"),
        )
    except Exception as e:
        # PDF is optional in the demo; continue silently
        print(f"PDF generation failed: {e}")
        pass

    print(f"\n✅ Demo complete. Outputs in: {out_dir}")
    return summary
