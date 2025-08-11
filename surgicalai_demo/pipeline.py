# surgicalai_demo/pipeline.py
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
    abcde = compute_abcde(img_rgb)  # dict‑like
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

    # 5) Tier‑0 triage — returns STRING label; use directly without attribute access
    tri_label = triage_tier0(abcde, age=meta["age"], body_site=meta["body_site"])
    
    # 6) Very simple class probs for planner (demo safe)
    # If you have a classifier, replace these with real outputs.
    probs: Dict[str, float] = {
        "melanoma": float(abcde.get("melanoma_prob", 0.0) or 0.0),
        "bcc": float(abcde.get("bcc_prob", 0.0) or 0.0),
        "scc": float(abcde.get("scc_prob", 0.0) or 0.0),
    }

    # 7) Planning (oncology gates + flap suggestions)
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
    neighbors_info: Dict[str, Any] = {}
    try:
        ann = SimpleAnn()
        vec = feature_vector_from_abcde(abcde)
        hits = ann.search(vec, k=6)
        thumbs = save_thumbnails(hits, out_dir)
        neighbors_info = {"neighbors": hits, "thumbnails": thumbs}
    except Exception:
        neighbors_info = {"neighbors": [], "thumbnails": []}

    # 9) Visual overlay
    try:
        overlay_path = out_dir / "overlay.png"
        save_overlay(img_rgb, heatmap, overlay_path)
    except Exception:
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

    with open(out_dir / "ai_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 11) PDF report (best effort)
    try:
        make_pdf(
            out_path=out_dir / "report.pdf",
            summary=summary,
            overlay_path=(out_dir / "overlay.png") if overlay_path else None,
        )
    except Exception:
        # PDF is optional in the demo; continue silently
        pass

    print(f"\n✅ Demo complete. Outputs in: {out_dir}")
    return summary
