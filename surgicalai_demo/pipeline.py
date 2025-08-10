# surgicalai_demo/pipeline.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from .config import SETTINGS
from .features import _read_rgb, compute_abcde, save_overlay, dump_metrics
from .rules import triage_tier0
from .retrieval import SimpleAnn, feature_vector_from_abcde, save_thumbnails
from .report import make_pdf


# ---------------------------- helpers ----------------------------

def _pick_file_if_needed(img_path: Optional[Path]) -> Optional[Path]:
    """Return a valid image path, or open a file dialog if missing."""
    if img_path and Path(img_path).exists():
        return Path(img_path)
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        f = filedialog.askopenfilename(
            title="Select lesion image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        return Path(f) if f else None
    except Exception:
        # Headless/CI fallback
        return img_path


def _ask_patient_info() -> Dict[str, Any]:
    """Collect optional demographics and context from the console."""
    def _clean(x: str) -> Optional[str]:
        x = (x or "").strip()
        return None if x == "" else x

    print("\nEnter optional patient/photo info (press Enter to skip):")
    age = _clean(input("  Age (years): "))
    sex = _clean(input("  Sex (male/female/other): "))
    site = _clean(input("  Body site (e.g., cheek, scalp, trunk): "))
    ptype = _clean(input("  Photo type (dermoscopy / clinical): "))
    scale = _clean(input("  Known scale (mm per pixel, e.g., 0.02) or blank: "))

    try:
        age = int(age) if age is not None else None
    except Exception:
        age = None
    try:
        scale = float(scale) if scale is not None else None
    except Exception:
        scale = None

    return {
        "age": age,
        "sex": sex,
        "body_site": site,
        "photo_type": ptype,
        "mm_per_px": scale,
    }


# ----------------------------- main ------------------------------

def run_demo(
    face_img_path: Optional[Path],
    lesion_img_path: Optional[Path],
    out_dir: Path,
    ask: bool = False,
    use_openai: bool = True,
):
    """
    Tier‑0 pipeline:
      • pick/ingest image (+ optional patient inputs)
      • compute ABCDE + overlay
      • rules triage
      • simple NN thumbnails
      • OpenAI JSON (or deterministic fallback)
      • 2‑page PDF (metrics + AI summary)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) choose image (dialog if not provided)
    lesion_img_path = _pick_file_if_needed(lesion_img_path)
    if not lesion_img_path or not lesion_img_path.exists():
        raise FileNotFoundError("No lesion image selected/found.")

    # 2) optional patient inputs (used by rules + reported in PDF)
    user = _ask_patient_info() if ask else {
        "age": None,
        "sex": None,
        "body_site": None,
        "photo_type": None,
        "mm_per_px": None,
    }

    # 3) features
    rgb = _read_rgb(lesion_img_path)
    abcde, blob = compute_abcde(rgb)

    # 4) triage (rules with priors from inputs)
    age = user.get("age")
    body_site = user.get("body_site")
    tri = triage_tier0(abcde, age, body_site)

    # 5) artifacts: overlay + metrics.json (merges ABCDE + context)
    overlay_path = out_dir / "overlay.png"
    save_overlay(rgb, blob, overlay_path)

    metrics: Dict[str, Any] = {
        "label": tri.label,
        "risk_pct": round(tri.risk_pct, 1),
        "rationale": tri.rationale,
        "age": user.get("age"),
        "sex": user.get("sex"),
        "body_site": user.get("body_site"),
        "photo_type": user.get("photo_type"),
        "mm_per_px": user.get("mm_per_px"),
        "filename": os.path.basename(str(lesion_img_path)),
    }
    dump_metrics(abcde, metrics, out_dir / "metrics.json")

    # 6) nearest neighbors (placeholder index using the same image to show UI)
    ann = SimpleAnn()
    vecs = np.vstack([feature_vector_from_abcde(abcde)])
    ann.fit(vecs, [{"path": str(lesion_img_path), "dx": "(demo)"}])
    hits = ann.query(feature_vector_from_abcde(abcde), k=6)
    nn_dir = out_dir / "neighbors"
    save_thumbnails(hits, nn_dir)
    neighbor_paths = sorted(str(p) for p in nn_dir.glob("nn_*.jpg"))

    # 7) OpenAI or fallback → ai_summary.json
    #    (always produce AI page content so the PDF never looks empty)
    full_metrics = json.loads((out_dir / "metrics.json").read_text())
    full_metrics.update({
        "asymmetry": abcde.asymmetry,
        "border_irregularity": abcde.border_irregularity,
        "color_variegation": abcde.color_variegation,
        "diameter_px": abcde.diameter_px,
        "elevation_satellite": abcde.elevation_satellite,
    })

    ai: Dict[str, Any] = {}
    if use_openai:
        try:
            from server.ai_openai import summarize_case  # lazy import
            ai = summarize_case(full_metrics)
        except Exception as e:
            ai = {"error": f"OpenAI call failed: {e}"}

    if not ai or "error" in ai:
        from server.ai_fallback import fallback_summarize_case
        ai = fallback_summarize_case(full_metrics)

    (out_dir / "ai_summary.json").write_text(json.dumps(ai, indent=2))

    # 8) PDF (page 1 metrics; page 2 AI summary)
    make_pdf(
        pdf_path=out_dir / "report.pdf",
        title=SETTINGS.report_title,
        version=SETTINGS.version,
        metrics=json.loads((out_dir / "metrics.json").read_text()),
        overlay_png=str(overlay_path),
        neighbor_paths=neighbor_paths,
        ai=ai,
    )

    print(f"[OK] Demo artifacts → {out_dir}")
    print("  - overlay.png")
    print("  - metrics.json")
    print("  - ai_summary.json")
    print("  - neighbors/*")
    print("  - report.pdf")
