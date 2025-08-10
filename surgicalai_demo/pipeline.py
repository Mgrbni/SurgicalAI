from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from PIL import Image

from .config import SETTINGS
from .features import compute_abcde, save_overlay, dump_metrics, ABCDE
from .rules import triage_tier0
from .retrieval import SimpleAnn, feature_vector_from_abcde, save_thumbnails
from .report import make_pdf

def run_demo(face_img: Path, lesion_img: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = np.array(Image.open(lesion_img).convert("RGB"))
    abcde, blob = compute_abcde(rgb)
    overlay_path = out_dir / "overlay.png"
    save_overlay(rgb, blob, overlay_path)

    tri = triage_tier0(abcde, age=None, body_site=None)
    extras = {
        "label": tri.label,
        "risk_pct": tri.risk_pct,
        "rationale": tri.rationale,
        "age": None,
        "body_site": None,
    }
    dump_metrics(abcde, extras, out_dir / "metrics.json")
    metrics = {**extras, **{
        "asymmetry": abcde.asymmetry,
        "border_irregularity": abcde.border_irregularity,
        "color_variegation": abcde.color_variegation,
        "diameter_px": abcde.diameter_px,
        "elevation_satellite": abcde.elevation_satellite,
    }}

    # Optional nearest-neighbor retrieval from precomputed features
    hits = []
    feats_dir = Path(SETTINGS.feats_dir)
    if feats_dir.exists():
        vectors = []
        meta = []
        for jf in feats_dir.glob("*.json"):
            try:
                d = json.loads(jf.read_text())
                vec = [
                    float(d.get("asymmetry", 0.0)),
                    float(d.get("border_irregularity", 0.0)),
                    float(d.get("color_variegation", 0.0)),
                    float(d.get("diameter_px", 0.0)),
                    float(d.get("elevation_satellite", 0.0)),
                ]
                vectors.append(vec)
                meta.append({"path": d.get("path", "")})
            except Exception:
                continue
        if vectors:
            ann = SimpleAnn()
            ann.fit(np.array(vectors, dtype=np.float32), meta)
            hits = ann.query(feature_vector_from_abcde(abcde), k=SETTINGS.neighbors_k)
            save_thumbnails(hits, out_dir / "nn")

    neighbor_imgs = []
    for i in range(len(hits)):
        p = out_dir / "nn" / f"nn_{i+1}.jpg"
        if p.exists():
            neighbor_imgs.append(str(p))

    make_pdf(out_dir / "report.pdf", SETTINGS.report_title, SETTINGS.version, metrics, str(overlay_path), neighbor_imgs)
