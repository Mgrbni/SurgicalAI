from __future__ import annotations
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from pathlib import Path

def make_pdf(pdf_path, title, version, metrics: dict, overlay_png: str, neighbor_paths: list[str]):
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    W, H = A4
    c.setTitle(title)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, H-2*cm, title)
    c.setFont("Helvetica", 9)
    c.drawString(2*cm, H-2.6*cm, f"Version: {version}")

    # Metrics
    y = H-4*cm
    c.setFont("Helvetica", 10)
    for k in ["label","risk_pct","rationale","asymmetry","border_irregularity","color_variegation","diameter_px","elevation_satellite","age","body_site"]:
        if k in metrics and metrics[k] is not None:
            c.drawString(2*cm, y, f"{k}: {metrics[k]}")
            y -= 0.6*cm

    # Overlay image
    try:
        c.drawImage(str(overlay_png), 2*cm, 7*cm, width=8*cm, height=8*cm, preserveAspectRatio=True, mask='auto')
    except Exception:
        pass

    # Neighbors
    x0, y0 = 12*cm, 11*cm
    thumb_w = 3*cm
    thumb_h = 3*cm
    for p in neighbor_paths:
        if not Path(p).exists():
            continue
        try:
            c.drawImage(str(p), x0, y0, width=thumb_w, height=thumb_h, preserveAspectRatio=True, mask='auto')
        except Exception:
            pass
        y0 -= thumb_h + 0.3*cm
        if y0 < 2*cm:
            y0 = 11*cm
            x0 += thumb_w + 0.3*cm
    c.showPage()
    c.save()
