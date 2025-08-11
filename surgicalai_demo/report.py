# surgicalai_demo/report.py
from __future__ import annotations

from typing import List, Dict, Any
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


def add_footer(c: canvas.Canvas, W: float, H: float) -> None:
    """Add footer with credit to Dr. Ghorbani on every page."""
    c.setFont("Helvetica", 10)
    footer_text = "© Dr. Mehdi Ghorbani Karimabad — Plastic, Reconstructive & Aesthetic Surgery"
    text_width = c.stringWidth(footer_text, "Helvetica", 10)
    c.drawString(W - text_width - 2*cm, 1*cm, footer_text)


def add_ai_page(c: canvas.Canvas, W, H, ai: Dict[str, Any]) -> None:
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, H-2*cm, "AI Summary (decision-support, not diagnosis)")
    y = H-3.2*cm

    def line(txt: str) -> None:
        nonlocal y
        c.setFont("Helvetica", 10)
        c.drawString(2*cm, y, str(txt)[:110])
        y -= 0.6*cm

    line(f"Triage label: {ai.get('triage_label')}")
    line("Differentials (prob%):")
    for d in (ai.get("diagnosis_candidates") or [])[:5]:
        name = d.get("name", "")
        pct = d.get("probability_pct", 0)
        try:
            pct = round(float(pct), 1)
        except Exception:
            pass
        line(f"  - {name}: {pct}%")

    flap = ai.get("flap_plan") or {}
    line("Flap suggestion:")
    line(f"  Indicated: {flap.get('indicated')}")
    line(f"  Type: {flap.get('type')}")
    line(f"  Design: {flap.get('design_summary')}")
    line(f"  Langer’s lines: {flap.get('tension_lines')}")
    line(f"  Rotation vector: {flap.get('rotation_vector')}")
    if flap.get("predicted_success_pct") is not None:
        try:
            sp = round(float(flap.get("predicted_success_pct")), 1)
        except Exception:
            sp = flap.get("predicted_success_pct")
        line(f"  Predicted success: {sp}%")
    line(f"  Key risks: {flap.get('key_risks')}")

    notes = ai.get("notes")
    if notes:
        line("Notes:")
        for i in range(0, len(notes), 100):
            line("  " + notes[i:i+100])
    
    # Add footer to this page
    add_footer(c, W, H)


def make_pdf(
    pdf_path,
    title: str,
    version: str,
    metrics: Dict[str, Any],
    overlay_png: str,
    neighbor_paths: List[str],
    ai: Dict[str, Any] | None = None,
) -> None:
    """Generate a 1–2 page PDF. Page 1: metrics/overlay/NN. Page 2: AI summary (if provided)."""
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    W, H = A4
    c.setTitle(title)

    # ---- Page 1 ----
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, H-2*cm, title)
    c.setFont("Helvetica", 9)
    c.drawString(2*cm, H-2.6*cm, f"Version: {version}")

    y = H-4*cm
    c.setFont("Helvetica", 10)
    fields = [
        "label","risk_pct","rationale",
        "asymmetry","border_irregularity","color_variegation",
        "diameter_px","elevation_satellite",
        "age","sex","body_site","photo_type","mm_per_px","filename"
    ]
    for k in fields:
        if k in metrics and metrics[k] is not None:
            c.drawString(2*cm, y, f"{k}: {metrics[k]}")
            y -= 0.6*cm

    # overlay image
    try:
        c.drawImage(str(overlay_png), 2*cm, 7*cm, width=8*cm, height=8*cm,
                    preserveAspectRatio=True, mask="auto")
    except Exception:
        pass

    # neighbors
    x0, y0 = 11*cm, 12*cm
    w, h = 4*cm, 3*cm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x0, y0 + 1*cm, "Nearest confirmed cases")
    for i, p in enumerate(neighbor_paths[:6]):
        try:
            c.drawImage(p,
                        x0 + (i % 2) * (w + 0.5*cm),
                        y0 - 0.5*cm - (i // 2) * (h + 0.5*cm),
                        width=w, height=h,
                        preserveAspectRatio=True, mask="auto")
        except Exception:
            continue

    # Add footer to first page
    add_footer(c, W, H)
    c.showPage()

    # ---- Page 2: AI summary ----
    if ai:
        add_ai_page(c, W, H, ai)
        c.showPage()

    c.save()
