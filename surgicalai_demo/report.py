# surgicalai_demo/report.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
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
    ai: Optional[Dict[str, Any]] = None,
    overlay_zoom_png: Optional[str] = None,
    overlay_full_png: Optional[str] = None,
    observer: Optional[Dict[str, Any]] = None,
    fusion: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate PDF with zoom + full overlay, observer + fusion summaries."""
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    W, H = A4
    c.setTitle(title)

    # Page 1
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, H-2*cm, title)
    c.setFont("Helvetica", 9)
    c.drawString(2*cm, H-2.6*cm, f"Version: {version}")

    y = H-4*cm
    c.setFont("Helvetica", 10)
    fields = [
        "label","risk_pct","rationale","age","sex","body_site","photo_type","mm_per_px"
    ]
    for k in fields:
        if k in metrics and metrics[k] is not None:
            c.drawString(2*cm, y, f"{k}: {metrics[k]}")
            y -= 0.55*cm

    primary_overlay = overlay_zoom_png or overlay_png
    secondary_overlay = overlay_full_png if overlay_zoom_png else None
    main_img_x, main_img_y = 10*cm, 8*cm
    main_img_w = main_img_h = 8*cm
    try:
        if primary_overlay:
            c.drawImage(str(primary_overlay), main_img_x, main_img_y,
                        width=main_img_w, height=main_img_h,
                        preserveAspectRatio=True, mask='auto')
            c.setFont("Helvetica", 8)
            c.drawString(main_img_x, main_img_y - 0.45*cm, "ROI overlay with contours (0.3/0.5/0.7)")
    except Exception:
        pass

    if secondary_overlay:
        sx, sy = main_img_x, main_img_y - 4*cm
        try:
            c.drawImage(str(secondary_overlay), sx, sy, width=3*cm, height=3*cm,
                        preserveAspectRatio=True, mask='auto')
            c.setFont("Helvetica", 8)
            c.drawString(sx, sy - 0.3*cm, "Full overlay")
        except Exception:
            pass

    # Nearest neighbors
    x0, y0 = 2*cm, 12*cm
    w_n, h_n = 3*cm, 2.5*cm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x0, y0 + 1*cm, "Nearest cases")
    for i, pth in enumerate(neighbor_paths[:6]):
        try:
            c.drawImage(pth, x0 + (i % 2) * (w_n + 0.5*cm),
                        y0 - 0.5*cm - (i // 2) * (h_n + 0.5*cm),
                        width=w_n, height=h_n, preserveAspectRatio=True, mask='auto')
        except Exception:
            continue

    # Observer + fusion
    table_y = main_img_y - 5.2*cm
    c.setFont("Helvetica-Bold", 10)
    c.drawString(main_img_x, table_y, "Observer")
    table_y -= 0.4*cm
    c.setFont("Helvetica", 8)
    if observer:
        desc = ", ".join(observer.get("descriptors", [])[:5]) or "-"
        c.drawString(main_img_x, table_y, f"Primary: {observer.get('primary_pattern','-')}")
        table_y -= 0.32*cm
        c.drawString(main_img_x, table_y, f"Desc: {desc[:60]}")
        table_y -= 0.32*cm
        c.drawString(main_img_x, table_y, f"Rec: {observer.get('recommendation','-')}")
        table_y -= 0.4*cm
    else:
        c.drawString(main_img_x, table_y, "(none)")
        table_y -= 0.4*cm

    c.setFont("Helvetica-Bold", 10)
    c.drawString(main_img_x, table_y, "Fusion Top‑3")
    table_y -= 0.4*cm
    c.setFont("Helvetica", 8)
    if fusion and fusion.get("top3"):
        for cls, p in fusion["top3"][:3]:
            c.drawString(main_img_x, table_y, f"{cls}: {p*100:.1f}%")
            table_y -= 0.3*cm
        note = fusion.get("notes", "")
        if note:
            c.drawString(main_img_x, table_y, note[:70])
            table_y -= 0.35*cm
    else:
        c.drawString(main_img_x, table_y, "(n/a)")
        table_y -= 0.35*cm

    c.setFont("Helvetica", 7)
    c.drawString(2*cm, 1.5*cm, "Not for clinical use. Research demonstration only.")
    add_footer(c, W, H)
    c.showPage()

    if ai:
        add_ai_page(c, W, H, ai)
        c.showPage()
    c.save()


# Legacy wrapper kept for existing pipeline usage (summary based)
def make_pdf_legacy(
    out_path,
    summary: Dict[str, Any],
    overlay_path=None,
) -> None:
    """Legacy PDF generation function for backwards compatibility."""
    from pathlib import Path
    
    # Extract data from summary
    title = "SurgicalAI Analysis Report"
    version = "Demo v1.0"
    
    # Build metrics from summary
    metrics = {
        "label": summary.get("triage", {}).get("label", "Unknown"),
        "risk_pct": "N/A",
        "rationale": "Demo analysis",
    }
    
    # Add probability metrics
    probs = summary.get("probs", {})
    for class_name, prob in probs.items():
        metrics[f"{class_name}_prob"] = f"{prob:.1%}"
    
    # Meta information
    meta = summary.get("meta", {})
    for key in ["age", "sex", "body_site", "photo_type", "mm_per_px"]:
        if key in meta:
            metrics[key] = meta[key]
    
    # Paths
    paths = summary.get("paths", {})
    overlay_png = str(overlay_path) if overlay_path and Path(overlay_path).exists() else None
    
    # Look for new overlay artifacts
    overlay_zoom_png = None
    overlay_full_png = None
    
    if overlay_path:
        run_dir = Path(overlay_path).parent
        zoom_path = run_dir / "overlay_zoom.png"
        full_path = run_dir / "overlay_full.png"
        
        if zoom_path.exists():
            overlay_zoom_png = str(zoom_path)
        if full_path.exists():
            overlay_full_png = str(full_path)
    
    # Neighbor paths
    neighbors = summary.get("neighbors", [])
    neighbor_paths = []
    for neighbor in neighbors[:6]:
        if isinstance(neighbor, dict) and "path" in neighbor:
            neighbor_paths.append(neighbor["path"])
    
    # AI summary
    ai_summary = None
    if "plan" in summary:
        plan = summary["plan"]
        ai_summary = {
            "triage_label": summary.get("triage", {}).get("label", "Unknown"),
            "diagnosis_candidates": [
                {"name": name, "probability_pct": prob * 100}
                for name, prob in probs.items()
            ],
            "flap_plan": {
                "indicated": plan.get("indicated", False),
                "type": plan.get("type", "Unknown"),
                "design_summary": plan.get("design_summary", ""),
                "tension_lines": plan.get("tension_lines", ""),
                "rotation_vector": plan.get("rotation_vector", ""),
                "predicted_success_pct": plan.get("predicted_success_pct"),
                "key_risks": ", ".join(plan.get("key_risks", [])),
            }
        }
    
    # Generate PDF
    make_pdf(
        pdf_path=out_path,
        title=title,
        version=version,
        metrics=metrics,
        overlay_png=overlay_png or "",
        neighbor_paths=neighbor_paths,
        ai=ai_summary,
        overlay_zoom_png=overlay_zoom_png,
        overlay_full_png=overlay_full_png,
        observer=summary.get("observer"),
        fusion=summary.get("fusion"),
    )
