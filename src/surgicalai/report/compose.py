from pathlib import Path
from typing import Dict, Any
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import Table, TableStyle
from PIL import Image
import datetime


def _draw_header(c: canvas.Canvas, title: str, ts: str):
    c.setFont("Helvetica-Bold", 18)
    c.drawString(30*mm, 285*mm, title)
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.grey)
    c.drawRightString(200*mm, 285*mm, ts)
    c.setFillColor(colors.black)
    c.line(20*mm, 282*mm, 190*mm, 282*mm)


def _safe_image(path: str | None, max_w_mm: float, max_h_mm: float):
    if not path or not Path(path).exists():
        return None
    try:
        im = Image.open(path)
        w, h = im.size
        ratio = min((max_w_mm*mm)/w, (max_h_mm*mm)/h)
        return im, w*ratio, h*ratio
    except Exception:
        return None


def make_pdf(pdf_path: Path, meta: Dict[str, Any], model_output: Dict[str, Any], guidelines: Dict[str, Any], artifacts: Dict[str, str]):
    """Generate a structured multi-section PDF with embedded images and stats.

    Sections:
      1. Header
      2. Probabilities table
      3. Gate decision
      4. Images (original / heatmap / overlay / guideline card)
      5. Footnote
    """
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    ts = datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds')
    _draw_header(c, "SurgicalAI Report", ts)

    y = 270*mm
    # Meta table
    meta_items = [["Patient", f"Age: {meta.get('age','?')}  Sex: {meta.get('sex','?')}"],
                  ["Site", meta.get('site','?')],
                  ["Prior Histology", meta.get('prior_histology','?')]]
    t = Table(meta_items, colWidths=[35*mm, 120*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(1,0), colors.whitesmoke),
        ('BOX',(0,0),(-1,-1), 0.25, colors.grey),
        ('INNERGRID',(0,0),(-1,-1), 0.25, colors.grey),
        ('FONT',(0,0),(-1,-1),'Helvetica',9),
        ('FONT',(0,0),(0,-1),'Helvetica-Bold',9),
    ]))
    w,h = t.wrapOn(c, 170*mm, 40*mm)
    t.drawOn(c, 25*mm, y-h)
    y -= h + 6*mm

    # Probabilities
    probs = model_output.get('probs', {})
    prob_rows = [["Diagnosis", "Probability"]]
    for k,v in probs.items():
        prob_rows.append([k, f"{v*100:.1f}%" if v <= 1 else f"{v:.2f}"])
    t2 = Table(prob_rows, colWidths=[60*mm, 30*mm])
    t2.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0), colors.lightgrey),
        ('BOX',(0,0),(-1,-1), 0.25, colors.grey),
        ('INNERGRID',(0,0),(-1,-1), 0.25, colors.grey),
        ('FONT',(0,0),(-1,0),'Helvetica-Bold',9),
        ('FONT',(0,1),(-1,-1),'Helvetica',9),
    ]))
    w2,h2 = t2.wrapOn(c, 90*mm, 60*mm)
    t2.drawOn(c, 25*mm, y-h2)

    # Gate decision
    gate = model_output.get('gate', {})
    c.setFont('Helvetica-Bold', 11)
    c.drawString(120*mm, y-15)
    c.drawString(120*mm, y-15, f"Gate: {'ALLOW' if gate.get('allow_flap') else 'DEFER'}")
    c.setFont('Helvetica', 9)
    c.drawString(120*mm, y-22, gate.get('reason',''))
    c.drawString(120*mm, y-29, gate.get('guidance',''))
    y -= max(h2, 30*mm) + 6*mm

    # Images
    slots = [artifacts.get('overlay'), artifacts.get('heatmap'), artifacts.get('guideline_card')]
    x = 25*mm
    for p in slots:
        img_info = _safe_image(p, 52, 60)
        if img_info:
            im, w_img, h_img = img_info
            im_path_tmp = pdf_path.parent / (Path(p).stem + '_tmp_preview.jpg')
            try:
                im.save(str(im_path_tmp), quality=85)
                c.drawImage(str(im_path_tmp), x, y - h_img, width=w_img, height=h_img, preserveAspectRatio=True)
            except Exception:
                pass
        x += 55*mm
    y -= 62*mm

    if y < 40*mm:
        c.showPage()
        y = 270*mm

    # Footnote on last page
    c.setFont('Helvetica-Oblique', 8)
    c.setFillColor(colors.grey)
    c.drawString(20*mm, 15*mm, 'Dr. Mehdi Ghorbani Karimabad – Plastic, Reconstructive & Aesthetic Surgery | Generated ' + ts)
    c.save()
