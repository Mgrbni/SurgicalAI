import pathlib, logging, datetime
from typing import Dict, Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image, ImageDraw, ImageFont

try:
    from weasyprint import HTML  # type: ignore
    WEASYPRINT_AVAILABLE = True
except Exception as e:  # import error OR runtime dependency issue
    WEASYPRINT_AVAILABLE = False
    _IMPORT_ERR = e

LOGGER = logging.getLogger(__name__)
TEMPLATES_DIR = pathlib.Path(__file__).resolve().parent / 'templates'

# Ensure template dir exists
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(['html','xml'])
)

def _upscale_image_if_needed(path: pathlib.Path):
    """Upscale an image in-place if width < 1200px using LANCZOS, up to max 1600px or 2x."""
    try:
        if not path.exists() or path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            return
        with Image.open(path) as im:
            w, h = im.size
            if w < 1200:
                target_w = min(1600, w * 2)
                target_h = int(h * (target_w / w))
                im = im.resize((target_w, target_h), Image.LANCZOS)
                im.save(path)
    except Exception as e:
        LOGGER.warning(f"Image upscale skipped for {path.name}: {e}")


def _fallback_pdf(pdf_path: pathlib.Path, context: Dict[str, Any]):
    """Create a minimal, well‑formed single‑page PDF using Pillow only.

    This avoids external native deps (Cairo/Pango) required by WeasyPrint on Windows.
    """
    W, H = 1654, 2339  # ~ A4 @ 200dpi
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
        font_small = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    y = 40
    draw.text((60, y), f"SurgicalAI Report — RUN {context.get('run_id')}", fill="black", font=font)
    y += 50
    draw.text((60, y), f"Created: {context.get('created_at')}", fill="black", font=font_small)
    y += 40
    draw.text((60, y), "Probabilities:", fill="black", font=font_small)
    y += 30
    probs = context.get('probs', {})
    for k,v in probs.items():
        draw.text((80, y), f"• {k}: {v*100:.1f}%", fill="black", font=font_small)
        y += 26
        if y > H - 120:
            break
    y += 20
    gate = context.get('gate', {})
    draw.text((60, y), f"Gate: {gate.get('status','?')} - {gate.get('reason','')}", fill="black", font=font_small)
    y += 26
    if gate.get('guidance'):
        draw.text((60, y), f"Guidance: {gate.get('guidance')}", fill="black", font=font_small)
    # Footer (doctor + timestamp)
    footer_text = f"Dr. Mehdi Ghorbani Karimabad – Plastic, Reconstructive & Aesthetic Surgery | Generated {datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds')}"
    draw.text((60, H - 60), footer_text, fill="black", font=font_small)
    # Embed small thumbnails if available
    thumb_w = 360
    thumb_y = 60
    for key in ('original_path', 'heatmap_path', 'overlay_path'):
        path_name = context.get(key)
        if not path_name:
            continue
        p = pdf_path.parent / path_name
        if p.exists():
            try:
                im = Image.open(p).convert('RGB')
                im.thumbnail((thumb_w, thumb_w))
                if thumb_y + im.height < H: # Bounds check
                    img.paste(im, (W - thumb_w - 60, thumb_y))
                    thumb_y += im.height + 20 # Stack vertically with padding
            except Exception:
                pass # ignore if thumbnail fails
    # Save as PDF (Pillow will create a valid PDF object structure)
    img.save(str(pdf_path), "PDF", resolution=200.0)

def build_report(
    run_id: str,
    run_dir: pathlib.Path,
    lesion_probs: Dict[str, float],
    gate: Dict[str, Any],
    original_path: pathlib.Path,
    heatmap_path: pathlib.Path,
    overlay_path: Optional[pathlib.Path] = None,
    metrics: Optional[Dict[str, Any]] = None,
    produce_pdf: bool = True,
):
    """Render HTML (+ optional PDF) report for a run using a print-grade template.

    Arguments:
        run_id: Unique run identifier.
        run_dir: Directory containing artifacts (used as base_url for WeasyPrint).
        lesion_probs: Mapping of class -> probability (0..1).
        gate: Dict with keys allow_flap(bool), reason(str), guidance(str) or already normalized.
        original_path / heatmap_path / overlay_path: Image artifacts saved inside run_dir.
        metrics: Optional additional metrics dict.
    """
    created_at = datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds')

    # Normalize gate to template schema (status, reason, guidance)
    if 'status' not in gate:
        status = 'OK' if gate.get('allow_flap') else 'DEFER'
        gate = {
            'status': status,
            'reason': gate.get('reason', ''),
            'guidance': gate.get('guidance', ''),
        }

    # Ensure overlay path object
    if overlay_path is None:
        # attempt to find one in directory
        candidates = list(run_dir.glob('overlay.*'))
        overlay_path = candidates[0] if candidates else heatmap_path

    # Pre-upscale images (in-place) for print clarity
    for p in (original_path, heatmap_path, overlay_path):
        _upscale_image_if_needed(p)

    # Prepare relative filenames for template
    original_rel = original_path.name if original_path.exists() else ''
    heatmap_rel = heatmap_path.name if heatmap_path.exists() else ''
    overlay_rel = overlay_path.name if overlay_path and overlay_path.exists() else ''

    probs = lesion_probs or {}

    artifact_filenames = {
        'original': original_rel,
        'heatmap': heatmap_rel,
        'overlay': overlay_rel,
        'html': 'report.html',
    }

    template = env.get_template('report.html')
    context = {
        'run_id': run_id,
        'created_at': created_at,
        'probs': probs,
        'gate': gate,
        'metrics': metrics or {},
        'original_path': original_rel,
        'heatmap_path': heatmap_rel,
        'overlay_path': overlay_rel,
        'data': {
            'run_id': run_id,
            'created_at': created_at,
            'probs': probs,
            'gate': gate,
            'artifacts': artifact_filenames,
        },
    }
    html_str = template.render(**context)
    html_path = run_dir / 'report.html'
    pdf_path = run_dir / 'report.pdf'
    html_path.write_text(html_str, encoding='utf-8')
    if not produce_pdf:
        return html_path
    # Optional PDF path (legacy). If disabled or failing, ignore silently.
    if produce_pdf:
        if WEASYPRINT_AVAILABLE:
            try:
                HTML(string=html_str, base_url=str(run_dir)).write_pdf(str(pdf_path))
                if pdf_path.exists() and pdf_path.stat().st_size < 1024:
                    raise RuntimeError("WeasyPrint output unexpectedly small; invoking fallback")
                return pdf_path
            except Exception as e:
                LOGGER.warning(f"WeasyPrint generation failed (fallback to Pillow PDF): {e}")
                try:
                    _fallback_pdf(pdf_path, context)
                    return pdf_path
                except Exception:
                    pass
        else:
            # Directly build fallback PDF when WeasyPrint not installed
            try:
                _fallback_pdf(pdf_path, context)
                return pdf_path
            except Exception as e:
                LOGGER.warning(f"Fallback PDF generation failed: {e}")
    return html_path
