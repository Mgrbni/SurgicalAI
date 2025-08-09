# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

from surgicalai.logging_utils import get_logger
from surgicalai.utils.io import read_json

LOGGER = get_logger(__name__)

try:  # optional dependency
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:  # pragma: no cover
    canvas = None


def generate(case_dir: Path) -> Path:
    pdf_path = case_dir / "report.pdf"
    lesion = read_json(case_dir / "lesion_probs.json")
    contraind = read_json(case_dir / "contraindications.json")
    if canvas is None:
        pdf_path.write_text("Report unavailable")
        return pdf_path
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(50, 750, f"Lesion probabilities: {lesion['class_probs']}")
    c.drawString(50, 730, f"Contraindication score: {contraind['score']}")
    c.showPage()
    c.save()
    LOGGER.info("report generated")
    return pdf_path
