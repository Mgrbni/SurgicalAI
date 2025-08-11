import pathlib, io
from surgicalai.report.compose import make_pdf
from PyPDF2 import PdfReader

def test_compose_pdf_basic(tmp_path):
    pdf_path = tmp_path / 'report.pdf'
    make_pdf(
        pdf_path=pdf_path,
        meta={'age': 55, 'sex': 'F', 'site': 'cheek', 'prior_histology': 'no'},
        model_output={'probs': {'melanoma': 0.2, 'bcc': 0.3}, 'gate': {'allow_flap': True, 'reason': 'Low risk', 'guidance': 'Proceed'}},
        guidelines={'flap_options': [], 'danger_structures': [], 'citations': []},
        artifacts={'overlay': '', 'heatmap': '', 'guideline_card': ''},
    )
    assert pdf_path.exists()
    b = pdf_path.read_bytes()
    assert b.startswith(b'%PDF') and len(b) > 1200
    reader = PdfReader(io.BytesIO(b))
    assert len(reader.pages) >= 1
