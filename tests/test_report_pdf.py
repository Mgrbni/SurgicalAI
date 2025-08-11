import io, os, json, time, pathlib
from fastapi.testclient import TestClient
from server.http_api import app

SAMPLES = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'samples'
SAMPLE_IMG = SAMPLES / 'lesion.jpg'

client = TestClient(app)

def test_full_pipeline_html_report_generation():
    assert SAMPLE_IMG.exists(), 'Sample image missing'
    payload = {
        "subunit": "cheek_lateral"
    }
    files = {
        'file': ('lesion.jpg', SAMPLE_IMG.read_bytes(), 'image/jpeg'),
        'payload': (None, json.dumps(payload)),
    }
    r = client.post('/api/analyze', files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data['ok']
    html_url = data['artifacts']['report_html']
    html_resp = client.get(html_url)
    if html_resp.status_code == 404 and html_url.startswith('/api/artifact/demo/'):
        time.sleep(0.5)
        html_resp = client.get(html_url)
    assert html_resp.status_code == 200, html_resp.text
    html = html_resp.text
    assert '<html' in html.lower() and 'SurgicalAI' in html
    assert 'Mehdi' in html
