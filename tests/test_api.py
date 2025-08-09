from fastapi.testclient import TestClient
from surgicalai.api.server import app
from pathlib import Path

client = TestClient(app)


def test_ingest_analyze_plan():
    scan_path = Path('data/samples/face_mock.obj')
    with scan_path.open('rb') as f:
        res = client.post('/ingest', files={'scan': ('face_mock.obj', f, 'application/octet-stream')})
    run_id = res.json()['run_id']
    res = client.post('/analyze', params={'run_id': run_id})
    assert 'probs' in res.json()
    res = client.post('/plan', params={'run_id': run_id})
    assert 'flap' in res.json()
