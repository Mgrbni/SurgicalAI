import json, pathlib
from fastapi.testclient import TestClient
from server.http_api import app

SAMPLES = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'samples'
SAMPLE_IMG = SAMPLES / 'lesion.jpg'

client = TestClient(app)

def test_demo_contract_and_artifacts():
    assert SAMPLE_IMG.exists(), 'Sample lesion image missing'
    payload = {'subunit': 'cheek_lateral'}
    files = {
        'file': ('lesion.jpg', SAMPLE_IMG.read_bytes(), 'image/jpeg'),
        'payload': (None, json.dumps(payload)),
    }
    r = client.post('/api/analyze', files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get('ok'), data
    # New contract fields
    assert 'run_id' in data and data['run_id']
    assert 'diagnosis' in data and isinstance(data['diagnosis'], dict)
    assert 'flap' in data and 'suggestion' in data['flap']
    artifacts_list = data.get('artifacts_list')
    assert artifacts_list and isinstance(artifacts_list, list)
    required = {"overlay","heatmap","metrics","plan","report"}
    names = {a['name'] for a in artifacts_list}
    assert required.issubset(names), names
    # Each artifact should be retrievable
    for a in artifacts_list:
        resp = client.get(f"/api/artifact/{a['path']}")
        assert resp.status_code == 200, f"Missing artifact {a['path']} status={resp.status_code}"


def test_health_endpoint():
    r = client.get('/api/health')
    assert r.status_code == 200
    assert r.json().get('ok') is True
