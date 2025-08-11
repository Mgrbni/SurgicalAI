import pytest
from fastapi.testclient import TestClient
from server.server import app as full_app

@pytest.mark.skipif(True, reason='Requires OPENAI_API_KEY and network; skip in offline CI')
def test_infer_structured_schema():
    client = TestClient(full_app)
    resp = client.post('/api/infer', json={
        'prompt': 'Return a LesionReport for a 5mm pigmented lesion on right cheek',
        'json_schema': True
    })
    assert resp.status_code == 200
    data = resp.json()['data']
    assert 'diagnosis' in data
