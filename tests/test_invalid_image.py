from fastapi.testclient import TestClient
from server.http_api import app
import json

def test_invalid_image_type():
    client = TestClient(app)
    files = {
        'file': ('bad.txt', b'notanimage', 'text/plain'),
        'payload': (None, json.dumps({'subunit': 'cheek_lateral'}))
    }
    r = client.post('/api/analyze', files=files)
    assert r.status_code == 400


def test_corrupted_image_bytes():
    client = TestClient(app)
    # image/jpeg header but invalid body
    files = {
        'file': ('fake.jpg', b'12345', 'image/jpeg'),
        'payload': (None, json.dumps({'subunit': 'cheek_lateral'}))
    }
    r = client.post('/api/analyze', files=files)
    assert r.status_code == 400
