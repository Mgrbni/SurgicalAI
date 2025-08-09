from surgicalai.io.polycam_loader import load_scan
from pathlib import Path

def test_load_scan():
    data = load_scan(Path('data/samples/face_mock.obj'))
    assert 'vertices' in data and data['vertices'].shape[1] == 3
