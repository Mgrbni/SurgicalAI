import sys, pathlib
from PIL import Image

# ensure package root is on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from surgicalai.io.polycam_loader import load_scan


def test_load_scan_obj():
    data = load_scan("data/samples/face_mock.obj")
    keys = {"vertices", "faces", "normals", "uvs", "texture", "pointcloud", "meta"}
    assert set(data.keys()) == keys
    assert data["vertices"].shape[1] == 3
    assert data["faces"].shape[1] == 3
    assert data["uvs"].shape[1] == 2
    if data["normals"] is not None:
        assert data["normals"].shape[1] == 3
    assert isinstance(data["texture"], Image.Image)
    assert data["pointcloud"] is None
