from __future__ import annotations

from pathlib import Path

import pytest
import trimesh


@pytest.fixture(scope="session")
def synthetic_mesh() -> Path:
    samples = Path("samples")
    samples.mkdir(exist_ok=True)
    mesh_path = samples / "synthetic_face.obj"
    if not mesh_path.exists():
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        mesh.export(mesh_path)
    return mesh_path
