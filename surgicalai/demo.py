from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from surgicalai.config import Config, load_config
from surgicalai.logging_utils import get_logger
from surgicalai.utils.io import ensure_dir, save_npz, write_json
from surgicalai.analyze import engine as analyze_engine
from surgicalai.plan import rotational_flap
from surgicalai.validate import biomech
from surgicalai.visualize import overlay
from surgicalai.llm.openai_client import OpenAIClient
from surgicalai import report

LOGGER = get_logger(__name__)


def _synthetic_mesh() -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=50)
    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)


def _ensure_case(case_dir: Path) -> None:
    ensure_dir(case_dir)
    mesh_path = case_dir / "mesh.npz"
    if not mesh_path.exists():
        verts, faces = _synthetic_mesh()
        save_npz(mesh_path, vertices=verts, faces=faces)
        glb = trimesh.Trimesh(vertices=verts, faces=faces)
        glb.export(case_dir / "synthetic.glb")
    anatomy_dir = case_dir / "anatomy"
    anatomy_dir.mkdir(exist_ok=True)
    langer = anatomy_dir / "langer_lines.npy"
    if not langer.exists():
        np.save(langer, np.ones((1, 3), dtype=np.float32))


def run(out_dir: Path, with_llm: bool = False, model: Optional[str] = None) -> None:
    config = load_config()
    if with_llm:
        config.llm.enabled = True
    if model:
        config.llm.model = model
    _ensure_case(out_dir)
    analyze_engine.run(out_dir, config)
    rotational_flap.run(out_dir, config)
    biomech.run(out_dir, config)
    overlay.run(out_dir, config)
    narrative = None
    if config.llm.enabled:
        client = OpenAIClient(config.llm)
        narrative = client.generate_narrative(out_dir)
        write_json(out_dir / "narrative.json", narrative)
        (out_dir / "narrative.txt").write_text(str(narrative))
    report.generate(out_dir)
    LOGGER.info("demo complete")
