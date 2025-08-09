# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import trimesh

from surgicalai.config import Config
from surgicalai.logging_utils import get_logger
from surgicalai.utils.io import load_npz, read_json
from surgicalai.utils.rendering import red_colormap

LOGGER = get_logger(__name__)


def run(case_dir: Path, config: Config) -> Path:
    mesh = load_npz(case_dir / "mesh.npz")
    heat = np.load(case_dir / "heatmap_vertex.npy")
    plan = read_json(case_dir / "flap_plan.json")
    verts = mesh["vertices"]
    faces = mesh["faces"]
    colors = (red_colormap(heat) * 255).astype(np.uint8)
    tri = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors)
    glb_path = case_dir / "surgical_plan.glb"
    tri.export(glb_path)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c=heat, cmap="Reds", s=2)
    pivot = np.array(plan["pivot"])
    arc = np.array(plan["arc"])
    ax.scatter(*pivot, color="blue", s=30)
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="green")
    ax.set_axis_off()
    fig.tight_layout()
    img_path = case_dir / "overview.png"
    fig.savefig(img_path)
    plt.close(fig)

    LOGGER.info("visualization complete")
    return glb_path
