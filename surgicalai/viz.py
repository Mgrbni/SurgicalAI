# SPDX-License-Identifier: Apache-2.0
"""Rendering helpers for PNG overlays and HTML reports."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template


def overlay_heatmap(mesh_data: Dict[str, Any], heatmap: np.ndarray, path: Path) -> None:
    coords = mesh_data["coordinates"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=heatmap, cmap="hot")
    fig.colorbar(sc)
    fig.savefig(path)
    plt.close(fig)


def overlay_flap(
    mesh_data: Dict[str, Any], flap_plan: Dict[str, Any], path: Path
) -> None:
    coords = mesh_data["coordinates"]
    fig, ax = plt.subplots()
    ax.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.3)
    vectors = np.array(flap_plan["flap_vectors"])
    origin = vectors.mean(axis=0)
    u = vectors[:, 0] - origin[0]
    v = vectors[:, 1] - origin[1]
    x0 = np.full_like(u, origin[0])
    y0 = np.full_like(v, origin[1])
    ax.quiver(x0, y0, u, v, angles="xy", scale_units="xy", scale=1)
    fig.savefig(path)
    plt.close(fig)


def export_report(images: Dict[str, Path], path: Path) -> None:
    encoded = {}
    for name, img_path in images.items():
        data = img_path.read_bytes()
        encoded[name] = base64.b64encode(data).decode("ascii")
    template = Template(
        """
        <html><body>
        <h1>SurgicalAI Report</h1>
        {% for name, data in images.items() %}
        <h2>{{ name }}</h2>
        <img src='data:image/png;base64,{{ data }}'/>
        {% endfor %}
        </body></html>
        """
    )
    html = template.render(images=encoded)
    Path(path).write_text(html)
