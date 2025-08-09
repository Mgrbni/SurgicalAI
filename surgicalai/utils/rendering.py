"""Rendering utilities using Open3D."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - handled at runtime
    o3d = None  # type: ignore


def render_views(mesh: Dict[str, np.ndarray], n_views: int = 8, img_size: int = 512) -> List[Dict[str, np.ndarray]]:
    """Render multiple views around a mesh.

    Parameters
    ----------
    mesh: dict
        Dictionary returned by :func:`surgicalai.io.polycam_loader.load_scan`.
    n_views: int
        Number of views to render.
    img_size: int
        Size of rendered square image in pixels.

    Returns
    -------
    list of dict
        Each item has keys ``image`` (uint8 HxWx3), ``K`` (3x3), ``Rt`` (4x4),
        and ``view_id``.
    """
    if o3d is None:
        raise ImportError("open3d is required for rendering")

    geom = o3d.geometry.TriangleMesh()
    geom.vertices = o3d.utility.Vector3dVector(mesh["vertices"])
    geom.triangles = o3d.utility.Vector3iVector(mesh["faces"])
    if mesh.get("normals") is not None:
        geom.vertex_normals = o3d.utility.Vector3dVector(mesh["normals"])
    else:
        geom.compute_vertex_normals()

    renderer = o3d.visualization.rendering.OffscreenRenderer(img_size, img_size)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", geom, mat)

    center = geom.get_center()
    extent = geom.get_max_bound() - geom.get_min_bound()
    radius = float(np.linalg.norm(extent)) / 2.0

    fov = 60.0
    fx = fy = 0.5 * img_size / np.tan(np.deg2rad(fov / 2.0))
    cx = cy = img_size / 2.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    outputs: List[Dict[str, np.ndarray]] = []
    for vid in range(n_views):
        theta = 2 * np.pi * vid / n_views
        cam_pos = center + 2.5 * radius * np.array([np.cos(theta), 0.5, np.sin(theta)])
        up = np.array([0.0, 1.0, 0.0])
        z = cam_pos - center
        z /= np.linalg.norm(z)
        x_axis = np.cross(up, z)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z, x_axis)
        R = np.stack([x_axis, y_axis, z], axis=0)
        t = -R @ cam_pos
        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = R
        Rt[:3, 3] = t
        renderer.scene.camera.look_at(center, cam_pos, up)
        img = np.asarray(renderer.render_to_image())
        outputs.append({"image": img, "K": K, "Rt": Rt, "view_id": vid})

    renderer.scene.clear_geometry()
    return outputs


__all__ = ["render_views"]
