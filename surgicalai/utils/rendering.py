"""Rendering utilities using Open3D if available."""
from __future__ import annotations
import numpy as np
from pathlib import Path

try:
    import open3d as o3d
except Exception:  # pragma: no cover
    o3d = None


def render_views(vertices: np.ndarray, faces: np.ndarray, img_size: int = 256, views: int = 4):
    """Render multiple views of a mesh. Returns list of images and camera info."""
    images = []
    cams = []
    if o3d is None:
        images.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
        cams.append({'R': np.eye(3), 't': np.zeros(3)})
        return images, cams
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    renderer = o3d.visualization.rendering.OffscreenRenderer(img_size, img_size)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    renderer.scene.add_geometry("mesh", mesh, mat)
    for i in range(views):
        angle = 2 * np.pi * i / views
        eye = [np.cos(angle)*2, np.sin(angle)*2, 0.5]
        center = [0.5,0.5,0.25]
        up = [0,0,1]
        renderer.setup_camera(60, center, eye, up)
        img = np.asarray(renderer.render_to_image())
        images.append(img)
        cams.append({'R': np.eye(3), 't': np.array(eye)})
    renderer.scene.clear_geometry()
    return images, cams
