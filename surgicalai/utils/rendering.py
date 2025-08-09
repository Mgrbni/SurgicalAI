"""Rendering utilities using Open3D."""
from typing import List

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - handled at runtime
    o3d = None


def render_mesh(mesh_path: str, views: int = 1) -> List[object]:
    """Render a mesh from multiple views. Returns list of images or geometries."""
    if o3d is None:
        raise ImportError("open3d is required for rendering")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    return [mesh] * views
