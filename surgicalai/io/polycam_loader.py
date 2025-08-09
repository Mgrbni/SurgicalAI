"""Utilities for loading Polycam scans in various formats."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None  # type: ignore


def _synthesize_texture(color: int = 128) -> Image.Image:
    """Create a tiny neutral texture as placeholder."""
    img = Image.new("RGB", (2, 2), (color, color, color))
    return img


def load_scan(path: str) -> Dict[str, Any]:
    """Load mesh or point cloud produced by Polycam.

    Parameters
    ----------
    path: str
        Path to a ``.obj`` / ``.glb`` / ``.ply`` file.

    Returns
    -------
    dict
        Dictionary with keys ``vertices``, ``faces``, ``normals``, ``uvs``,
        ``texture``, ``pointcloud``, ``meta``.
    """
    if trimesh is None:
        raise ImportError("trimesh is required for load_scan")

    ext = os.path.splitext(path)[1].lower()
    mesh: Optional[trimesh.Trimesh] = None
    pointcloud = None

    try:
        mesh = trimesh.load(path, force="mesh")  # type: ignore[arg-type]
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump().sum()  # merge into single mesh
    except Exception:
        mesh = None

    if mesh is None or mesh.vertices is None or len(mesh.vertices) == 0:
        try:
            import open3d as o3d  # type: ignore
        except Exception as exc:  # pragma: no cover - handled gracefully
            raise ValueError(f"Failed to load mesh from {path}: {exc}") from exc
        pcd = o3d.io.read_point_cloud(path)
        pointcloud = np.asarray(pcd.points, dtype=np.float32)
        vertices = np.zeros((0, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.int32)
        normals = None
        uvs = None
        texture = _synthesize_texture()
    else:
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        normals = None
        if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(mesh.vertices):
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        uvs = None
        texture = None
        if hasattr(mesh, "visual") and getattr(mesh.visual, "uv", None) is not None:
            uvs = np.asarray(mesh.visual.uv, dtype=np.float32)
            try:
                image = mesh.visual.material.image
                if isinstance(image, Image.Image):
                    texture = image
                else:
                    texture = Image.fromarray(np.array(image))
            except Exception:
                texture = None
        if texture is None:
            texture = _synthesize_texture()

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "uvs": uvs,
        "texture": texture,
        "pointcloud": pointcloud,
        "meta": {"path": path, "ext": ext},
    }


__all__ = ["load_scan"]
