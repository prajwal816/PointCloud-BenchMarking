"""
Surface Reconstruction
======================
Poisson and Ball-Pivoting surface reconstruction via Open3D.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _ensure_open3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        raise ImportError(
            "open3d is required for surface reconstruction. "
            "Install with: pip install open3d"
        )


def _to_o3d_pcd(points: np.ndarray, normals: Optional[np.ndarray] = None):
    """Convert NumPy arrays to an Open3D PointCloud."""
    o3d = _ensure_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    return pcd


def poisson_reconstruction(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    depth: int = 9,
    estimate_normals: bool = True,
) -> dict:
    """Poisson surface reconstruction.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    normals : np.ndarray or None, shape (N, 3)
    depth : int
        Octree depth for Poisson reconstruction.
    estimate_normals : bool
        Estimate normals if not provided.

    Returns
    -------
    dict:
        mesh      – Open3D TriangleMesh
        vertices  – (V, 3) mesh vertex positions
        triangles – (F, 3) face indices
        densities – per-vertex density values
    """
    o3d = _ensure_open3d()
    pcd = _to_o3d_pcd(points, normals)

    if normals is None and estimate_normals:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
        logger.info("Estimated and oriented normals for %d points", len(points))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    densities = np.asarray(densities)

    logger.info(
        "Poisson reconstruction: %d vertices, %d triangles",
        len(mesh.vertices),
        len(mesh.triangles),
    )

    return {
        "mesh": mesh,
        "vertices": np.asarray(mesh.vertices),
        "triangles": np.asarray(mesh.triangles),
        "densities": densities,
    }


def ball_pivoting_reconstruction(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    radii: Optional[list] = None,
    estimate_normals: bool = True,
) -> dict:
    """Ball-Pivoting Algorithm (BPA) surface reconstruction.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    normals : np.ndarray or None, shape (N, 3)
    radii : list[float] or None
        Ball radii. If None, automatically estimated.
    estimate_normals : bool
        Estimate normals if not provided.

    Returns
    -------
    dict:
        mesh      – Open3D TriangleMesh
        vertices  – (V, 3)
        triangles – (F, 3)
    """
    o3d = _ensure_open3d()
    pcd = _to_o3d_pcd(points, normals)

    if normals is None and estimate_normals:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

    if radii is None:
        # Estimate from average nearest-neighbour distance
        dists = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(dists)
        radii = [avg_dist * f for f in [0.5, 1.0, 2.0, 4.0]]

    radii_vec = o3d.utility.DoubleVector(radii)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, radii_vec
    )

    logger.info(
        "BPA reconstruction: %d vertices, %d triangles",
        len(mesh.vertices),
        len(mesh.triangles),
    )

    return {
        "mesh": mesh,
        "vertices": np.asarray(mesh.vertices),
        "triangles": np.asarray(mesh.triangles),
    }
