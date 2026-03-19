"""
Voxel Downsampling
==================
Reduce point cloud density by averaging points within each voxel cell.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float = 0.02,
    normals: Optional[np.ndarray] = None,
) -> dict:
    """Down-sample a point cloud using a voxel grid.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    voxel_size : float
        Edge length of each voxel cell (metres).
    normals : np.ndarray or None, shape (N, 3)
        Optional per-point normals to average inside voxels.

    Returns
    -------
    dict:
        points  – (M, 3) downsampled points
        normals – (M, 3) downsampled normals (or None)
        reduction_ratio – fraction of points removed
    """
    points = np.asarray(points, dtype=np.float64)
    n_original = len(points)

    # Quantise into voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)

    # Use structured array for unique voxel identification
    dtype = np.dtype([("x", np.int64), ("y", np.int64), ("z", np.int64)])
    structured = np.empty(len(voxel_indices), dtype=dtype)
    structured["x"] = voxel_indices[:, 0]
    structured["y"] = voxel_indices[:, 1]
    structured["z"] = voxel_indices[:, 2]

    _, inverse, counts = np.unique(structured, return_inverse=True, return_counts=True)

    # Average positions within each voxel
    n_voxels = len(counts)
    ds_points = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(ds_points, inverse, points)
    ds_points /= counts[:, None]

    ds_normals = None
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)
        ds_normals = np.zeros((n_voxels, 3), dtype=np.float64)
        np.add.at(ds_normals, inverse, normals)
        # Re-normalise
        norms = np.linalg.norm(ds_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        ds_normals /= norms

    return {
        "points": ds_points,
        "normals": ds_normals,
        "reduction_ratio": 1.0 - n_voxels / n_original if n_original > 0 else 0.0,
    }
