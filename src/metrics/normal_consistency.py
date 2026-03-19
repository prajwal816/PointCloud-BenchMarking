"""
Normal Consistency
==================
Measures alignment of surface normals between predicted and ground-truth
point clouds at corresponding (nearest-neighbour) points.

    NC(P, Q) = (1/|P|) Σ |n_p · n_{nn(p,Q)}|

where nn(p, Q) is the nearest neighbour of p in Q.

Values close to 1.0 indicate strong normal alignment. Requires normals
to be present or estimated on both clouds.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Optional


def normal_consistency(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    pred_normals: Optional[np.ndarray] = None,
    gt_normals: Optional[np.ndarray] = None,
    estimate_if_missing: bool = True,
) -> Dict[str, float | np.ndarray]:
    """Compute Normal Consistency between two point clouds.

    Parameters
    ----------
    pred_points : np.ndarray, shape (N, 3)
    gt_points : np.ndarray, shape (M, 3)
    pred_normals : np.ndarray or None, shape (N, 3)
    gt_normals : np.ndarray or None, shape (M, 3)
    estimate_if_missing : bool
        If True, estimate normals via Open3D when not provided.

    Returns
    -------
    dict with keys:
        normal_consistency_mean – average |dot product| over all pred points
        normal_consistency_std  – standard deviation
        per_point_consistency   – (N,) array of per-point |dot products|
    """
    pred_points = np.asarray(pred_points, dtype=np.float64)
    gt_points = np.asarray(gt_points, dtype=np.float64)

    if pred_points.ndim != 2 or pred_points.shape[1] != 3:
        raise ValueError(f"pred_points must be (N, 3), got {pred_points.shape}")
    if gt_points.ndim != 2 or gt_points.shape[1] != 3:
        raise ValueError(f"gt_points must be (M, 3), got {gt_points.shape}")

    # Estimate normals if not provided
    if pred_normals is None or gt_normals is None:
        if estimate_if_missing:
            pred_normals, gt_normals = _estimate_normals(
                pred_points, gt_points, pred_normals, gt_normals
            )
        else:
            raise ValueError(
                "Normals are required. Set estimate_if_missing=True to auto-estimate."
            )

    pred_normals = np.asarray(pred_normals, dtype=np.float64)
    gt_normals = np.asarray(gt_normals, dtype=np.float64)

    # Normalise normals (safety)
    pred_normals = _safe_normalize(pred_normals)
    gt_normals = _safe_normalize(gt_normals)

    # Find nearest neighbours: pred → gt
    tree_gt = cKDTree(gt_points)
    _, nn_indices = tree_gt.query(pred_points, k=1)

    # Dot product of pred normals with corresponding gt normals
    gt_nn_normals = gt_normals[nn_indices]
    dot_products = np.abs(np.sum(pred_normals * gt_nn_normals, axis=1))

    # Clamp to [0, 1] for numerical safety
    dot_products = np.clip(dot_products, 0.0, 1.0)

    return {
        "normal_consistency_mean": float(np.mean(dot_products)),
        "normal_consistency_std": float(np.std(dot_products)),
        "normal_consistency_median": float(np.median(dot_products)),
        "per_point_consistency": dot_products,
    }


def _safe_normalize(normals: np.ndarray) -> np.ndarray:
    """Normalise vectors, handling zero-length gracefully."""
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return normals / norms


def _estimate_normals(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    pred_normals: Optional[np.ndarray],
    gt_normals: Optional[np.ndarray],
):
    """Estimate normals via Open3D where missing."""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d is required to estimate normals. "
            "Install with: pip install open3d, or provide normals explicitly."
        )

    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)

    if pred_normals is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred_points)
        pcd.estimate_normals(search_param=search_param)
        pcd.orient_normals_consistent_tangent_plane(k=15)
        pred_normals = np.asarray(pcd.normals)

    if gt_normals is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(gt_points)
        pcd.estimate_normals(search_param=search_param)
        pcd.orient_normals_consistent_tangent_plane(k=15)
        gt_normals = np.asarray(pcd.normals)

    return pred_normals, gt_normals
