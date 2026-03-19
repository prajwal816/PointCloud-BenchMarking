"""
Chamfer Distance
================
Bidirectional average nearest-neighbor distance between two point clouds.

    CD(P, Q) = (1/|P|) Σ min‖p - q‖² + (1/|Q|) Σ min‖q - p‖²

Lower is better. Returns per-point distances for heatmap visualization.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict


def chamfer_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    squared: bool = False,
) -> Dict[str, float | np.ndarray]:
    """Compute bidirectional Chamfer Distance.

    Parameters
    ----------
    pred : np.ndarray, shape (N, 3)
        Predicted (reconstructed) point cloud.
    gt : np.ndarray, shape (M, 3)
        Ground-truth point cloud.
    squared : bool
        If True, return squared L2 distances (skip the sqrt).

    Returns
    -------
    dict with keys:
        chamfer_mean  – scalar mean CD
        chamfer_max   – scalar max CD
        pred_to_gt    – per-point distances from pred to gt  (N,)
        gt_to_pred    – per-point distances from gt to pred  (M,)
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"pred must be (N, 3), got {pred.shape}")
    if gt.ndim != 2 or gt.shape[1] != 3:
        raise ValueError(f"gt must be (M, 3), got {gt.shape}")

    tree_gt = cKDTree(gt)
    tree_pred = cKDTree(pred)

    # Forward: pred → gt
    dists_p2g, _ = tree_gt.query(pred, k=1)
    # Backward: gt → pred
    dists_g2p, _ = tree_pred.query(gt, k=1)

    if squared:
        dists_p2g = dists_p2g ** 2
        dists_g2p = dists_g2p ** 2

    mean_p2g = float(np.mean(dists_p2g))
    mean_g2p = float(np.mean(dists_g2p))

    return {
        "chamfer_mean": mean_p2g + mean_g2p,
        "chamfer_forward": mean_p2g,
        "chamfer_backward": mean_g2p,
        "chamfer_max": float(max(np.max(dists_p2g), np.max(dists_g2p))),
        "pred_to_gt": dists_p2g,
        "gt_to_pred": dists_g2p,
    }
