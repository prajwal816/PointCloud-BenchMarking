"""
Hausdorff Distance
==================
Maximum nearest-neighbor deviation between two point clouds.

    HD(P, Q) = max( max_p min_q ‖p-q‖,  max_q min_p ‖q-p‖ )

Measures worst-case reconstruction error. Supports robust percentile variant.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict


def hausdorff_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    percentile: float = 100.0,
) -> Dict[str, float | np.ndarray]:
    """Compute symmetric Hausdorff Distance.

    Parameters
    ----------
    pred : np.ndarray, shape (N, 3)
        Predicted point cloud.
    gt : np.ndarray, shape (M, 3)
        Ground-truth point cloud.
    percentile : float
        Percentile for robust Hausdorff (100 = standard Hausdorff).

    Returns
    -------
    dict with keys:
        hausdorff          – symmetric Hausdorff distance
        hausdorff_forward  – directed HD from pred → gt
        hausdorff_backward – directed HD from gt → pred
        pred_to_gt         – per-point NN distances (N,)
        gt_to_pred         – per-point NN distances (M,)
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"pred must be (N, 3), got {pred.shape}")
    if gt.ndim != 2 or gt.shape[1] != 3:
        raise ValueError(f"gt must be (M, 3), got {gt.shape}")

    tree_gt = cKDTree(gt)
    tree_pred = cKDTree(pred)

    dists_p2g, _ = tree_gt.query(pred, k=1)
    dists_g2p, _ = tree_pred.query(gt, k=1)

    if percentile < 100.0:
        hd_forward = float(np.percentile(dists_p2g, percentile))
        hd_backward = float(np.percentile(dists_g2p, percentile))
    else:
        hd_forward = float(np.max(dists_p2g))
        hd_backward = float(np.max(dists_g2p))

    return {
        "hausdorff": max(hd_forward, hd_backward),
        "hausdorff_forward": hd_forward,
        "hausdorff_backward": hd_backward,
        "pred_to_gt": dists_p2g,
        "gt_to_pred": dists_g2p,
    }
