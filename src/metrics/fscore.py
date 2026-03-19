"""
F-Score
=======
Precision, Recall, and F1 at a given distance threshold τ.

    Precision(τ) = |{p ∈ P : min_q ‖p-q‖ < τ}| / |P|
    Recall(τ)    = |{q ∈ Q : min_p ‖q-p‖ < τ}| / |Q|
    F1(τ)        = 2 · Precision · Recall / (Precision + Recall)

Higher is better. Threshold τ controls strictness.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Union


def f_score(
    pred: np.ndarray,
    gt: np.ndarray,
    thresholds: Union[float, List[float]] = 0.01,
) -> Dict[str, object]:
    """Compute F-score at one or more distance thresholds.

    Parameters
    ----------
    pred : np.ndarray, shape (N, 3)
        Predicted point cloud.
    gt : np.ndarray, shape (M, 3)
        Ground-truth point cloud.
    thresholds : float or list[float]
        Distance threshold(s) τ for classifying a point as "matched".

    Returns
    -------
    dict with keys:
        results       – list of {threshold, precision, recall, f1}
        pred_to_gt    – per-point NN distances (N,)
        gt_to_pred    – per-point NN distances (M,)
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"pred must be (N, 3), got {pred.shape}")
    if gt.ndim != 2 or gt.shape[1] != 3:
        raise ValueError(f"gt must be (M, 3), got {gt.shape}")

    if isinstance(thresholds, (int, float)):
        thresholds = [float(thresholds)]

    tree_gt = cKDTree(gt)
    tree_pred = cKDTree(pred)

    dists_p2g, _ = tree_gt.query(pred, k=1)
    dists_g2p, _ = tree_pred.query(gt, k=1)

    results = []
    for tau in thresholds:
        precision = float(np.mean(dists_p2g < tau))
        recall = float(np.mean(dists_g2p < tau))

        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        results.append({
            "threshold": tau,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return {
        "results": results,
        "pred_to_gt": dists_p2g,
        "gt_to_pred": dists_g2p,
    }
