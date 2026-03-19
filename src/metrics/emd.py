"""
Earth Mover's Distance (EMD)
============================
Also known as Wasserstein distance. Computes the optimal transport
cost between two equal-sized point sets.

    EMD(P, Q) = min_φ (1/|P|) Σ ‖p - φ(p)‖²

where φ is a bijection from P to Q.

Uses the linear sum assignment (Hungarian algorithm) from SciPy.

.. warning::
    EMD has O(n³) complexity. For clouds >10K points, consider
    downsampling first or using an approximate solver.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def earth_movers_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    max_points: int = 2048,
    subsample_seed: Optional[int] = None,
) -> Dict[str, float]:
    """Compute Earth Mover's Distance between two point clouds.

    Parameters
    ----------
    pred : np.ndarray, shape (N, 3)
        Predicted point cloud.
    gt : np.ndarray, shape (M, 3)
        Ground-truth point cloud.
    max_points : int
        If either cloud exceeds this, subsample to this size for
        tractable computation. Set to -1 to disable.
    subsample_seed : int or None
        RNG seed for reproducible subsampling.

    Returns
    -------
    dict with keys:
        emd            – total EMD (mean cost per point)
        emd_total      – total transport cost
        assignment     – (K, 2) array of matched index pairs
        n_points_used  – number of points used (after subsampling)
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"pred must be (N, 3), got {pred.shape}")
    if gt.ndim != 2 or gt.shape[1] != 3:
        raise ValueError(f"gt must be (M, 3), got {gt.shape}")

    rng = np.random.default_rng(subsample_seed)

    # Subsample to equal size if needed
    n_pred, n_gt = len(pred), len(gt)
    n_use = min(n_pred, n_gt)

    if max_points > 0 and n_use > max_points:
        logger.info(
            "EMD: subsampling %d → %d points (O(n³) constraint)",
            n_use,
            max_points,
        )
        n_use = max_points

    if n_pred > n_use:
        idx = rng.choice(n_pred, n_use, replace=False)
        pred = pred[idx]
    if n_gt > n_use:
        idx = rng.choice(n_gt, n_use, replace=False)
        gt = gt[idx]

    # Ensure equal size
    n = min(len(pred), len(gt))
    pred = pred[:n]
    gt = gt[:n]

    # Compute pairwise cost matrix
    logger.info("EMD: computing %d×%d cost matrix...", n, n)
    cost_matrix = cdist(pred, gt, metric="sqeuclidean")

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Matched costs
    matched_costs = cost_matrix[row_ind, col_ind]
    total_cost = float(np.sum(matched_costs))
    mean_cost = float(np.mean(matched_costs))

    return {
        "emd": mean_cost,
        "emd_total": total_cost,
        "emd_sqrt": float(np.mean(np.sqrt(matched_costs))),
        "n_points_used": n,
    }
