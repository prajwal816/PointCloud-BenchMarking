"""
Statistical Outlier Removal
============================
Remove points whose mean distance to k-nearest neighbours exceeds
μ + std_ratio · σ.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def statistical_outlier_removal(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> dict:
    """Remove statistical outliers from a point cloud.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    nb_neighbors : int
        Number of nearest neighbours to consider.
    std_ratio : float
        Standard-deviation multiplier for the distance threshold.

    Returns
    -------
    dict:
        points   – (M, 3) cleaned point cloud
        mask     – (N,) boolean mask of inliers
        n_removed – number of points removed
    """
    points = np.asarray(points, dtype=np.float64)
    tree = cKDTree(points)

    # Query k+1 because the first neighbour is the point itself
    dists, _ = tree.query(points, k=nb_neighbors + 1)
    # Mean distance to the k nearest neighbours (skip self at index 0)
    mean_dists = np.mean(dists[:, 1:], axis=1)

    global_mean = np.mean(mean_dists)
    global_std = np.std(mean_dists)
    threshold = global_mean + std_ratio * global_std

    mask = mean_dists < threshold
    return {
        "points": points[mask],
        "mask": mask,
        "n_removed": int(np.sum(~mask)),
        "threshold": float(threshold),
    }
