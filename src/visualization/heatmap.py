"""
Error Heatmap Visualization
============================
Colour each point in the predicted cloud by its nearest-neighbour distance
to the ground truth.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def visualize_error_heatmap(
    pred: np.ndarray,
    gt: np.ndarray,
    colormap: str = "jet",
    percentile_clip: float = 99.0,
    point_size: float = 2.0,
    window_name: str = "Error Heatmap",
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Render the predicted cloud coloured by per-point error.

    Parameters
    ----------
    pred : np.ndarray, shape (N, 3)
    gt : np.ndarray, shape (M, 3)
    colormap : str – matplotlib colormap name
    percentile_clip : float – clip distances above this percentile
    point_size : float
    window_name : str
    save_path : str or None – save screenshot

    Returns
    -------
    np.ndarray (N,) – per-point distances (for further analysis)
    """
    from scipy.spatial import cKDTree
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    # Compute per-point distances
    tree = cKDTree(gt)
    dists, _ = tree.query(pred, k=1)

    # Clip for visualisation clarity
    clip_val = np.percentile(dists, percentile_clip)
    dists_clipped = np.clip(dists, 0, clip_val)

    # Normalise to [0, 1]
    d_min, d_max = dists_clipped.min(), dists_clipped.max()
    if d_max - d_min > 0:
        normalised = (dists_clipped - d_min) / (d_max - d_min)
    else:
        normalised = np.zeros_like(dists_clipped)

    # Map to RGB via colormap
    cmap = cm.get_cmap(colormap)
    colors_rgba = cmap(normalised)
    colors_rgb = colors_rgba[:, :3]

    try:
        import open3d as o3d
    except ImportError:
        logger.warning("open3d not installed – returning distances only.")
        return dists

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

    if save_path:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, visible=False)
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.array([0.05, 0.05, 0.05])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
        logger.info("Heatmap saved to %s", save_path)
    else:
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=window_name,
        )

    return dists
