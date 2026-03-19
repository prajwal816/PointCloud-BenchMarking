"""
Overlay Visualization
=====================
Render predicted and ground-truth point clouds side by side or overlaid.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Optional, List

logger = logging.getLogger(__name__)


def visualize_overlay(
    pred: np.ndarray,
    gt: np.ndarray,
    pred_color: Optional[List[float]] = None,
    gt_color: Optional[List[float]] = None,
    point_size: float = 2.0,
    window_name: str = "Pred (blue) vs GT (red)",
    save_path: Optional[str] = None,
) -> None:
    """Overlay two point clouds with distinct colours.

    Parameters
    ----------
    pred : np.ndarray, shape (N, 3)
    gt : np.ndarray, shape (M, 3)
    pred_color : RGB in [0, 1], default blue
    gt_color : RGB in [0, 1], default red
    point_size : float
    window_name : str
    save_path : str or None – save screenshot instead of interactive window
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.warning("open3d not installed – skipping overlay visualization.")
        return

    if pred_color is None:
        pred_color = [0.0, 0.6, 1.0]
    if gt_color is None:
        gt_color = [1.0, 0.3, 0.1]

    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(np.asarray(pred, dtype=np.float64))
    pcd_pred.paint_uniform_color(pred_color)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(np.asarray(gt, dtype=np.float64))
    pcd_gt.paint_uniform_color(gt_color)

    if save_path:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, visible=False)
        vis.add_geometry(pcd_pred)
        vis.add_geometry(pcd_gt)
        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.array([0.05, 0.05, 0.05])
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
        logger.info("Overlay saved to %s", save_path)
    else:
        o3d.visualization.draw_geometries(
            [pcd_pred, pcd_gt],
            window_name=window_name,
            point_show_normal=False,
        )
