"""
Point Cloud Evaluator
=====================
Run all configured metrics on a single predicted / ground-truth pair,
with optional preprocessing.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Dict, Any, Optional

from src.metrics import chamfer_distance, hausdorff_distance, f_score
from src.processing.downsampling import voxel_downsample
from src.processing.outlier_removal import statistical_outlier_removal

logger = logging.getLogger(__name__)


class PointCloudEvaluator:
    """Evaluate a single predicted point cloud against ground truth.

    Parameters
    ----------
    config : dict
        Full evaluation configuration (from YAML).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def preprocess(self, points: np.ndarray) -> np.ndarray:
        """Apply configured preprocessing steps."""
        cfg = self.config.get("preprocessing", {})
        if not cfg.get("enabled", False):
            return points

        # Voxel downsample
        voxel_cfg = cfg.get("voxel_downsample", {})
        if voxel_cfg.get("enabled", False):
            result = voxel_downsample(points, voxel_size=voxel_cfg.get("voxel_size", 0.02))
            logger.info(
                "Voxel downsample: %d → %d points (%.1f%% reduction)",
                len(points),
                len(result["points"]),
                result["reduction_ratio"] * 100,
            )
            points = result["points"]

        # Outlier removal
        outlier_cfg = cfg.get("outlier_removal", {})
        if outlier_cfg.get("enabled", False):
            result = statistical_outlier_removal(
                points,
                nb_neighbors=outlier_cfg.get("nb_neighbors", 20),
                std_ratio=outlier_cfg.get("std_ratio", 2.0),
            )
            logger.info(
                "Outlier removal: removed %d points", result["n_removed"]
            )
            points = result["points"]

        return points

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        preprocess: bool = True,
    ) -> Dict[str, Any]:
        """Run all enabled metrics.

        Parameters
        ----------
        pred : np.ndarray, shape (N, 3)
        gt : np.ndarray, shape (M, 3)
        preprocess : bool
            Whether to apply preprocessing.

        Returns
        -------
        dict – combined metric results.
        """
        pred = np.asarray(pred, dtype=np.float64)
        gt = np.asarray(gt, dtype=np.float64)

        if preprocess:
            pred = self.preprocess(pred)
            gt = self.preprocess(gt)

        results: Dict[str, Any] = {
            "n_pred": len(pred),
            "n_gt": len(gt),
        }
        metrics_cfg = self.config.get("metrics", {})

        # Chamfer Distance
        chamfer_cfg = metrics_cfg.get("chamfer", {})
        if chamfer_cfg.get("enabled", True):
            cd = chamfer_distance(
                pred, gt, squared=chamfer_cfg.get("squared", False)
            )
            results["chamfer"] = {
                "mean": cd["chamfer_mean"],
                "forward": cd["chamfer_forward"],
                "backward": cd["chamfer_backward"],
                "max": cd["chamfer_max"],
            }
            results["_pred_to_gt_dists"] = cd["pred_to_gt"]
            results["_gt_to_pred_dists"] = cd["gt_to_pred"]

        # Hausdorff Distance
        hausdorff_cfg = metrics_cfg.get("hausdorff", {})
        if hausdorff_cfg.get("enabled", True):
            hd = hausdorff_distance(
                pred, gt, percentile=hausdorff_cfg.get("percentile", 100)
            )
            results["hausdorff"] = {
                "symmetric": hd["hausdorff"],
                "forward": hd["hausdorff_forward"],
                "backward": hd["hausdorff_backward"],
            }

        # F-Score
        fscore_cfg = metrics_cfg.get("fscore", {})
        if fscore_cfg.get("enabled", True):
            thresholds = fscore_cfg.get("thresholds", [0.01])
            fs = f_score(pred, gt, thresholds=thresholds)
            results["fscore"] = fs["results"]

        return results

    def evaluate_from_files(
        self,
        pred_path: str,
        gt_path: str,
        preprocess: bool = True,
    ) -> Dict[str, Any]:
        """Convenience: load files and evaluate."""
        from src.processing.io_utils import load_point_cloud

        pred = load_point_cloud(pred_path)
        gt = load_point_cloud(gt_path)
        results = self.evaluate(pred, gt, preprocess=preprocess)
        results["pred_file"] = str(pred_path)
        results["gt_file"] = str(gt_path)
        return results
