"""
Batch Evaluator
===============
Run evaluation across directories of predicted and ground-truth point clouds,
aggregate results into a DataFrame, and export JSON / CSV.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation.evaluator import PointCloudEvaluator
from src.processing.io_utils import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class BatchEvaluator:
    """Evaluate all matching point cloud pairs in two directories.

    Matching is by filename stem: files with the same name (ignoring
    extension) in ``pred_dir`` and ``gt_dir`` are paired.

    Parameters
    ----------
    config : dict
        Full evaluation configuration.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._evaluator = PointCloudEvaluator(config)

    # ------------------------------------------------------------------
    # Pairing
    # ------------------------------------------------------------------
    @staticmethod
    def find_pairs(pred_dir: str, gt_dir: str) -> List[Dict[str, Path]]:
        """Find matching pred / gt files by stem."""
        pred_dir = Path(pred_dir)
        gt_dir = Path(gt_dir)

        gt_map: Dict[str, Path] = {}
        for f in gt_dir.iterdir():
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                gt_map[f.stem] = f

        pairs = []
        for f in sorted(pred_dir.iterdir()):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.stem in gt_map:
                pairs.append({"pred": f, "gt": gt_map[f.stem]})

        logger.info("Found %d matching pairs", len(pairs))
        return pairs

    # ------------------------------------------------------------------
    # Batch Run
    # ------------------------------------------------------------------
    def evaluate_batch(
        self,
        pred_dir: str,
        gt_dir: str,
        preprocess: bool = True,
    ) -> pd.DataFrame:
        """Run evaluation on all pairs.

        Returns
        -------
        pd.DataFrame – one row per sample with scalar metric columns.
        """
        pairs = self.find_pairs(pred_dir, gt_dir)
        if not pairs:
            logger.warning("No matching pairs found.")
            return pd.DataFrame()

        rows = []
        for pair in tqdm(pairs, desc="Evaluating", unit="pair"):
            try:
                result = self._evaluator.evaluate_from_files(
                    str(pair["pred"]), str(pair["gt"]), preprocess=preprocess
                )
                row = self._flatten(result, pair)
                rows.append(row)
            except Exception as e:
                logger.error("Error evaluating %s: %s", pair["pred"].name, e)
                rows.append({"sample": pair["pred"].stem, "error": str(e)})

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Flatten dict for DataFrame
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten(result: Dict[str, Any], pair: Dict[str, Path]) -> Dict[str, Any]:
        row: Dict[str, Any] = {"sample": pair["pred"].stem}
        row["n_pred"] = result.get("n_pred", 0)
        row["n_gt"] = result.get("n_gt", 0)

        # Chamfer
        cd = result.get("chamfer", {})
        row["chamfer_mean"] = cd.get("mean")
        row["chamfer_forward"] = cd.get("forward")
        row["chamfer_backward"] = cd.get("backward")
        row["chamfer_max"] = cd.get("max")

        # Hausdorff
        hd = result.get("hausdorff", {})
        row["hausdorff_symmetric"] = hd.get("symmetric")
        row["hausdorff_forward"] = hd.get("forward")
        row["hausdorff_backward"] = hd.get("backward")

        # F-Scores
        for entry in result.get("fscore", []):
            tau = entry["threshold"]
            row[f"precision@{tau}"] = entry["precision"]
            row[f"recall@{tau}"] = entry["recall"]
            row[f"f1@{tau}"] = entry["f1"]

        return row

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export(
        self,
        df: pd.DataFrame,
        output_path: str,
        fmt: str = "json",
    ) -> None:
        """Save batch results to file.

        Parameters
        ----------
        df : pd.DataFrame
        output_path : str
        fmt : str – ``"json"`` or ``"csv"``
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "csv":
            df.to_csv(output_path, index=False)
        else:
            df.to_json(output_path, orient="records", indent=2)

        logger.info("Exported %d results to %s", len(df), output_path)
