"""
Batch Evaluator
===============
Run evaluation across directories of predicted and ground-truth point clouds,
aggregate results into a DataFrame, and export JSON / CSV.
Supports parallel evaluation via multiprocessing.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation.evaluator import PointCloudEvaluator
from src.processing.io_utils import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


# ---- Module-level worker function for multiprocessing (must be picklable) ----
def _evaluate_pair_worker(pair_dict: Dict[str, str], config: dict, preprocess: bool) -> Dict[str, Any]:
    """Worker function for parallel evaluation (runs in subprocess)."""
    try:
        evaluator = PointCloudEvaluator(config)
        result = evaluator.evaluate_from_files(
            pair_dict["pred"], pair_dict["gt"], preprocess=preprocess
        )
        row = BatchEvaluator._flatten(result, {
            "pred": Path(pair_dict["pred"]),
            "gt": Path(pair_dict["gt"]),
        })
        return row
    except Exception as e:
        return {"sample": Path(pair_dict["pred"]).stem, "error": str(e)}


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
    # Sequential Batch Run
    # ------------------------------------------------------------------
    def evaluate_batch(
        self,
        pred_dir: str,
        gt_dir: str,
        preprocess: bool = True,
    ) -> pd.DataFrame:
        """Run evaluation on all pairs (sequential).

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
    # Parallel Batch Run
    # ------------------------------------------------------------------
    def evaluate_batch_parallel(
        self,
        pred_dir: str,
        gt_dir: str,
        preprocess: bool = True,
        n_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run evaluation on all pairs in parallel using multiprocessing.

        Parameters
        ----------
        pred_dir : str
        gt_dir : str
        preprocess : bool
        n_workers : int or None
            Number of parallel workers. Defaults to min(cpu_count, n_pairs).

        Returns
        -------
        pd.DataFrame
        """
        pairs = self.find_pairs(pred_dir, gt_dir)
        if not pairs:
            logger.warning("No matching pairs found.")
            return pd.DataFrame()

        # Serialise paths for pickling across processes
        pair_dicts = [
            {"pred": str(p["pred"]), "gt": str(p["gt"])} for p in pairs
        ]

        if n_workers is None:
            n_workers = min(mp.cpu_count(), len(pairs))
        n_workers = max(1, n_workers)

        logger.info(
            "Parallel evaluation: %d pairs across %d workers", len(pairs), n_workers
        )

        worker_fn = partial(
            _evaluate_pair_worker,
            config=self.config,
            preprocess=preprocess,
        )

        if n_workers == 1:
            # Fall back to sequential for single worker
            rows = [worker_fn(pd) for pd in tqdm(pair_dicts, desc="Evaluating", unit="pair")]
        else:
            with mp.Pool(processes=n_workers) as pool:
                rows = list(tqdm(
                    pool.imap(worker_fn, pair_dicts),
                    total=len(pair_dicts),
                    desc=f"Evaluating ({n_workers} workers)",
                    unit="pair",
                ))

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

        # Normal Consistency
        nc = result.get("normal_consistency", {})
        if nc:
            row["normal_consistency_mean"] = nc.get("mean")

        # EMD
        emd = result.get("emd", {})
        if emd:
            row["emd"] = emd.get("emd")

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
