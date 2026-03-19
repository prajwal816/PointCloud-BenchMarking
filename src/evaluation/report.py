"""
Report Generator
================
Pretty-print and save evaluation summaries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

logger = logging.getLogger(__name__)


def generate_report(
    results: pd.DataFrame | Dict[str, Any] | List[Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Point Cloud Evaluation Report",
) -> str:
    """Generate a formatted evaluation report.

    Parameters
    ----------
    results : DataFrame, dict, or list of dicts
    output_path : str or None – if provided, save report to file.
    title : str

    Returns
    -------
    str – formatted report text.
    """
    lines: list[str] = []
    sep = "=" * 60
    lines.append(sep)
    lines.append(f"  {title}")
    lines.append(sep)
    lines.append("")

    if isinstance(results, pd.DataFrame):
        _report_dataframe(results, lines)
    elif isinstance(results, list):
        df = pd.DataFrame(results)
        _report_dataframe(df, lines)
    elif isinstance(results, dict):
        _report_single(results, lines)

    lines.append(sep)
    report_text = "\n".join(lines)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_text, encoding="utf-8")
        logger.info("Report saved to %s", path)

    return report_text


def _report_single(result: Dict[str, Any], lines: list[str]) -> None:
    """Format a single evaluation result."""
    if "pred_file" in result:
        lines.append(f"  Predicted : {result['pred_file']}")
        lines.append(f"  Ground Truth: {result['gt_file']}")
    lines.append(f"  Points (pred): {result.get('n_pred', '?')}")
    lines.append(f"  Points (gt)  : {result.get('n_gt', '?')}")
    lines.append("")

    # Chamfer
    cd = result.get("chamfer", {})
    if cd:
        lines.append("  -- Chamfer Distance --")
        lines.append(f"    Mean (bidirectional) : {cd.get('mean', 'N/A'):.6f}")
        lines.append(f"    Forward  (pred->gt)  : {cd.get('forward', 'N/A'):.6f}")
        lines.append(f"    Backward (gt->pred)  : {cd.get('backward', 'N/A'):.6f}")
        lines.append(f"    Max                  : {cd.get('max', 'N/A'):.6f}")
        lines.append("")

    # Hausdorff
    hd = result.get("hausdorff", {})
    if hd:
        lines.append("  -- Hausdorff Distance --")
        lines.append(f"    Symmetric : {hd.get('symmetric', 'N/A'):.6f}")
        lines.append(f"    Forward   : {hd.get('forward', 'N/A'):.6f}")
        lines.append(f"    Backward  : {hd.get('backward', 'N/A'):.6f}")
        lines.append("")

    # F-Score
    fs = result.get("fscore", [])
    if fs:
        lines.append("  -- F-Score --")
        lines.append(f"    {'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
        lines.append(f"    {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
        for entry in fs:
            lines.append(
                f"    {entry['threshold']:>10.4f}  "
                f"{entry['precision']:>10.4f}  "
                f"{entry['recall']:>10.4f}  "
                f"{entry['f1']:>10.4f}"
            )
        lines.append("")


def _report_dataframe(df: pd.DataFrame, lines: list[str]) -> None:
    """Format batch results summary."""
    lines.append(f"  Samples evaluated: {len(df)}")
    lines.append("")

    # Summary statistics for key columns
    metric_cols = [c for c in df.columns if c not in ("sample", "error", "n_pred", "n_gt")]
    if metric_cols:
        lines.append("  -- Summary Statistics --")
        lines.append(
            f"    {'Metric':<28}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}"
        )
        lines.append(f"    {'-' * 28}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
        for col in metric_cols:
            if df[col].dtype in ("float64", "float32", "int64"):
                series = df[col].dropna()
                if len(series) > 0:
                    lines.append(
                        f"    {col:<28}  "
                        f"{series.mean():>10.6f}  "
                        f"{series.std():>10.6f}  "
                        f"{series.min():>10.6f}  "
                        f"{series.max():>10.6f}"
                    )
        lines.append("")

    # Per-sample breakdown
    lines.append("  -- Per-Sample Results --")
    lines.append(df.to_string(index=False, max_rows=50))
    lines.append("")
