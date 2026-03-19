"""
Metric Plots
=============
Matplotlib-based charts: error distributions, precision-recall, benchmark bars.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_metric_distributions(
    pred_to_gt: np.ndarray,
    gt_to_pred: np.ndarray,
    title: str = "Nearest-Neighbour Distance Distribution",
    save_path: Optional[str] = None,
    dpi: int = 150,
    figsize: tuple = (12, 5),
) -> None:
    """Plot histograms of per-point nearest-neighbour distances.

    Parameters
    ----------
    pred_to_gt : np.ndarray (N,) distances from pred to gt
    gt_to_pred : np.ndarray (M,) distances from gt to pred
    title : str
    save_path : str or None
    dpi : int
    figsize : tuple
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].hist(pred_to_gt, bins=100, color="#1e90ff", alpha=0.85, edgecolor="none")
    axes[0].set_title("Pred → GT distances")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Count")
    axes[0].axvline(np.mean(pred_to_gt), color="red", ls="--", label=f"mean={np.mean(pred_to_gt):.4f}")
    axes[0].legend()

    axes[1].hist(gt_to_pred, bins=100, color="#ff6347", alpha=0.85, edgecolor="none")
    axes[1].set_title("GT → Pred distances")
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Count")
    axes[1].axvline(np.mean(gt_to_pred), color="blue", ls="--", label=f"mean={np.mean(gt_to_pred):.4f}")
    axes[1].legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Distribution plot saved to %s", save_path)
    plt.close(fig)


def plot_precision_recall(
    fscore_results: List[Dict[str, float]],
    title: str = "Precision / Recall / F1 vs Threshold",
    save_path: Optional[str] = None,
    dpi: int = 150,
    figsize: tuple = (8, 5),
) -> None:
    """Plot precision, recall, and F1 across distance thresholds.

    Parameters
    ----------
    fscore_results : list of {threshold, precision, recall, f1}
    title : str
    save_path : str or None
    """
    thresholds = [r["threshold"] for r in fscore_results]
    precision = [r["precision"] for r in fscore_results]
    recall = [r["recall"] for r in fscore_results]
    f1 = [r["f1"] for r in fscore_results]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(thresholds, precision, "o-", label="Precision", color="#2196F3", linewidth=2)
    ax.plot(thresholds, recall, "s-", label="Recall", color="#4CAF50", linewidth=2)
    ax.plot(thresholds, f1, "D-", label="F1", color="#FF9800", linewidth=2)

    ax.set_xlabel("Distance Threshold (τ)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("PR plot saved to %s", save_path)
    plt.close(fig)


def plot_benchmark_results(
    benchmark_data: Dict[str, Any],
    title: str = "Benchmark: Metric Computation Time",
    save_path: Optional[str] = None,
    dpi: int = 150,
    figsize: tuple = (12, 6),
) -> None:
    """Plot benchmark timing results.

    Parameters
    ----------
    benchmark_data : dict with keys:
        point_counts – list of int
        timings     – dict[metric_name → list of float (ms)]
    title : str
    save_path : str or None
    """
    point_counts = benchmark_data["point_counts"]
    timings = benchmark_data["timings"]

    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.2
    x = np.arange(len(point_counts))
    colors = ["#1e90ff", "#ff6347", "#32cd32", "#ffa500"]

    for i, (metric_name, times) in enumerate(timings.items()):
        offset = (i - len(timings) / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            times,
            bar_width,
            label=metric_name,
            color=colors[i % len(colors)],
            alpha=0.85,
        )

    ax.set_xlabel("Number of Points", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in point_counts])
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Benchmark plot saved to %s", save_path)
    plt.close(fig)
