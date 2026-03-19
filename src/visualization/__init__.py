"""Visualization: overlays, error heatmaps, distribution plots."""

from .overlay import visualize_overlay
from .heatmap import visualize_error_heatmap
from .plots import plot_metric_distributions, plot_precision_recall, plot_benchmark_results

__all__ = [
    "visualize_overlay",
    "visualize_error_heatmap",
    "plot_metric_distributions",
    "plot_precision_recall",
    "plot_benchmark_results",
]
