#!/usr/bin/env python3
"""
Benchmark Runner
================
Generate synthetic point clouds at multiple scales and profile
metric computation time, KD-tree build/query latency.

Can be run standalone or invoked by the CLI.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def run_benchmarks(
    output_path: str = "benchmarks/results.json",
    generate_plot: bool = False,
    point_counts: Optional[list] = None,
    num_trials: int = 5,
) -> dict:
    """Run performance benchmarks.

    Parameters
    ----------
    output_path : str – where to save results JSON.
    generate_plot : bool – also generate a bar chart.
    point_counts : list[int] or None – sizes to test.
    num_trials : int – repetitions for averaging.

    Returns
    -------
    dict – benchmark results.
    """
    from src.metrics.chamfer import chamfer_distance
    from src.metrics.hausdorff import hausdorff_distance
    from src.metrics.fscore import f_score
    from src.indexing.kdtree import KDTreeIndex
    from src.processing.io_utils import generate_synthetic_sphere

    if point_counts is None:
        point_counts = [1000, 5000, 10000, 50000, 100000, 500000]

    results = {
        "point_counts": point_counts,
        "num_trials": num_trials,
        "timings": {
            "Chamfer Distance": [],
            "Hausdorff Distance": [],
            "F-Score": [],
            "KD-Tree Build": [],
            "KD-Tree Query (k=1)": [],
        },
        "details": [],
    }

    print("\n" + "=" * 65)
    print("  Point Cloud Benchmarking — Performance Profile")
    print("=" * 65)
    print(f"  Trials per size: {num_trials}")
    print(f"  Sizes: {point_counts}")
    print("=" * 65)

    header = f"{'Points':>10} │ {'Chamfer':>10} │ {'Hausdorff':>10} │ {'F-Score':>10} │ {'KD Build':>10} │ {'KD Query':>10}"
    print(header)
    print("─" * len(header))

    for n in point_counts:
        trial_timings = {k: [] for k in results["timings"]}

        for trial in range(num_trials):
            seed = trial * 1000 + n
            pred = generate_synthetic_sphere(n, radius=1.0, noise_std=0.01, seed=seed)
            gt = generate_synthetic_sphere(n, radius=1.0, noise_std=0.0, seed=seed + 1)

            # Chamfer
            t0 = time.perf_counter()
            chamfer_distance(pred, gt)
            trial_timings["Chamfer Distance"].append((time.perf_counter() - t0) * 1000)

            # Hausdorff
            t0 = time.perf_counter()
            hausdorff_distance(pred, gt)
            trial_timings["Hausdorff Distance"].append((time.perf_counter() - t0) * 1000)

            # F-Score
            t0 = time.perf_counter()
            f_score(pred, gt, thresholds=0.01)
            trial_timings["F-Score"].append((time.perf_counter() - t0) * 1000)

            # KD-Tree Build
            t0 = time.perf_counter()
            tree = KDTreeIndex(gt, backend="scipy")
            trial_timings["KD-Tree Build"].append((time.perf_counter() - t0) * 1000)

            # KD-Tree Query
            t0 = time.perf_counter()
            tree.query_knn(pred, k=1)
            trial_timings["KD-Tree Query (k=1)"].append((time.perf_counter() - t0) * 1000)

        # Average across trials
        detail = {"n_points": n}
        row_vals = []
        for key in results["timings"]:
            avg = float(np.mean(trial_timings[key]))
            results["timings"][key].append(avg)
            detail[key] = {
                "mean_ms": avg,
                "std_ms": float(np.std(trial_timings[key])),
                "min_ms": float(np.min(trial_timings[key])),
                "max_ms": float(np.max(trial_timings[key])),
            }
            row_vals.append(f"{avg:>9.2f}ms")

        results["details"].append(detail)
        print(f"{n:>10,} │ {'│'.join(row_vals)}")

    print("=" * len(header))

    # Save
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")

    # Plot
    if generate_plot:
        try:
            from src.visualization.plots import plot_benchmark_results
            plot_path = str(out_path.with_suffix(".png"))
            plot_benchmark_results(results, save_path=plot_path)
            print(f"✓ Plot saved to {plot_path}")
        except Exception as e:
            logger.error("Failed to generate plot: %s", e)

    return results


if __name__ == "__main__":
    run_benchmarks()
