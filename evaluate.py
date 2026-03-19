#!/usr/bin/env python3
"""
evaluate.py — CLI for Point Cloud Benchmarking & Evaluation Framework
=====================================================================

Subcommands:
    evaluate   – Evaluate a single pred/gt pair
    batch      – Evaluate all matching pairs in two directories
    visualize  – Render overlay or error heatmap
    benchmark  – Run synthetic performance benchmarks
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import yaml
import numpy as np


def _load_config(config_path: str | None) -> dict:
    """Load YAML config, falling back to defaults."""
    default = Path(__file__).parent / "configs" / "default.yaml"
    path = Path(config_path) if config_path else default
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}


# ═══════════════════════════════════════════════════════════════
#  Root CLI Group
# ═══════════════════════════════════════════════════════════════
@click.group()
@click.option("--config", "-c", default=None, help="Path to YAML config file.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option("--log-file/--no-log-file", default=True, help="Enable/disable file logging (default: on).")
@click.pass_context
def cli(ctx, config, verbose, log_file):
    """Point Cloud Benchmarking & Evaluation Framework."""
    from src.logging_config import setup_logging

    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level, log_to_file=log_file, log_to_console=True)

    ctx.ensure_object(dict)
    ctx.obj["config"] = _load_config(config)


# ═══════════════════════════════════════════════════════════════
#  evaluate — single pair
# ═══════════════════════════════════════════════════════════════
@cli.command()
@click.option("--pred", required=True, help="Path to predicted point cloud.")
@click.option("--gt", required=True, help="Path to ground-truth point cloud.")
@click.option("--output", "-o", default=None, help="Save results to JSON file.")
@click.option("--no-preprocess", is_flag=True, help="Skip preprocessing.")
@click.pass_context
def evaluate(ctx, pred, gt, output, no_preprocess):
    """Evaluate a single predicted vs ground-truth point cloud."""
    from src.evaluation.evaluator import PointCloudEvaluator
    from src.evaluation.report import generate_report

    config = ctx.obj["config"]
    evaluator = PointCloudEvaluator(config)
    results = evaluator.evaluate_from_files(pred, gt, preprocess=not no_preprocess)

    # Print report
    report = generate_report(results)
    click.echo(report)

    # Save
    if output:
        serialisable = {k: v for k, v in results.items() if not k.startswith("_")}
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(serialisable, f, indent=2, default=str)
        click.echo(f"\n✓ Results saved to {output}")


# ═══════════════════════════════════════════════════════════════
#  batch — directory-level evaluation
# ═══════════════════════════════════════════════════════════════
@cli.command()
@click.option("--pred-dir", required=True, help="Directory of predicted point clouds.")
@click.option("--gt-dir", required=True, help="Directory of ground-truth point clouds.")
@click.option("--output", "-o", default="results/batch_results.json", help="Output file.")
@click.option("--format", "fmt", type=click.Choice(["json", "csv"]), default="json")
@click.option("--no-preprocess", is_flag=True)
@click.option("--parallel", "-p", is_flag=True, help="Enable parallel evaluation.")
@click.option("--workers", "-w", type=int, default=None, help="Number of parallel workers (default: auto).")
@click.pass_context
def batch(ctx, pred_dir, gt_dir, output, fmt, no_preprocess, parallel, workers):
    """Batch-evaluate all matching point clouds in two directories."""
    from src.evaluation.batch_evaluator import BatchEvaluator
    from src.evaluation.report import generate_report

    config = ctx.obj["config"]
    be = BatchEvaluator(config)

    if parallel:
        df = be.evaluate_batch_parallel(
            pred_dir, gt_dir,
            preprocess=not no_preprocess,
            n_workers=workers,
        )
    else:
        df = be.evaluate_batch(pred_dir, gt_dir, preprocess=not no_preprocess)

    if df.empty:
        click.echo("No matching pairs found.")
        return

    report = generate_report(df)
    click.echo(report)

    be.export(df, output, fmt=fmt)
    click.echo(f"\n✓ Results exported to {output}")


# ═══════════════════════════════════════════════════════════════
#  visualize — overlay / heatmap
# ═══════════════════════════════════════════════════════════════
@cli.command()
@click.option("--pred", required=True, help="Path to predicted point cloud.")
@click.option("--gt", required=True, help="Path to ground-truth point cloud.")
@click.option("--mode", type=click.Choice(["overlay", "heatmap", "both"]), default="both")
@click.option("--save-dir", default=None, help="Directory to save images (non-interactive).")
@click.pass_context
def visualize(ctx, pred, gt, mode, save_dir):
    """Visualize predicted vs ground-truth point clouds."""
    from src.processing.io_utils import load_point_cloud
    from src.visualization.overlay import visualize_overlay
    from src.visualization.heatmap import visualize_error_heatmap

    pred_pts = load_point_cloud(pred)
    gt_pts = load_point_cloud(gt)

    save_overlay = None
    save_heatmap = None
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_overlay = str(Path(save_dir) / "overlay.png")
        save_heatmap = str(Path(save_dir) / "heatmap.png")

    if mode in ("overlay", "both"):
        visualize_overlay(pred_pts, gt_pts, save_path=save_overlay)

    if mode in ("heatmap", "both"):
        visualize_error_heatmap(pred_pts, gt_pts, save_path=save_heatmap)

    click.echo("✓ Visualization complete.")


# ═══════════════════════════════════════════════════════════════
#  benchmark — synthetic performance profiling
# ═══════════════════════════════════════════════════════════════
@cli.command()
@click.option("--output", "-o", default="benchmarks/results.json", help="Output file.")
@click.option("--plot", is_flag=True, help="Generate benchmark plot.")
@click.pass_context
def benchmark(ctx, output, plot):
    """Run performance benchmarks on synthetic data."""
    import importlib
    spec = importlib.util.spec_from_file_location(
        "run_benchmarks",
        str(Path(__file__).parent / "benchmarks" / "run_benchmarks.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run_benchmarks(output_path=output, generate_plot=plot)


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cli()

