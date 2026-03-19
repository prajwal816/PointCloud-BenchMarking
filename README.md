# Point Cloud Benchmarking & Evaluation Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A high-performance benchmarking suite for evaluating 3D point cloud reconstruction quality. Computes **Chamfer Distance**, **Hausdorff Distance**, **F-Score**, **Normal Consistency**, and **Earth Mover's Distance (EMD)** metrics with sub-5 ms KD-tree spatial indexing, and provides batch evaluation (parallel), visualization, and performance profiling tools.

---

## 📁 Project Structure

```
PointCloud-BenchMarking/
├── src/
│   ├── metrics/          # Chamfer, Hausdorff, F-Score, Normal Consistency, EMD
│   ├── indexing/         # KD-tree spatial index (scipy / Open3D)
│   ├── processing/       # Downsampling, outlier removal, surface reconstruction, I/O
│   ├── evaluation/       # Single-pair & parallel batch evaluators, report generator
│   ├── visualization/    # Overlay, error heatmaps, metric plots
│   └── logging_config.py # Console + file logging
├── benchmarks/           # Synthetic performance profiling
├── configs/              # YAML evaluation configs
├── data/                 # Point cloud data (pred + GT directories)
├── tests/                # Unit tests (32 tests)
├── notebooks/            # Jupyter demo notebook
├── evaluate.py           # CLI entry point
├── pyproject.toml        # pip-installable package config
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PointCloud-BenchMarking.git
cd PointCloud-BenchMarking

# Install dependencies
pip install -r requirements.txt

# Or install as editable package
pip install -e .
```

### Single Evaluation

```bash
python evaluate.py evaluate --pred data/predicted/sample.ply --gt data/ground_truth/sample.ply
```

### Batch Evaluation

```bash
# Sequential
python evaluate.py batch \
    --pred-dir data/predicted/ \
    --gt-dir data/ground_truth/ \
    --output results/batch_results.csv \
    --format csv

# Parallel (auto-detect workers)
python evaluate.py batch \
    --pred-dir data/predicted/ \
    --gt-dir data/ground_truth/ \
    --parallel \
    --output results/batch_results.csv \
    --format csv
```

### Visualization

```bash
# Interactive overlay + heatmap
python evaluate.py visualize --pred sample_pred.ply --gt sample_gt.ply

# Save to files (non-interactive)
python evaluate.py visualize --pred pred.ply --gt gt.ply --save-dir output/vis/
```

### Run Benchmarks

```bash
python evaluate.py benchmark --output benchmarks/results.json --plot
```

---

## 📐 Metric Definitions

### 1. Chamfer Distance (CD)

Measures the **average** nearest-neighbour distance between two point clouds in both directions.

$$
\text{CD}(P, Q) = \frac{1}{|P|} \sum_{p \in P} \min_{q \in Q} \|p - q\|_2 \;+\; \frac{1}{|Q|} \sum_{q \in Q} \min_{p \in P} \|q - p\|_2
$$

**Intuition**: CD penalises missing regions equally from both sides. A low CD means the two clouds overlap closely on average. It is the most widely used metric in point cloud reconstruction benchmarks.

| Property | Value |
|---|---|
| Range | [0, ∞) |
| Ideal | 0.0 |
| Sensitive to | Overall surface coverage |

### 2. Hausdorff Distance (HD)

Measures the **worst-case** nearest-neighbour deviation between two point clouds.

$$
\text{HD}(P, Q) = \max\!\Big(\, \max_{p \in P} \min_{q \in Q} \|p - q\|, \;\; \max_{q \in Q} \min_{p \in P} \|q - p\| \,\Big)
$$

**Intuition**: HD captures the single largest reconstruction error. It is useful for detecting outlier regions that Chamfer Distance may average out. A robust variant uses a percentile (e.g., 95th) instead of the max.

| Property | Value |
|---|---|
| Range | [0, ∞) |
| Ideal | 0.0 |
| Sensitive to | Worst-case outliers |

### 3. F-Score at Threshold τ

Precision, recall, and their harmonic mean at a distance threshold τ.

$$
\text{Precision}(\tau) = \frac{|\{p \in P : \min_q \|p - q\| < \tau\}|}{|P|}, \quad
\text{Recall}(\tau) = \frac{|\{q \in Q : \min_p \|q - p\| < \tau\}|}{|Q|}
$$

$$
F_1(\tau) = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**Intuition**: F-Score tells you what fraction of points are "correctly reconstructed" within tolerance τ. Smaller τ = stricter. Multiple thresholds give a profile of reconstruction quality at different tolerances.

| Property | Value |
|---|---|
| Range | [0, 1] |
| Ideal | 1.0 |
| Sensitive to | Fraction of well-reconstructed surface |

### 4. Normal Consistency (NC)

Measures the alignment of surface normals at nearest-neighbour correspondences.

$$
\text{NC}(P, Q) = \frac{1}{|P|} \sum_{p \in P} |\mathbf{n}_p \cdot \mathbf{n}_{\text{nn}(p, Q)}|
$$

**Intuition**: NC measures how well the predicted surface orientation matches the ground truth. A value of 1.0 means perfect normal alignment. Requires normals on both clouds (auto-estimated via Open3D if missing).

| Property | Value |
|---|---|
| Range | [0, 1] |
| Ideal | 1.0 |
| Sensitive to | Surface normal orientation accuracy |

### 5. Earth Mover's Distance (EMD)

Optimal transport cost between two equal-sized point sets.

$$
\text{EMD}(P, Q) = \min_\phi \frac{1}{|P|} \sum_{p \in P} \|p - \phi(p)\|^2
$$

where φ is a bijection from P to Q.

**Intuition**: EMD finds the optimal one-to-one matching and measures the average squared distance. It is stricter than Chamfer Distance because every point must be matched. O(n³) complexity — auto-subsamples large clouds.

| Property | Value |
|---|---|
| Range | [0, ∞) |
| Ideal | 0.0 |
| Sensitive to | Global point distribution |

---

## ⚡ Performance Benchmarks

Benchmarks measured on synthetic sphere point clouds (average of 5 trials per size):

| Points | Chamfer | Hausdorff | F-Score | KD-Tree Build | KD-Tree Query |
|-------:|--------:|----------:|--------:|-------------:|--------------:|
| 1,000 | ~1 ms | ~1 ms | ~1 ms | <1 ms | <1 ms |
| 10,000 | ~5 ms | ~5 ms | ~5 ms | ~2 ms | ~3 ms |
| 50,000 | ~25 ms | ~25 ms | ~25 ms | ~10 ms | ~15 ms |
| 100,000 | ~55 ms | ~55 ms | ~55 ms | ~20 ms | ~35 ms |
| 500,000 | ~350 ms | ~350 ms | ~350 ms | ~120 ms | ~230 ms |

> **Note**: Exact timings vary by hardware. Run `python evaluate.py benchmark` to measure on your machine.

### KD-Tree Performance

- **Build**: O(n log n), sub-5 ms for 10K points using `scipy.spatial.cKDTree`
- **Query**: O(log n) per point, vectorised batch queries for maximum throughput
- **Backend**: Defaults to scipy (fastest), with Open3D fallback for integration with rendering

---

## 🔄 Benchmark Workflow

```
┌──────────────────┐     ┌──────────────────┐
│  Predicted Cloud  │     │  Ground Truth     │
│  (reconstruction) │     │  (LiDAR / mesh)   │
└────────┬─────────┘     └────────┬──────────┘
         │                        │
         ▼                        ▼
┌────────────────────────────────────────────┐
│         Preprocessing Pipeline              │
│  • Voxel downsampling (configurable)       │
│  • Statistical outlier removal             │
└────────────────────┬───────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────┐
│         Metric Computation                  │
│  • Chamfer Distance                        │
│  • Hausdorff Distance                      │
│  • F-Score @ multiple thresholds           │
│  • Normal Consistency (optional)           │
│  • Earth Mover's Distance (optional)       │
└────────────────────┬───────────────────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
     ┌─────────┐ ┌────────┐ ┌──────────┐
     │ Report  │ │ Plots  │ │ Heatmaps │
     │ (JSON/  │ │ (PNG)  │ │ (3D vis) │
     │  CSV)   │ │        │ │          │
     └─────────┘ └────────┘ └──────────┘
```

---

## 🧪 Example Comparison

```bash
# Generate synthetic data and evaluate
python -c "
from src.processing.io_utils import generate_synthetic_sphere, save_point_cloud
from src.evaluation.evaluator import PointCloudEvaluator
from src.evaluation.report import generate_report

# Create test data
gt = generate_synthetic_sphere(10000, radius=1.0, seed=42)
pred = generate_synthetic_sphere(10000, radius=1.0, noise_std=0.02, seed=43)

save_point_cloud(gt, 'data/ground_truth/sphere.npy')
save_point_cloud(pred, 'data/predicted/sphere.npy')

# Evaluate
evaluator = PointCloudEvaluator()
results = evaluator.evaluate(pred, gt, preprocess=False)
print(generate_report(results))
"
```

**Sample Output:**
```
============================================================
  Point Cloud Evaluation Report
============================================================

  Points (pred): 10000
  Points (gt)  : 10000

  ── Chamfer Distance ──
    Mean (bidirectional) : 0.038921
    Forward  (pred→gt)   : 0.019412
    Backward (gt→pred)   : 0.019509
    Max                  : 0.146532

  ── Hausdorff Distance ──
    Symmetric : 0.146532
    Forward   : 0.143219
    Backward  : 0.146532

  ── F-Score ──
      Threshold   Precision      Recall          F1
      ──────────  ──────────  ──────────  ──────────
        0.0010      0.0012      0.0010      0.0011
        0.0050      0.0298      0.0284      0.0291
        0.0100      0.1182      0.1156      0.1169
        0.0200      0.4310      0.4278      0.4294
        0.0500      0.9648      0.9634      0.9641
============================================================
```

---

## ⚙️ Configuration

All parameters are controlled via `configs/default.yaml`:

```yaml
preprocessing:
  voxel_downsample:
    voxel_size: 0.02        # metres
  outlier_removal:
    nb_neighbors: 20
    std_ratio: 2.0

metrics:
  chamfer:
    squared: false
  hausdorff:
    percentile: 100         # 100 = exact, <100 = robust
  fscore:
    thresholds: [0.001, 0.005, 0.01, 0.02, 0.05]
  normal_consistency:
    enabled: false          # requires Open3D for normal estimation
  emd:
    enabled: false          # O(n³) — auto-subsamples
    max_points: 2048

logging:
  log_to_file: true
  log_dir: "logs"
```

Override via CLI:

```bash
python evaluate.py -c configs/custom.yaml evaluate --pred pred.ply --gt gt.ply
```

---

## 📈 Scaling Strategy

| Scale | Points | Strategy |
|-------|-------:|----------|
| **Small** | <50K | Direct computation, all metrics in <100 ms |
| **Medium** | 50K–500K | Voxel downsampling to reduce density, batch KD-tree queries |
| **Large** | 500K–5M | Aggressive downsampling + chunked evaluation, parallel metric computation |
| **Massive** | >5M | Spatial partitioning (octree), distributed evaluation, approximate NN |

**Key optimizations built in:**
- `scipy.spatial.cKDTree` with configurable leaf size for optimal cache behaviour
- Vectorised NumPy operations — no Python loops in hot paths
- Configurable preprocessing to reduce point count before evaluation
- Memory-efficient: per-point distances computed on-the-fly

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src
```

---

## 📄 API Reference

### Metrics

```python
from src.metrics import chamfer_distance, hausdorff_distance, f_score
from src.metrics import normal_consistency, earth_movers_distance

cd = chamfer_distance(pred, gt)              # → dict with mean, max, per-point
hd = hausdorff_distance(pred, gt)             # → dict with symmetric, directed
fs = f_score(pred, gt, thresholds=[0.01])     # → dict with precision, recall, F1
nc = normal_consistency(pred, gt)             # → dict with mean, std, per-point
emd = earth_movers_distance(pred, gt)         # → dict with emd, emd_sqrt
```

### Processing

```python
from src.processing import voxel_downsample, statistical_outlier_removal

ds = voxel_downsample(points, voxel_size=0.02)
clean = statistical_outlier_removal(points, nb_neighbors=20, std_ratio=2.0)
```

### Indexing

```python
from src.indexing import KDTreeIndex

tree = KDTreeIndex(points, backend="scipy")
dists, idxs = tree.query_knn(query_pts, k=1)
neighbours = tree.query_radius(query_pts, radius=0.05)
```

### Evaluation

```python
from src.evaluation import PointCloudEvaluator, BatchEvaluator

evaluator = PointCloudEvaluator(config)
results = evaluator.evaluate(pred, gt)

batch = BatchEvaluator(config)
df = batch.evaluate_batch("data/predicted/", "data/ground_truth/")
# or parallel:
df = batch.evaluate_batch_parallel("data/predicted/", "data/ground_truth/", n_workers=4)
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
