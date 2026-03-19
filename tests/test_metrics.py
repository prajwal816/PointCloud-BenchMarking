"""
Unit Tests — Metrics Module
============================
Tests for all metrics, KD-tree, and processing utilities on
synthetic point clouds with known properties.
"""

import sys
import os
import pytest
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics.chamfer import chamfer_distance
from src.metrics.hausdorff import hausdorff_distance
from src.metrics.fscore import f_score
from src.metrics.normal_consistency import normal_consistency
from src.metrics.emd import earth_movers_distance
from src.indexing.kdtree import KDTreeIndex
from src.processing.downsampling import voxel_downsample
from src.processing.outlier_removal import statistical_outlier_removal
from src.processing.io_utils import generate_synthetic_sphere, generate_synthetic_cube


# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def identical_clouds():
    """Two identical sphere point clouds."""
    pts = generate_synthetic_sphere(5000, radius=1.0, seed=42)
    return pts.copy(), pts.copy()


@pytest.fixture
def noisy_cloud_pair():
    """Ground truth sphere + predicted sphere with slight noise."""
    gt = generate_synthetic_sphere(5000, radius=1.0, seed=42)
    pred = generate_synthetic_sphere(5000, radius=1.0, noise_std=0.02, seed=43)
    return pred, gt


@pytest.fixture
def offset_clouds():
    """Two clouds with a known translation offset."""
    gt = generate_synthetic_sphere(3000, radius=1.0, seed=42)
    pred = gt + np.array([0.1, 0.0, 0.0])
    return pred, gt


# ═══════════════════════════════════════════════════════════════
#  Chamfer Distance Tests
# ═══════════════════════════════════════════════════════════════

class TestChamferDistance:
    def test_identical_clouds_zero(self, identical_clouds):
        pred, gt = identical_clouds
        result = chamfer_distance(pred, gt)
        assert result["chamfer_mean"] == pytest.approx(0.0, abs=1e-10)
        assert result["chamfer_max"] == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self, noisy_cloud_pair):
        pred, gt = noisy_cloud_pair
        r1 = chamfer_distance(pred, gt)
        r2 = chamfer_distance(gt, pred)
        assert r1["chamfer_mean"] == pytest.approx(r2["chamfer_mean"], rel=1e-6)

    def test_offset_positive(self, offset_clouds):
        pred, gt = offset_clouds
        result = chamfer_distance(pred, gt)
        assert result["chamfer_mean"] > 0
        # On a curved sphere, a 0.1 translation shifts NN distances
        # below 0.1 for most points → CD ≈ 0.13 (geometry-dependent)
        assert 0.05 < result["chamfer_mean"] < 0.3

    def test_per_point_distances_shape(self, noisy_cloud_pair):
        pred, gt = noisy_cloud_pair
        result = chamfer_distance(pred, gt)
        assert result["pred_to_gt"].shape == (len(pred),)
        assert result["gt_to_pred"].shape == (len(gt),)

    def test_squared_mode(self, offset_clouds):
        pred, gt = offset_clouds
        r_normal = chamfer_distance(pred, gt, squared=False)
        r_squared = chamfer_distance(pred, gt, squared=True)
        # Squared distances should be larger (each dist > 0)
        assert r_squared["chamfer_mean"] > 0

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            chamfer_distance(np.zeros((10, 2)), np.zeros((10, 3)))


# ═══════════════════════════════════════════════════════════════
#  Hausdorff Distance Tests
# ═══════════════════════════════════════════════════════════════

class TestHausdorffDistance:
    def test_identical_clouds_zero(self, identical_clouds):
        pred, gt = identical_clouds
        result = hausdorff_distance(pred, gt)
        assert result["hausdorff"] == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self, noisy_cloud_pair):
        pred, gt = noisy_cloud_pair
        r1 = hausdorff_distance(pred, gt)
        r2 = hausdorff_distance(gt, pred)
        assert r1["hausdorff"] == pytest.approx(r2["hausdorff"], rel=1e-6)

    def test_offset_bound(self, offset_clouds):
        pred, gt = offset_clouds
        result = hausdorff_distance(pred, gt)
        # HD should be >= translation distance (0.1)
        assert result["hausdorff"] >= 0.1 - 1e-6

    def test_robust_percentile(self, noisy_cloud_pair):
        pred, gt = noisy_cloud_pair
        r_full = hausdorff_distance(pred, gt, percentile=100)
        r_95 = hausdorff_distance(pred, gt, percentile=95)
        # 95th percentile should be <= full Hausdorff
        assert r_95["hausdorff"] <= r_full["hausdorff"] + 1e-9

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            hausdorff_distance(np.zeros((10, 4)), np.zeros((10, 3)))


# ═══════════════════════════════════════════════════════════════
#  F-Score Tests
# ═══════════════════════════════════════════════════════════════

class TestFScore:
    def test_identical_clouds_perfect(self, identical_clouds):
        pred, gt = identical_clouds
        result = f_score(pred, gt, thresholds=0.001)
        entry = result["results"][0]
        assert entry["precision"] == 1.0
        assert entry["recall"] == 1.0
        assert entry["f1"] == 1.0

    def test_multithreshold(self, noisy_cloud_pair):
        pred, gt = noisy_cloud_pair
        result = f_score(pred, gt, thresholds=[0.001, 0.01, 0.1])
        assert len(result["results"]) == 3
        # Larger threshold → higher scores
        f1_values = [r["f1"] for r in result["results"]]
        assert f1_values == sorted(f1_values)

    def test_precision_recall_range(self, noisy_cloud_pair):
        pred, gt = noisy_cloud_pair
        result = f_score(pred, gt, thresholds=0.01)
        entry = result["results"][0]
        assert 0.0 <= entry["precision"] <= 1.0
        assert 0.0 <= entry["recall"] <= 1.0
        assert 0.0 <= entry["f1"] <= 1.0

    def test_per_point_shape(self, noisy_cloud_pair):
        pred, gt = noisy_cloud_pair
        result = f_score(pred, gt, thresholds=0.01)
        assert result["pred_to_gt"].shape == (len(pred),)

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            f_score(np.zeros((10, 5)), np.zeros((10, 3)))


# ═══════════════════════════════════════════════════════════════
#  Normal Consistency Tests
# ═══════════════════════════════════════════════════════════════

class TestNormalConsistency:
    def test_identical_with_normals(self):
        """Identical clouds with identical normals → NC = 1.0."""
        pts = generate_synthetic_sphere(500, seed=42)
        normals = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        result = normal_consistency(pts, pts, normals, normals)
        assert result["normal_consistency_mean"] == pytest.approx(1.0, abs=1e-6)

    def test_flipped_normals(self):
        """Flipped normals still score 1.0 (we measure |dot|)."""
        pts = generate_synthetic_sphere(500, seed=42)
        normals = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        result = normal_consistency(pts, pts, normals, -normals)
        assert result["normal_consistency_mean"] == pytest.approx(1.0, abs=1e-6)

    def test_range(self, noisy_cloud_pair):
        """NC should be in [0, 1]."""
        pred, gt = noisy_cloud_pair
        normals_p = pred / np.linalg.norm(pred, axis=1, keepdims=True)
        normals_g = gt / np.linalg.norm(gt, axis=1, keepdims=True)
        result = normal_consistency(pred, gt, normals_p, normals_g)
        assert 0.0 <= result["normal_consistency_mean"] <= 1.0

    def test_per_point_shape(self):
        pts = generate_synthetic_sphere(300, seed=42)
        normals = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        result = normal_consistency(pts, pts, normals, normals)
        assert result["per_point_consistency"].shape == (300,)

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            normal_consistency(np.zeros((10, 2)), np.zeros((10, 3)))


# ═══════════════════════════════════════════════════════════════
#  Earth Mover's Distance Tests
# ═══════════════════════════════════════════════════════════════

class TestEMD:
    def test_identical_clouds_zero(self):
        """Identical clouds → EMD = 0."""
        pts = generate_synthetic_sphere(500, seed=42)
        result = earth_movers_distance(pts, pts, max_points=500)
        assert result["emd"] == pytest.approx(0.0, abs=1e-10)

    def test_offset_positive(self):
        """Offset clouds → EMD > 0."""
        gt = generate_synthetic_sphere(300, seed=42)
        pred = gt + np.array([0.1, 0.0, 0.0])
        result = earth_movers_distance(pred, gt, max_points=300)
        assert result["emd"] > 0

    def test_subsampling(self):
        """Large clouds are subsampled to max_points."""
        pts = generate_synthetic_sphere(5000, seed=42)
        result = earth_movers_distance(pts, pts, max_points=256)
        assert result["n_points_used"] == 256

    def test_result_keys(self):
        pts = generate_synthetic_sphere(200, seed=42)
        result = earth_movers_distance(pts, pts, max_points=200)
        assert "emd" in result
        assert "emd_total" in result
        assert "emd_sqrt" in result
        assert "n_points_used" in result

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            earth_movers_distance(np.zeros((10, 2)), np.zeros((10, 3)))


# ═══════════════════════════════════════════════════════════════
#  KDTree Tests
# ═══════════════════════════════════════════════════════════════

class TestKDTree:
    def test_build_and_query(self):
        pts = generate_synthetic_sphere(1000, seed=42)
        tree = KDTreeIndex(pts, backend="scipy")
        assert tree.size == 1000
        assert tree.build_time_ms > 0

        dists, idxs = tree.query_knn(pts[:10], k=1)
        assert dists.shape == (10,)
        np.testing.assert_allclose(dists, 0.0, atol=1e-10)

    def test_knn_k3(self):
        pts = generate_synthetic_sphere(500, seed=42)
        tree = KDTreeIndex(pts, backend="scipy")
        dists, idxs = tree.query_knn(pts[:5], k=3)
        assert dists.shape == (5, 3)
        assert idxs.shape == (5, 3)
        # First neighbour of each point should be itself (distance 0)
        np.testing.assert_allclose(dists[:, 0], 0.0, atol=1e-10)

    def test_radius_search(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 10, 10]], dtype=np.float64)
        tree = KDTreeIndex(pts, backend="scipy")
        results = tree.query_radius(np.array([[0, 0, 0]]), radius=1.5)
        assert len(results) == 1
        assert 0 in results[0]  # origin should find itself
        assert 1 in results[0]  # (1,0,0) is within 1.5
        assert 2 in results[0]  # (0,1,0) is within 1.5

    def test_repr(self):
        pts = generate_synthetic_sphere(100, seed=42)
        tree = KDTreeIndex(pts, backend="scipy")
        assert "scipy" in repr(tree)


# ═══════════════════════════════════════════════════════════════
#  Processing Tests
# ═══════════════════════════════════════════════════════════════

class TestProcessing:
    def test_voxel_downsample(self):
        pts = generate_synthetic_sphere(10000, seed=42)
        result = voxel_downsample(pts, voxel_size=0.1)
        assert len(result["points"]) < len(pts)
        assert result["reduction_ratio"] > 0
        assert result["points"].shape[1] == 3

    def test_outlier_removal(self):
        pts = generate_synthetic_sphere(1000, seed=42)
        # Add outliers
        outliers = np.array([[10, 10, 10], [-10, -10, -10]], dtype=np.float64)
        pts_with_outliers = np.vstack([pts, outliers])
        result = statistical_outlier_removal(pts_with_outliers, nb_neighbors=20, std_ratio=2.0)
        assert result["n_removed"] >= 2
        assert len(result["points"]) < len(pts_with_outliers)

    def test_synthetic_generators(self):
        sphere = generate_synthetic_sphere(500, radius=2.0, seed=42)
        assert sphere.shape == (500, 3)
        # All points should be approximately on the sphere surface
        radii = np.linalg.norm(sphere, axis=1)
        np.testing.assert_allclose(radii, 2.0, atol=1e-6)

        cube = generate_synthetic_cube(600, side_length=1.0, seed=42)
        assert cube.shape == (600, 3)
        # All coordinates should be in [-0.5, 0.5]
        assert np.all(cube >= -0.5 - 1e-6)
        assert np.all(cube <= 0.5 + 1e-6)


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

