"""
KD-Tree Spatial Index
=====================
Unified KD-tree wrapper supporting both ``scipy.spatial.cKDTree`` and
Open3D backends, with batch k-NN and radius search.
"""

from __future__ import annotations

import time
import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class KDTreeIndex:
    """High-performance KD-tree for nearest-neighbor queries on 3-D points.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
        Reference point cloud to index.
    backend : str
        ``"scipy"`` (default) or ``"open3d"``.
    leaf_size : int
        Leaf size for the KD-tree (scipy only).
    """

    def __init__(
        self,
        points: np.ndarray,
        backend: str = "scipy",
        leaf_size: int = 16,
    ) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        self.backend = backend.lower()
        self.leaf_size = leaf_size
        self._tree = None
        self._build_time: float = 0.0

        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self) -> None:
        t0 = time.perf_counter()

        if self.backend == "scipy":
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.points, leafsize=self.leaf_size)

        elif self.backend == "open3d":
            try:
                import open3d as o3d
            except ImportError:
                raise ImportError(
                    "open3d is required for the 'open3d' backend. "
                    "Install with: pip install open3d"
                )
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.points)
            self._tree = o3d.geometry.KDTreeFlann(pcd)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self._build_time = time.perf_counter() - t0
        logger.info(
            "KDTree built (%s) – %d pts in %.3f ms",
            self.backend,
            len(self.points),
            self._build_time * 1000,
        )

    # ------------------------------------------------------------------
    # k-NN Query
    # ------------------------------------------------------------------
    def query_knn(
        self,
        query_points: np.ndarray,
        k: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch k-nearest-neighbor query.

        Parameters
        ----------
        query_points : np.ndarray, shape (Q, 3)
        k : int

        Returns
        -------
        distances : np.ndarray, shape (Q,) if k==1 else (Q, k)
        indices   : np.ndarray, shape (Q,) if k==1 else (Q, k)
        """
        query_points = np.asarray(query_points, dtype=np.float64)
        t0 = time.perf_counter()

        if self.backend == "scipy":
            distances, indices = self._tree.query(query_points, k=k)

        elif self.backend == "open3d":
            dists_list, idx_list = [], []
            for pt in query_points:
                _, idxs, sqd = self._tree.search_knn_vector_3d(pt, k)
                idx_list.append(np.asarray(idxs))
                dists_list.append(np.sqrt(np.asarray(sqd)))
            distances = np.array(dists_list).squeeze()
            indices = np.array(idx_list).squeeze()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "KNN query (%d pts, k=%d) in %.2f ms", len(query_points), k, elapsed_ms
        )
        return distances, indices

    # ------------------------------------------------------------------
    # Radius Search
    # ------------------------------------------------------------------
    def query_radius(
        self,
        query_points: np.ndarray,
        radius: float,
    ) -> list:
        """Batch radius search.

        Parameters
        ----------
        query_points : np.ndarray, shape (Q, 3)
        radius : float

        Returns
        -------
        list of np.ndarray – indices of neighbours within radius for each query.
        """
        query_points = np.asarray(query_points, dtype=np.float64)
        t0 = time.perf_counter()

        if self.backend == "scipy":
            results = self._tree.query_ball_point(query_points, r=radius)

        elif self.backend == "open3d":
            results = []
            for pt in query_points:
                _, idxs, _ = self._tree.search_radius_vector_3d(pt, radius)
                results.append(np.asarray(idxs))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "Radius query (%d pts, r=%.4f) in %.2f ms",
            len(query_points),
            radius,
            elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def build_time_ms(self) -> float:
        """Wall-clock time to build the tree, in milliseconds."""
        return self._build_time * 1000

    @property
    def size(self) -> int:
        """Number of points in the index."""
        return len(self.points)

    def __repr__(self) -> str:
        return (
            f"KDTreeIndex(backend={self.backend!r}, "
            f"n_points={self.size}, "
            f"build_time={self.build_time_ms:.2f}ms)"
        )
