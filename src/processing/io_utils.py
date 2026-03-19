"""
I/O Utilities
=============
Load and save point clouds in common formats (.ply, .pcd, .xyz, .npy).
Includes synthetic point cloud generators for testing.
"""

from __future__ import annotations

import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".ply", ".pcd", ".xyz", ".npy", ".npz"}


def load_point_cloud(filepath: str) -> np.ndarray:
    """Load a point cloud from disk.

    Parameters
    ----------
    filepath : str
        Path to point cloud file (.ply, .pcd, .xyz, .npy, .npz).

    Returns
    -------
    np.ndarray, shape (N, 3)
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext in (".ply", ".pcd"):
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(filepath))
            points = np.asarray(pcd.points, dtype=np.float64)
        except ImportError:
            raise ImportError("open3d is required to read .ply/.pcd files")

    elif ext == ".xyz":
        points = np.loadtxt(str(filepath), dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        points = points[:, :3]

    elif ext == ".npy":
        points = np.load(str(filepath)).astype(np.float64)
        points = points[:, :3]

    elif ext == ".npz":
        data = np.load(str(filepath))
        key = list(data.keys())[0]
        points = data[key].astype(np.float64)[:, :3]

    else:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )

    logger.info("Loaded %d points from %s", len(points), filepath.name)
    return points


def save_point_cloud(
    points: np.ndarray,
    filepath: str,
    normals: Optional[np.ndarray] = None,
) -> None:
    """Save a point cloud to disk.

    Parameters
    ----------
    points : np.ndarray, shape (N, 3)
    filepath : str
    normals : np.ndarray or None, shape (N, 3)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ext = filepath.suffix.lower()

    if ext in (".ply", ".pcd"):
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
            if normals is not None:
                pcd.normals = o3d.utility.Vector3dVector(
                    np.asarray(normals, dtype=np.float64)
                )
            o3d.io.write_point_cloud(str(filepath), pcd)
        except ImportError:
            raise ImportError("open3d is required to write .ply/.pcd files")

    elif ext == ".xyz":
        np.savetxt(str(filepath), points, fmt="%.8f")

    elif ext == ".npy":
        np.save(str(filepath), points)

    elif ext == ".npz":
        np.savez_compressed(str(filepath), points=points)

    else:
        raise ValueError(f"Unsupported format '{ext}'")

    logger.info("Saved %d points to %s", len(points), filepath.name)


# ------------------------------------------------------------------
# Synthetic generators
# ------------------------------------------------------------------

def generate_synthetic_sphere(
    n_points: int = 10000,
    radius: float = 1.0,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate points uniformly sampled on a sphere surface.

    Parameters
    ----------
    n_points : int
    radius : float
    noise_std : float – Gaussian noise added to each coordinate.
    seed : int or None

    Returns
    -------
    np.ndarray, shape (n_points, 3)
    """
    rng = np.random.default_rng(seed)
    # Uniform on sphere via Marsaglia's method
    phi = rng.uniform(0, 2 * np.pi, n_points)
    cos_theta = rng.uniform(-1, 1, n_points)
    sin_theta = np.sqrt(1 - cos_theta ** 2)

    x = radius * sin_theta * np.cos(phi)
    y = radius * sin_theta * np.sin(phi)
    z = radius * cos_theta

    points = np.column_stack([x, y, z])

    if noise_std > 0:
        points += rng.normal(0, noise_std, points.shape)

    return points


def generate_synthetic_cube(
    n_points: int = 10000,
    side_length: float = 1.0,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate points uniformly sampled on cube surfaces.

    Parameters
    ----------
    n_points : int
    side_length : float
    noise_std : float
    seed : int or None

    Returns
    -------
    np.ndarray, shape (n_points, 3)
    """
    rng = np.random.default_rng(seed)
    half = side_length / 2.0

    # Allocate evenly across 6 faces
    per_face = n_points // 6
    remainder = n_points - per_face * 6
    counts = [per_face] * 6
    for i in range(remainder):
        counts[i] += 1

    faces = []
    for face_idx, count in enumerate(counts):
        uv = rng.uniform(-half, half, (count, 2))
        fixed = np.full((count, 1), half if face_idx % 2 == 0 else -half)

        axis = face_idx // 2  # 0=x, 1=y, 2=z
        if axis == 0:
            face = np.hstack([fixed, uv])
        elif axis == 1:
            face = np.hstack([uv[:, :1], fixed, uv[:, 1:]])
        else:
            face = np.hstack([uv, fixed])

        faces.append(face)

    points = np.vstack(faces)

    if noise_std > 0:
        points += rng.normal(0, noise_std, points.shape)

    return points
