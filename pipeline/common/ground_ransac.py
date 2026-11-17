"""Floor plane RANSAC and leveling (verbatim from OpenPCDet/tools/glim_data_file_lidar_10m_ground_remove.py:
estimate_plane_ransac, rotation_from_vectors, compute_ground_z_after_level)."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def rotation_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return R such that:
        R @ a ~= b
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-8:
        if c > 0:
            return np.eye(3, dtype=np.float32)

        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])

        v = np.cross(a, axis)
        v = v / (np.linalg.norm(v) + 1e-12)

        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ], dtype=np.float64)

        R = np.eye(3) + 2.0 * (vx @ vx)
        return R.astype(np.float32)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ], dtype=np.float64)

    R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s ** 2))

    return R.astype(np.float32)


def estimate_plane_ransac(
    points: np.ndarray,
    num_iter: int = 300,
    dist_thresh: float = 0.08,
    z_percentile: float = 40.0,
    seed: int = 42,
) -> Tuple[np.ndarray, float, int]:
    """
    Estimate dominant floor-like plane from points.

    Plane:
        n dot x + d = 0

    Returns:
        normal, d, inlier_count
    """

    xyz = points[:, :3]

    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]

    if xyz.shape[0] < 3:
        return np.array([0, 0, 1], dtype=np.float32), 0.0, 0

    z_limit = np.percentile(xyz[:, 2], z_percentile)
    cand = xyz[xyz[:, 2] <= z_limit]

    if cand.shape[0] < 100:
        cand = xyz

    rng = np.random.default_rng(seed)

    best_inliers = 0
    best_normal = np.array([0, 0, 1], dtype=np.float32)
    best_d = 0.0

    for _ in range(num_iter):
        ids = rng.choice(cand.shape[0], size=3, replace=False)
        p1, p2, p3 = cand[ids]

        n = np.cross(p2 - p1, p3 - p1)
        norm = float(np.linalg.norm(n))

        if norm < 1e-6:
            continue

        n = n / norm

        if n[2] < 0:
            n = -n

        # 너무 수직에 가까운 평면은 벽일 가능성이 높음
        if n[2] < 0.35:
            continue

        d = -float(np.dot(n, p1))
        dist = np.abs(cand @ n + d)
        inliers = int(np.sum(dist < dist_thresh))

        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = n.astype(np.float32)
            best_d = d

    return best_normal, best_d, best_inliers


def compute_ground_z_after_level(
    normal: np.ndarray,
    plane_d: float,
    R_level: np.ndarray,
) -> float:
    """
    Original plane:
        normal dot p + d = 0

    Point on plane:
        p0 = -d * normal

    After leveling:
        p0_level = R_level @ p0

    ground_z = p0_level[2]
    """

    n = np.asarray(normal, dtype=np.float32)
    n = n / (np.linalg.norm(n) + 1e-12)

    p0 = -float(plane_d) * n
    p0_level = R_level @ p0

    return float(p0_level[2])
