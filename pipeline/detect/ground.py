"""Per-frame ground plane and ceiling height from the candidate scans (device frame at the image time), same functions and
logic as proposed_pipeline/build_windows.py (RANSAC 300 / 0.08 m / lowest 40 %, gravity check, automatic ceiling layer)."""
from __future__ import annotations

import numpy as np

from pipeline.common.ground_ransac import compute_ground_z_after_level, estimate_plane_ransac, rotation_from_vectors


def fit_frame_plane(xyz_dev: np.ndarray, acc_up: np.ndarray, cfg: dict, seed: int = 0) -> dict:
    """xyz_dev: candidate-scan points in the device frame at t_c (N,3). acc_up: mean accelerometer (device frame) over the
    candidate scans. -> dict(n, d, R_level, ground_z, ceiling_h, source, inliers, n_used): plane n.x + d = 0."""
    g = cfg["ground"]
    near = np.hypot(xyz_dev[:, 0], xyz_dev[:, 1]) <= g["plane_radius"]
    cand = np.nonzero(near)[0]
    if len(cand) > g["plane_points"]:
        cand = np.random.default_rng(seed).choice(cand, g["plane_points"], replace=False)
    P = xyz_dev[cand]
    n_pl, d_pl, inl = estimate_plane_ransac(P, num_iter=g["ransac_iter"], dist_thresh=g["ransac_dist"], z_percentile=g["ransac_zpct"], seed=42)
    n_pl = np.asarray(n_pl, float)
    src = "ransac"
    up = np.asarray(acc_up, float)
    up = up / max(np.linalg.norm(up), 1e-9)
    n_chk = n_pl if n_pl[2] >= 0 else -n_pl
    if g["gravity_max_deg"] > 0 and np.degrees(np.arccos(np.clip(float(n_chk @ up), -1.0, 1.0))) > g["gravity_max_deg"]:
        n_pl = up
        s_up = P @ up
        low = s_up[s_up <= np.percentile(s_up, g["ransac_zpct"])]
        h, edges = np.histogram(low, bins=np.arange(low.min(), low.max() + 0.05, 0.05))
        s0 = float(edges[int(np.argmax(h))] + 0.025)
        d_pl = -s0
        inl = int(np.sum(np.abs(s_up - s0) < g["ransac_dist"]))
        src = "gravity"
    R_lv = rotation_from_vectors(n_pl, np.array([0.0, 0.0, 1.0])).astype(np.float64)
    gz = float(compute_ground_z_after_level(n_pl, d_pl, R_lv))
    pl = dict(n=n_pl, d=float(d_pl), R_level=R_lv, ground_z=gz, source=src, inliers=int(inl), n_used=int(len(cand)))
    zc = (xyz_dev @ R_lv.T)[:, 2] - gz
    ceiling_h = g["ceiling_default"]
    if g["ceiling_auto"] > 0:
        sel = zc[(zc > 1.9) & (zc < 4.0)]
        hh, edges = np.histogram(sel, bins=np.arange(1.9, 4.0 + 1e-6, 0.05))
        if len(hh) and hh.max() >= 0.02 * len(zc):
            ceiling_h = max(float(edges[int(np.argmax(hh))] + 0.025) - g["ceiling_auto"], 2.0)
    pl["ceiling_h"] = float(ceiling_h)
    return pl


def to_level(xyz_dev: np.ndarray, plane: dict) -> np.ndarray:
    """Device frame -> leveled frame (floor z = 0)."""
    out = np.asarray(xyz_dev, float) @ plane["R_level"].T
    out[:, 2] -= plane["ground_z"]
    return out


def height_above_ground(xyz_dev: np.ndarray, plane: dict) -> np.ndarray:
    return to_level(xyz_dev, plane)[:, 2]


def plane_to_odom(plane: dict, R_img: np.ndarray, p_img: np.ndarray) -> dict:
    """Express the device-frame plane n.x + d = 0 in the odometry frame (x_dev = R^T (x_odom - p))."""
    n_o = R_img @ plane["n"]
    d_o = plane["d"] - float(n_o @ p_img)
    return dict(n=n_o, d=d_o)


def height_odom(xyz_odom: np.ndarray, plane_odom: dict) -> np.ndarray:
    """Signed distance above the plane (n unit, pointing up) = height above ground for odometry-frame points."""
    return np.asarray(xyz_odom, float) @ plane_odom["n"] + plane_odom["d"]
