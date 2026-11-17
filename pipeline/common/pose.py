"""Device pose: orientation from the integrated gyro (proposed_pipeline/common.py GyroRotation), position from the LiDAR odometry
(proposed_pipeline/fusion_common.py Odometry, PoseInterp). Verbatim."""
from __future__ import annotations

import csv

import numpy as np
from scipy.spatial.transform import Rotation

from pipeline.common.recording import Run  # noqa: F401  (type hints)


class GyroRotation:
    """Orientation of the device from the integrated gyro: R(t) maps device-frame vectors at time t into the
    gyro reference frame (the device orientation at the first IMU sample). Bias = mean gyro of the initial
    stationary samples (|w| < bias_max_rate within the first bias_window_s)."""

    def __init__(self, imu_ts_us, gyro, bias_window_s: float = 25.0, bias_max_rate: float = 0.03):
        self.ts = np.asarray(imu_ts_us, np.int64)
        g = np.asarray(gyro, float)
        m = (self.ts < self.ts[0] + int(bias_window_s * 1e6)) & (np.linalg.norm(g, axis=1) < bias_max_rate)
        self.bias = g[m].mean(0) if m.sum() > 200 else np.zeros(3)
        self.w = g - self.bias
        dt = np.diff(self.ts) / 1e6
        steps = Rotation.from_rotvec(self.w[:-1] * dt[:, None]).as_matrix()
        R = np.empty((len(self.ts), 3, 3))
        R[0] = np.eye(3)
        for k in range(len(dt)):
            R[k + 1] = R[k] @ steps[k]
        self.R = R

    def at(self, t_us) -> np.ndarray:
        """R for one time or an array of times (-> (3,3) or (N,3,3))."""
        t = np.atleast_1d(np.asarray(t_us, np.int64))
        k = np.clip(np.searchsorted(self.ts, t, side="right") - 1, 0, len(self.ts) - 2)
        dt = np.clip((t - self.ts[k]) / 1e6, 0.0, None)
        out = self.R[k] @ Rotation.from_rotvec(self.w[k] * dt[:, None]).as_matrix()
        return out[0] if np.ndim(t_us) == 0 else out


# ----------------------------------------------------------------------------------------------------------------------
# poses
# ----------------------------------------------------------------------------------------------------------------------
class Odometry:
    """Device position p(t) in the odometry frame: linear interpolation between the scan-start positions
    (copy of build_windows.Odometry, kept here to avoid that module's import side effects)."""

    def __init__(self, path: str):
        rows = list(csv.DictReader(open(path)))
        self.t = np.array([int(r["stamp_us"]) for r in rows], np.int64)
        self.p = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])

    def at(self, t_us):
        t = np.atleast_1d(np.asarray(t_us, np.int64))
        out = np.stack([np.interp(t.astype(float), self.t.astype(float), self.p[:, k]) for k in range(3)], 1)
        return out[0] if np.ndim(t_us) == 0 else out


class PoseInterp:
    """R_odom_dev(t) from the integrated gyro, p(t) from the LiDAR odometry. Reproduces T_odom_dev of the windows."""

    def __init__(self, run: Run, odometry_csv: str):
        self.gyro = GyroRotation(run.imu_ts, run.gyro)
        self.odo = Odometry(odometry_csv)

    def at(self, t_us: int):
        return self.gyro.at(int(t_us)), self.odo.at(int(t_us))

    def T(self, t_us: int):
        R, p = self.at(t_us)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return T

    def to_device(self, xyz_odom, t_us):
        R, p = self.at(t_us)
        return (np.asarray(xyz_odom, float) - p) @ R          # R^T (x - p)

    def to_odom(self, xyz_dev, t_us):
        R, p = self.at(t_us)
        return np.asarray(xyz_dev, float) @ R.T + p
