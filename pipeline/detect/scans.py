"""Single-scan preprocessing for detection: device mount / operator masks (legacy masks), sensor range <= det_max_range,
per-point time, odometry-frame coordinates x_odom = R(t_i) s_i + p(t_i) with the legacy IMU + ICP trajectory.

The odometry itself is not recomputed; these filtered points are the detection input only (the odometry's own ICP input was
the unfiltered scan minus the 0.7 m / 40 m limits, see proposed_pipeline/lidar_odometry.py).
"""
from __future__ import annotations

import collections
import os

import numpy as np

from pipeline.common.lidar_masks import operator_hit, voxel_key
from pipeline.common.recording import SCAN_PERIOD_US


def load_masks(masks_dir: str) -> dict:
    """self-mask voxel keys and operator sector mask of the masks stage (0_prep/masks, read only)."""
    sm = np.load(os.path.join(masks_dir, "self_mask.npz"))
    om = np.load(os.path.join(masks_dir, "operator_mask.npz"))
    op_args = dict(operator_z=tuple(float(v) for v in om["z"]), operator_core=float(om["core"]), operator_range=float(om["range"]),
                   operator_az_step=float(om["az_step"]), operator_z_step=float(om["z_step"]))
    return dict(self_keys=sm["keys"], self_voxel=float(sm["voxel"]), self_radius=float(sm["radius"]), op_mask=om["mask"], op_args=op_args)


def anchor_scan(run, t_c_us: int, max_dt_ms: float):
    """Scan whose centre time (start + 25 ms) is nearest to t_c; None if farther than max_dt_ms (LiDAR gap / outside)."""
    centres = run.lidar_ts + SCAN_PERIOD_US // 2
    k = int(np.clip(np.searchsorted(centres, t_c_us), 0, len(centres) - 1))
    cand = [j for j in (k - 1, k) if 0 <= j < len(centres)]
    a = min(cand, key=lambda j: abs(int(centres[j]) - t_c_us))
    return a if abs(int(centres[a]) - t_c_us) <= max_dt_ms * 1e3 else None


class ScanStore:
    """Preprocessed single scans with an LRU cache."""

    def __init__(self, run, pose, masks: dict | None, cfg: dict, cache: int = 200):
        self.run, self.pose, self.masks, self.cfg = run, pose, masks, cfg["scan_prep"]
        self.cache = collections.OrderedDict()
        self.n_cache = cache

    def get(self, i: int) -> dict:
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        s = self.run.load_scan(i)
        c = self.cfg
        valid = s["range"] > 0.05
        keep = s["range"] >= c["min_range"]
        n_near = int((valid & ~keep).sum())
        n_self = n_oper = 0
        if self.masks is not None and c.get("self_mask", True):
            near = keep & (s["range"] < self.masks["self_radius"])
            hit = np.zeros(len(keep), bool)
            hit[near] = np.isin(voxel_key(s["xyz"][near], self.masks["self_voxel"]), self.masks["self_keys"])
            n_self = int(hit.sum())
            keep &= ~hit
        if self.masks is not None and c.get("operator_mask", True):
            hit = keep & operator_hit(s["xyz"], s["range"], self.masks["op_args"], self.masks["op_mask"])
            n_oper = int(hit.sum())
            keep &= ~hit
        far = keep & (s["range"] > c["det_max_range"])
        keep &= ~far
        tp = s["t_point_us"][keep]
        xyz = s["xyz"][keep]
        R = self.pose.gyro.at(tp)
        x_odom = np.einsum("nij,nj->ni", R, xyz) + self.pose.odo.at(tp)
        out = dict(scan_id=int(i), t_scan_us=int(s["t_scan_us"]), point_id=np.int64(i) * (1 << 20) + s["index"][keep],
                   t_point_us=tp, ring=s["ring"][keep], col=s["col"][keep], range=s["range"][keep], xyz_dev_raw=xyz, xyz_odom=x_odom,
                   n_raw=int(valid.sum()), n_near=n_near, n_self=n_self, n_oper=n_oper, n_far=int(far.sum()), n_range=int(keep.sum()))
        self.cache[i] = out
        if len(self.cache) > self.n_cache:
            self.cache.popitem(last=False)
        return out
