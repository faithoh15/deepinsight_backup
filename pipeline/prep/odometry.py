"""Device trajectory without GLIM: orientation from the IMU gyro, translation from the LiDAR (proposed_pipeline/lidar_odometry.py,
verbatim; main() became run_odometry(), argparse values come from configs/pipeline.yaml prep.odometry).

For every scan k (time order):
  * each point is rotated with the gyro orientation at ITS OWN time (column time): x = R(t_i) s_i
  * the scan position p_k (at the scan start) is the only unknown: positions inside the scan use the current velocity,
    x_i = R(t_i) s_i + p_k + v (t_i - t_k)
  * p_k is estimated by translation-only point-to-plane ICP against a local map (the previous map_scans scans placed
    with their estimated positions, 0.1 m voxels, normals from open3d); Huber weights, a prior on the constant-velocity
    prediction keeps degenerate directions (corridors) from drifting; correspondence radius 1.0 -> 0.5 -> 0.3 m
  * v = (p_k - p_{k-1}) / dt (smoothed), then the scan is added to the local map.
The odometry frame = the gyro reference frame (device orientation at the first IMU sample) with the origin at the first scan.
Output: odometry.csv (per scan: stamp, position, orientation quaternion, velocity, ICP statistics), args.json, trajectory.png (figure).
"""
from __future__ import annotations

import collections
import csv
import json
import os
import time
from types import SimpleNamespace

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from pipeline.common.pose import GyroRotation
from pipeline.common.recording import voxel_downsample


class LocalMap:
    def __init__(self, n_scans: int, voxel: float, normal_radius: float, normal_nn: int):
        self.scans = collections.deque(maxlen=n_scans)
        self.voxel, self.normal_radius, self.normal_nn = voxel, normal_radius, normal_nn
        self.pts = self.nrm = self.tree = None

    def add(self, pts: np.ndarray):
        self.scans.append(pts[voxel_downsample(pts, self.voxel)])
        allp = np.concatenate(list(self.scans))
        allp = allp[voxel_downsample(allp, self.voxel)]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(allp))
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.normal_radius, max_nn=self.normal_nn))
        self.pts = allp
        self.nrm = np.asarray(pc.normals)
        self.tree = cKDTree(allp)

    def empty(self):
        return self.tree is None


def solve_translation(rot_pts, dt_s, v, p_pred, lmap: LocalMap, a):
    """Translation-only point-to-plane ICP. rot_pts: gyro-rotated source points, dt_s: point time - scan start (s)."""
    p = p_pred.copy()
    stats = dict(n_corr=0, rms=float("nan"), min_eig=float("nan"), iters=0)
    base = rot_pts + v[None, :] * dt_s[:, None]
    prior = np.eye(3) / a.sigma_prior ** 2
    for it, radius in enumerate(a.radii):
        for _ in range(a.iters_per_radius):
            x = base + p
            d, j = lmap.tree.query(x, distance_upper_bound=radius)
            ok = np.isfinite(d)
            if ok.sum() < 30:
                break
            q, n = lmap.pts[j[ok]], lmap.nrm[j[ok]]
            r = np.einsum("ij,ij->i", x[ok] - q, n)
            w = np.where(np.abs(r) <= a.huber, 1.0, a.huber / np.maximum(np.abs(r), 1e-9)) / a.sigma_r ** 2
            H = (n * w[:, None]).T @ n
            g = (n * (w * r)[:, None]).sum(0)
            A = H + prior
            b = -(g + prior @ (p - p_pred))
            dp = np.linalg.solve(A, b)
            p = p + dp
            stats.update(n_corr=int(ok.sum()), rms=float(np.sqrt(np.mean(r ** 2))), min_eig=float(np.linalg.eigvalsh(H)[0]), iters=stats["iters"] + 1)
            if np.linalg.norm(dp) < 1e-4:
                break
    return p, stats


def draw_trajectory(path, P, scale_px_per_m=20, margin=40):
    xy = P[:, :2]
    lo, hi = xy.min(0), xy.max(0)
    W = int((hi[0] - lo[0]) * scale_px_per_m) + 2 * margin
    H = int((hi[1] - lo[1]) * scale_px_per_m) + 2 * margin
    img = np.full((max(H, 200), max(W, 200), 3), 25, np.uint8)
    u = ((xy[:, 0] - lo[0]) * scale_px_per_m + margin).astype(int)
    v = (img.shape[0] - ((xy[:, 1] - lo[1]) * scale_px_per_m + margin)).astype(int)
    for k in range(1, len(u)):
        c = tuple(int(x) for x in cv2.applyColorMap(np.uint8([[255 * k // len(u)]]), cv2.COLORMAP_TURBO)[0, 0])
        cv2.line(img, (u[k - 1], v[k - 1]), (u[k], v[k]), c, 2)
    cv2.circle(img, (u[0], v[0]), 6, (255, 255, 255), -1)
    cv2.putText(img, "start", (u[0] + 8, v[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"odometry top view (gyro rotation + LiDAR translation), {scale_px_per_m} px/m, colour = time", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(path, img)


def run_odometry(run, out_dir: str, params: dict, scans: str | None = None, figure: bool = False) -> dict:
    a = SimpleNamespace(**params, scans=scans)
    out = out_dir
    os.makedirs(out, exist_ok=True)
    gyro = GyroRotation(run.imu_ts, run.gyro)
    scans = range(run.n_scans())
    if a.scans:
        lo, hi = map(int, a.scans.split(":"))
        scans = range(lo, min(hi, run.n_scans()))
    lmap = LocalMap(a.map_scans, a.map_voxel, a.normal_radius, a.normal_nn)
    rows, P = [], []
    p_prev = np.zeros(3)
    v = np.zeros(3)
    t_prev = None
    t0 = time.time()
    for k in scans:
        s = run.load_scan(k)
        keep = (s["range"] >= a.min_range) & (s["range"] <= a.max_range)
        pts, tp = s["xyz"][keep], s["t_point_us"][keep]
        Rp = gyro.at(tp)
        rot = np.einsum("nij,nj->ni", Rp, pts)
        t_k = s["t_scan_us"]
        dt_pts = (tp - t_k) / 1e6
        if lmap.empty():
            p_k = np.zeros(3)
            st = dict(n_corr=0, rms=float("nan"), min_eig=float("nan"), iters=0)
        else:
            dt = (t_k - t_prev) / 1e6
            p_pred = p_prev + v * dt
            src = voxel_downsample(pts, a.source_voxel)
            p_k, st = solve_translation(rot[src], dt_pts[src], v, p_pred, lmap, a)
            v_new = (p_k - p_prev) / max(dt, 1e-3)
            sp = np.linalg.norm(v_new)
            if sp > a.v_max:
                v_new *= a.v_max / sp
            v = 0.5 * v + 0.5 * v_new
        lmap.add(rot + p_k + v[None, :] * dt_pts[:, None])
        q = Rotation.from_matrix(gyro.at(t_k)).as_quat()
        rows.append(dict(scan=k, stamp_us=t_k, x=p_k[0], y=p_k[1], z=p_k[2], qx=q[0], qy=q[1], qz=q[2], qw=q[3], vx=v[0], vy=v[1], vz=v[2],
                         speed=float(np.linalg.norm(v[:2])), n_corr=st["n_corr"], rms=st["rms"], min_eig=st["min_eig"], iters=st["iters"]))
        P.append(p_k)
        p_prev, t_prev = p_k, t_k
        if (k - scans[0]) % 250 == 0:
            print(f"  scan {k} ({k - scans[0] + 1}/{len(scans)}): p=({p_k[0]:.2f},{p_k[1]:.2f},{p_k[2]:.2f}) speed {np.linalg.norm(v[:2]):.2f} m/s "
                  f"corr {st['n_corr']} rms {st['rms']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(out, "odometry.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v_:.6f}" if isinstance(v_, float) else v_) for k, v_ in r.items()})
    json.dump(dict(vars(a), data=run.data_dir, out=out, gyro_bias=gyro.bias.tolist(), n_scans=len(rows), seconds=round(time.time() - t0, 1)),
              open(os.path.join(out, "args.json"), "w"), indent=1, ensure_ascii=False)
    P = np.array(P)
    if figure:
        draw_trajectory(os.path.join(out, "trajectory.png"), P)
    path_len = float(np.sum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)))
    print(f"{len(rows)} scans in {time.time() - t0:.0f}s; path length {path_len:.1f} m; median ICP rms "
          f"{np.nanmedian([r['rms'] for r in rows]):.3f} m; output {out}")
    return dict(n_scans=len(rows), path_length_m=round(path_len, 1), seconds=round(time.time() - t0, 1))
