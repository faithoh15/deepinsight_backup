"""Mount (self) mask, operator sector mask and the device pose at every camera frame time (from proposed_pipeline/build_windows.py,
verbatim: self_mask, operator_mask, _shift_z, the frame selection of main() and the window pose T_odom_dev = [gyro R | odometry p](t_img)).
The 20-scan point windows (about 1 GB per recording) are not written: the later stages only use these masks and poses.

Outputs: self_mask.npz, operator_mask.npz (same keys as before), frame_poses.csv (frame, cam0_stamp_us, t_img_us, T00..T33 written with
repr() so float64 round-trips exactly; the frames = the frames that had a window), args.json.
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np

from pipeline.common.lidar_masks import voxel_key
from pipeline.common.pose import GyroRotation, Odometry


def self_mask(run, a):
    """Voxels (--self-voxel) within --self-radius occupied in >= --self-occ of the sampled scans: the mount / handle and
    anything else rigidly attached to the device. The nodding LiDAR only hits a given part in some scans, so the
    threshold is low (0.1); a person standing near the device never reaches it over the whole run."""
    counts = {}
    scans = range(0, run.n_scans(), a["self_step"])
    for i in scans:
        s = run.load_scan(i)
        m = (s["range"] >= a["min_range"]) & (s["range"] < a["self_radius"])
        for k in np.unique(voxel_key(s["xyz"][m], a["self_voxel"])):
            counts[k] = counts.get(k, 0) + 1
    keys = np.array(list(counts), np.int64)
    frac = np.array(list(counts.values()), float) / len(scans)
    return keys[frac >= a["self_occ"]], keys, frac


def operator_mask(run, a):
    """Sector occupancy (azimuth x z, device frame) of the points within --operator-range over the sampled scans."""
    na = int(round(360 / a["operator_az_step"]))
    zlo, zhi = a["operator_z"]
    nz = int(round((zhi - zlo) / a["operator_z_step"]))
    scans = range(0, run.n_scans(), a["self_step"])
    occ = np.zeros((na, nz), int)
    for i in scans:
        s = run.load_scan(i)
        m = (s["range"] >= a["min_range"]) & (s["range"] < a["operator_range"])
        p = s["xyz"][m]
        az = ((np.degrees(np.arctan2(p[:, 1], p[:, 0])) + 360) % 360 / a["operator_az_step"]).astype(int) % na
        zb = ((p[:, 2] - zlo) / a["operator_z_step"]).astype(int)
        ok = (zb >= 0) & (zb < nz)
        cnt = np.zeros((na, nz), int)
        np.add.at(cnt, (az[ok], zb[ok]), 1)
        occ += cnt >= 3
    frac = occ / len(scans)
    core = frac >= a["operator_occ"]
    mask = core.copy()
    for da in (-1, 0, 1):                                  # dilate by one sector (azimuth wraps, z does not)
        for dz in (-1, 0, 1):
            mask |= _shift_z(np.roll(core, da, 0), dz)
    return mask, frac


def _shift_z(m, dz):
    out = np.zeros_like(m)
    if dz > 0:
        out[:, dz:] = m[:, :-dz]
    elif dz < 0:
        out[:, :dz] = m[:, -dz:]
    else:
        out = m
    return out


def select_frames(run, odo_t, n_scans: int, max_gap_ms: float, frames=None) -> list:
    """camera frames with a full n_scans window before them (build_windows main(): same skip rules) -> [(frame, scan_ids)]."""
    frames = range(len(run.cam_stamp["cam0"])) if frames is None else frames
    tasks = []
    for fr in frames:
        t_img = run.image_time(fr)
        k = int(np.searchsorted(run.lidar_ts, t_img, side="right")) - 1
        if k < n_scans - 1 or t_img - run.lidar_ts[k] > max_gap_ms * 1000:
            continue
        if run.lidar_ts[k - n_scans + 1] < odo_t[0] or t_img > odo_t[-1] + 50000:
            continue
        tasks.append((fr, list(range(k - n_scans + 1, k + 1))))
    return tasks


def frame_poses(run, gyro, odo, frames) -> list:
    """T_odom_dev at the image time of each frame (build_window(): R_img, p_img = gyro.at(t_img), odo.at(t_img))."""
    rows = []
    for fr in frames:
        t_img = run.image_time(fr)
        R_img, p_img = gyro.at(t_img), odo.at(t_img)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R_img, p_img
        rows.append(dict(frame=int(fr), cam0_stamp_us=int(run.cam_stamp["cam0"][fr]), t_img_us=int(t_img),
                         **{f"T{i}{j}": float(T[i, j]) for i in range(4) for j in range(4)}))
    return rows


POSE_FIELDS = ["frame", "cam0_stamp_us", "t_img_us"] + [f"T{i}{j}" for i in range(4) for j in range(4)]


def load_frame_poses(path: str):
    """-> (poses {frame: 4x4 float64}, times {frame: t_img_us})."""
    poses, times = {}, {}
    for r in csv.DictReader(open(path)):
        fr = int(r["frame"])
        poses[fr] = np.array([[float(r[f"T{i}{j}"]) for j in range(4)] for i in range(4)])
        times[fr] = int(r["t_img_us"])
    return poses, times


def run_masks(run, odometry_csv: str, out_dir: str, params: dict) -> dict:
    a = dict(params)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    if a["self_occ"] > 0:
        self_keys, all_keys, frac = self_mask(run, a)
        np.savez_compressed(os.path.join(out_dir, "self_mask.npz"), keys=self_keys, all_keys=all_keys, frac=frac, voxel=a["self_voxel"],
                            radius=a["self_radius"], occ=a["self_occ"])
        print(f"  self-mask: {len(self_keys)} voxels of {a['self_voxel']} m within {a['self_radius']} m occupied in >= "
              f"{a['self_occ'] * 100:.0f}% of scans ({time.time() - t0:.0f}s)", flush=True)
    if a["operator_occ"] > 0:
        op_mask, frac = operator_mask(run, a)
        np.savez_compressed(os.path.join(out_dir, "operator_mask.npz"), mask=op_mask, frac=frac, az_step=a["operator_az_step"],
                            z_step=a["operator_z_step"], z=np.array(a["operator_z"]), range=a["operator_range"], core=a["operator_core"],
                            occ=a["operator_occ"])
        print(f"  operator mask: {int(op_mask.sum())} sectors ({time.time() - t0:.0f}s)", flush=True)
    odo = Odometry(odometry_csv)
    tasks = select_frames(run, odo.t, a["n_scans"], a["max_gap_ms"])
    rows = frame_poses(run, GyroRotation(run.imu_ts, run.gyro), odo, [fr for fr, _ in tasks])
    with open(os.path.join(out_dir, "frame_poses.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(POSE_FIELDS)
        for r in rows:
            w.writerow([r[k] if isinstance(r[k], int) else repr(r[k]) for k in POSE_FIELDS])
    skipped = len(run.cam_stamp["cam0"]) - len(rows)
    json.dump(dict(a, odometry=odometry_csv, out=out_dir, image_offset_us=run.image_offset_us, frames=len(rows), skipped=skipped),
              open(os.path.join(out_dir, "args.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{len(rows)} frames with a full {a['n_scans']}-scan window ({skipped} skipped) -> {out_dir}")
    return dict(frames=len(rows), skipped=skipped, seconds=round(time.time() - t0, 1))
