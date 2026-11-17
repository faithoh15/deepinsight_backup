"""One recording folder (lidar/*.ply, images_cam0|1/*.jpg, imu_data.csv) and PLY helpers.
Verbatim from proposed_pipeline/common.py (Run, _stamp, SCAN_PERIOD_US, voxel_downsample, write_ply); the only change: the image clock
offset is a required argument (it came from proposed_pipeline/tweaks/<run>.yaml before; now 0_prep/clock.yaml).

Facts used: PLY ASCII `x y z intensity ring`, file name = scan start time (epoch us), 864 columns x 16 rings stored column-major, point
time = scan start + column / n_columns * 50 ms; PLY frame == IMU frame (x forward = cam0 axis, y left, z up).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd


SCAN_PERIOD_US = 50_000


def _stamp(path: str) -> int:
    return int(os.path.splitext(os.path.basename(path))[0])


class Run:
    """One recording folder: lidar/*.ply, images_cam0|1/*.jpg, imu_data.csv."""

    def __init__(self, data_dir: str, image_offset_us: int):
        """image_offset_us = image clock offset of the recording (0_prep/clock.yaml, pipeline.prep.clock); required."""
        self.data_dir = os.path.abspath(data_dir)
        self.name = os.path.basename(self.data_dir.rstrip("/"))
        if image_offset_us is None:
            raise ValueError("image_offset_us is required (0_prep/clock.yaml)")
        self.lidar_files = sorted(glob.glob(os.path.join(self.data_dir, "lidar", "*.ply")), key=_stamp)
        self.lidar_ts = np.array([_stamp(f) for f in self.lidar_files], np.int64)
        self.cam_files = {c: sorted(glob.glob(os.path.join(self.data_dir, f"images_{c}", "*.jpg")), key=_stamp) for c in ("cam0", "cam1")}
        self.cam_stamp = {c: np.array([_stamp(f) for f in v], np.int64) for c, v in self.cam_files.items()}
        self.image_offset_us = int(image_offset_us)
        imu = pd.read_csv(os.path.join(self.data_dir, "imu_data.csv"))
        self.imu_ts = imu["Stamp"].to_numpy(np.int64)
        self.gyro = imu[["GyroX", "GyroY", "GyroZ"]].to_numpy(float)
        self.acc = imu[["AccX", "AccY", "AccZ"]].to_numpy(float)

    def n_scans(self) -> int:
        return len(self.lidar_files)

    def image_time(self, frame: int, cam: str = "cam0") -> int:
        """Clock-corrected image time (us)."""
        return int(self.cam_stamp[cam][frame]) + self.image_offset_us

    def load_scan(self, i: int) -> dict:
        """-> xyz (N,3) f8, intensity, ring, col, point index, range, t_scan_us, t_point_us (estimated, column order)."""
        path = self.lidar_files[i]
        with open(path, "rb") as f:
            n_header = 0
            props, n_vertex = [], None
            while True:
                s = f.readline().decode("ascii", "ignore").strip()
                n_header += 1
                if s.startswith("element vertex"):
                    n_vertex = int(s.split()[-1])
                elif s.startswith("property"):
                    props.append(s.split()[-1])
                elif s == "end_header":
                    break
        d = pd.read_csv(path, sep=" ", skiprows=n_header, header=None, names=props, usecols=range(len(props)), dtype=np.float64).to_numpy()
        if n_vertex is not None and len(d) != n_vertex:
            raise ValueError(f"{path}: header {n_vertex} points, read {len(d)}")
        col = {p: k for k, p in enumerate(props)}
        xyz = d[:, [col["x"], col["y"], col["z"]]]
        ring = d[:, col["ring"]].astype(np.int16)
        n = len(d)
        n_rings = int(ring.max()) + 1 if n else 1
        column_major = n_rings > 1 and n % n_rings == 0 and bool((ring.reshape(-1, n_rings) == np.arange(n_rings)).all())
        cidx = (np.arange(n) // n_rings) if column_major else np.zeros(n, int)
        n_cols = n // n_rings if column_major else 1
        t0 = int(self.lidar_ts[i])
        return dict(xyz=xyz, intensity=d[:, col["intensity"]].astype(np.float32), ring=ring, col=cidx.astype(np.int16),
                    index=np.arange(n, dtype=np.int64), range=np.linalg.norm(xyz, axis=1), t_scan_us=t0,
                    t_point_us=t0 + np.round(cidx / max(n_cols, 1) * SCAN_PERIOD_US).astype(np.int64),
                    timestamp_source="column_estimate" if column_major else "scan", scan=i)


def voxel_downsample(xyz: np.ndarray, voxel: float):
    """Index of one point (the first) per voxel."""
    key = np.floor(xyz / voxel).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return np.sort(first)


def write_ply(path: str, fields: dict):
    """Binary little-endian PLY with arbitrary per-point scalar fields (x, y, z first)."""
    types = {np.dtype("float32"): "float", np.dtype("float64"): "double", np.dtype("int32"): "int", np.dtype("int16"): "short",
             np.dtype("uint8"): "uchar", np.dtype("int64"): "double"}
    names = list(fields)
    n = len(fields[names[0]])
    dt = []
    for k in names:
        a = np.asarray(fields[k])
        if a.dtype == np.int64:
            a = a.astype(np.float64)
            fields[k] = a
        dt.append((k, a.dtype.newbyteorder("<")))
    arr = np.empty(n, dt)
    for k in names:
        arr[k] = fields[k]
    with open(path, "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n".encode())
        for k in names:
            f.write(f"property {types[np.dtype(arr.dtype[k].newbyteorder('='))]} {k}\n".encode())
        f.write(b"end_header\n")
        arr.tofile(f)
