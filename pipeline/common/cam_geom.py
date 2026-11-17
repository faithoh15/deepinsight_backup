"""Camera geometry for the DIMENVUE D2 run: device (PLY/IMU) frame <-> camera frames <-> side-by-side unwrapped pixels.

Conventions (README section 2):
  * imu_camera0.yaml `transform` = 16 values in COLUMN-major order = T_imu_cam0 (cam0 -> IMU/device).
  * camera.yaml camera_pose of camera_1 = T_cam0_cam1 (row-major).
  * rotation corrections (the yaml files are never edited): R = Rz(yaw) Ry(pitch) Rx(roll) with
      imu rpy = (-0.2460, +0.1858, -0.2374) deg,  cam1 rpy = (+0.3611, +0.0749, -0.4285) deg
      T_cam0_ply = T_cam0_imu . R(imu_rpy)        T_cam1_ply = R(cam1_rpy) . T_cam1_cam0 . T_cam0_ply
  * camera frame: x right, y down, z forward. Unwrapped tile (unwrap_sidebyside.py): yaw = atan2(x, z),
    pitch = atan2(y, hypot(x, z)); u = (yaw + 90) / 180 (W - 1) (+ W for cam1), v = (pitch + 60) / 120 (H - 1).
  * a device-frame direction belongs to the camera whose optical axis is closer (hemisphere split).
"""
from __future__ import annotations

import os

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

IMU_RPY_DEG = (-0.2460, 0.1858, -0.2374)
CAM1_RPY_DEG = (0.3611, 0.0749, -0.4285)


def _rot_rpy(rpy_deg):
    r, p, y = np.radians(rpy_deg)
    return Rotation.from_euler("z", y).as_matrix() @ Rotation.from_euler("y", p).as_matrix() @ Rotation.from_euler("x", r).as_matrix()


class CamGeom:
    def __init__(self, data_dir: str, W: int = 4000, H: int = 3000, corrections: bool = True):
        cfg = os.path.join(data_dir, "calib", "config")
        t = np.array(yaml.safe_load(open(os.path.join(cfg, "imu_camera0.yaml"), encoding="utf-8-sig"))["transform"], float)
        T_imu_cam0 = t.reshape(4, 4, order="F")                       # column-major
        cam = yaml.safe_load(open(os.path.join(cfg, "camera.yaml"), encoding="utf-8-sig"))
        T_cam0_cam1 = np.array(cam["camera_1"]["camera_pose"], float).reshape(4, 4)   # row-major
        T_cam0_ply = np.linalg.inv(T_imu_cam0)
        T_cam1_ply = np.linalg.inv(T_cam0_cam1) @ T_cam0_ply
        if corrections:
            Ri = np.eye(4)
            Ri[:3, :3] = _rot_rpy(IMU_RPY_DEG)
            T_cam0_ply = T_cam0_ply @ Ri
            Rc = np.eye(4)
            Rc[:3, :3] = _rot_rpy(CAM1_RPY_DEG)
            T_cam1_ply = Rc @ np.linalg.inv(T_cam0_cam1) @ T_cam0_ply
        self.T = {0: T_cam0_ply, 1: T_cam1_ply}            # device -> camera
        self.W, self.H = W, H

    # ---- device frame -> pixels ----
    def project(self, xyz_dev: np.ndarray):
        """Device-frame points (N,3) -> cam id (N,), u (N,), v (N,) in the side-by-side image, depth along the axis.
        Each point goes to the camera whose axis is closer to it."""
        P = np.asarray(xyz_dev, float)
        out_cam = np.zeros(len(P), int)
        out_u = np.zeros(len(P))
        out_v = np.zeros(len(P))
        out_z = np.zeros(len(P))
        cos = []
        pc = {}
        for c in (0, 1):
            Rm, tm = self.T[c][:3, :3], self.T[c][:3, 3]
            q = P @ Rm.T + tm
            pc[c] = q
            cos.append(q[:, 2] / np.maximum(np.linalg.norm(q, axis=1), 1e-9))
        out_cam[:] = np.where(cos[0] >= cos[1], 0, 1)
        for c in (0, 1):
            m = out_cam == c
            q = pc[c][m]
            yaw = np.arctan2(q[:, 0], q[:, 2])
            pitch = np.arctan2(q[:, 1], np.hypot(q[:, 0], q[:, 2]))
            out_u[m] = (np.degrees(yaw) + 90.0) / 180.0 * (self.W - 1) + c * self.W
            out_v[m] = (np.degrees(pitch) + 60.0) / 120.0 * (self.H - 1)
            out_z[m] = q[:, 2]
        return out_cam, out_u, out_v, out_z

    # ---- pixels -> device-frame rays ----
    def ray(self, u, v):
        """Side-by-side pixel(s) -> unit direction(s) in the device frame."""
        u = np.atleast_1d(np.asarray(u, float))
        v = np.atleast_1d(np.asarray(v, float))
        cam = (u >= self.W).astype(int)
        yaw = np.radians((u - cam * self.W) / (self.W - 1) * 180.0 - 90.0)
        pitch = np.radians(v / (self.H - 1) * 120.0 - 60.0)
        d = np.stack([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)], 1)
        out = np.zeros_like(d)
        for c in (0, 1):
            m = cam == c
            Rm = self.T[c][:3, :3]
            out[m] = d[m] @ Rm                     # R^T d : camera -> device (rotation only, rays)
        return out

    def cam_center_dev(self, c):
        """Camera centre in the device frame."""
        Rm, tm = self.T[c][:3, :3], self.T[c][:3, 3]
        return -Rm.T @ tm
