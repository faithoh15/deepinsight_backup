"""2D person boxes of the legacy detector/tracker and the time-aware projection used to test LiDAR points against them.

PersonBoxes reads (read only) yolo_coco26l/detections.csv (operator column, conf, box-height distance), track2d_polar/tracks.csv
(2D ids) and sidebyside/frames.csv (the image times used everywhere else in the detector; detections.csv carries its own t_img_us
column computed with a possibly different clock offset, so it is NOT used for time). Keys: "t<id>" for tracked rows, "d<frame>_<det>"
for untracked detections (conf >= det_fallback_conf). Operator boxes are removed from both.
boxes_at(t): the angular box (AngBox: cam, yaw, pitch, dyaw, dpitch in the tile camera frame, device pose at t) of every person whose
box can be interpolated to t from two frames of the same id <= max_gap_s apart, or taken from one frame within single_dt_s.
Only the pixel box is used: direction = ray through the box centre, size = box angles, distance = box-height estimate (parallax of
the moving device between the frame time and t is corrected with that rough distance; the tracker's position is never used).
"""
from __future__ import annotations

import csv
import os

import numpy as np

from pipeline.common.angbox import AngBox, TILE_H, TILE_W
from pipeline.common.angbox import box_to_angles


def project_points(geom, xyz_dev):
    """Device-frame points -> (cam, yaw_deg, pitch_deg, depth, u, v) in the side-by-side image (camera = nearer axis)."""
    cam, u, v, z = geom.project(np.asarray(xyz_dev, float))
    yaw = (u - cam * TILE_W) / (TILE_W - 1) * 180.0 - 90.0
    pitch = v / (TILE_H - 1) * 120.0 - 60.0
    return cam, yaw, pitch, z, u, v


def fov_mask(cam, pitch, depth, u, cfg):
    pm = cfg["fov"]["pitch_max_deg"]
    uu = u - cam * TILE_W
    return (depth > 0) & (np.abs(pitch) <= pm) & (uu >= 0) & (uu < TILE_W)


def in_box_mask(cam, yaw, pitch, abox, margin: float):
    """Inside the angular box enlarged by `margin` x size on each side."""
    dy = np.abs((yaw - abox.yaw + 180.0) % 360.0 - 180.0)
    dp = np.abs(pitch - abox.pitch)
    return (cam == abox.cam) & (dy <= abox.dyaw * (0.5 + margin)) & (dp <= abox.dpitch * (0.5 + margin))


def _dir_to_cam(geom, d):
    """Device-frame direction -> (cam, yaw, pitch) of the nearer camera (rotation only, same as the unwrapped tiles)."""
    best = None
    for c in (0, 1):
        q = geom.T[c][:3, :3] @ d
        cosang = q[2] / max(np.linalg.norm(q), 1e-9)
        if best is None or cosang > best[0]:
            best = (cosang, c, q)
    _, c, q = best
    return c, float(np.degrees(np.arctan2(q[0], q[2]))), float(np.degrees(np.arctan2(q[1], np.hypot(q[0], q[2]))))


class PersonBoxes:
    def __init__(self, sidebyside_dir: str, detections_csv: str, tracks_csv: str, pose, geom, cfg: dict, dist_clip=(0.8, 15.0)):
        b = cfg["boxes"]
        self.pose, self.geom, self.cfg = pose, geom, cfg
        self.t_img = {int(r["frame"]): int(r["t_img_us"]) for r in csv.DictReader(open(os.path.join(sidebyside_dir, "frames.csv")))}
        self.dets, operator = {}, set()
        for r in csv.DictReader(open(detections_csv)):
            fr, det = int(r["frame"]), int(r["det"])
            self.dets[(fr, det)] = r
            if int(r.get("operator", r["operator_flag"])):
                operator.add((fr, det))
        pose_cache = {}

        def make_row(fr, det, x1, y1, x2, y2, conf, dist_m):
            d = self.dets.get((fr, det))
            trunc = int(int(d["truncated_top"]) or int(d["touches_bottom"])) if d else 0
            if fr not in pose_cache:
                pose_cache[fr] = pose.at(self.t_img[fr])
            R, p = pose_cache[fr]
            dirw = R @ geom.ray((x1 + x2) / 2, (y1 + y2) / 2)[0]
            pw = p + dirw * float(np.clip(dist_m, *dist_clip))        # rough world point for the device-parallax correction only
            return dict(frame=fr, det=det, t=self.t_img[fr], x1=x1, y1=y1, x2=x2, y2=y2, conf=conf, dist_m=dist_m, trunc=trunc, pw=pw)

        self.rows = {}
        tracked = set()
        for r in csv.DictReader(open(tracks_csv)):
            fr, det = int(r["frame"]), int(r["det"])
            if fr not in self.t_img or (fr, det) in operator:
                continue
            tracked.add((fr, det))
            self.rows.setdefault(f"t{int(r['track_id'])}", []).append(make_row(fr, det, *[float(r[k]) for k in ("x1", "y1", "x2", "y2", "conf", "dist_m")]))
        for (fr, det), d in self.dets.items():
            if (fr, det) in tracked or (fr, det) in operator or fr not in self.t_img or float(d["conf"]) < b["det_fallback_conf"]:
                continue
            self.rows[f"d{fr}_{det}"] = [make_row(fr, det, *[float(d[k]) for k in ("x1", "y1", "x2", "y2", "conf", "dist_m")])]
        for k in self.rows:
            self.rows[k].sort(key=lambda r: r["t"])
        self.times = {k: np.array([r["t"] for r in v]) for k, v in self.rows.items()}
        self.by_frame = {}
        for k, rows in self.rows.items():
            for r in rows:
                self.by_frame.setdefault(r["frame"], []).append((k, r))

    def frame_boxes(self, frame: int):
        """[(key, row)] of the boxes present in this camera frame (row: frame, det, t, x1..y2, conf, dist_m, trunc)."""
        return list(self.by_frame.get(frame, []))

    def _abox(self, key, t_us, pw, dyaw, dpitch, conf, dist, trunc, mode, r0, r1):
        R, p = self.pose.at(int(t_us))
        d = R.T @ (pw - p)
        cam, yaw, pitch = _dir_to_cam(self.geom, d / max(np.linalg.norm(d), 1e-9))
        tid = int(key[1:]) if key.startswith("t") else -1
        return AngBox(key, tid, "track" if key.startswith("t") else "det", cam, yaw, pitch, float(dyaw), float(dpitch), float(conf), float(dist),
                        float(r1["y2"] - r1["y1"]), int(trunc), mode, 1.0, r0["frame"], r1["frame"], r0["t"], r1["t"], pw, r1["frame"], r1["det"])

    def boxes_at(self, t_us: int):
        """[(key, AngBox)] of every person whose 2D box exists at time t (interpolated between two frames or one frame)."""
        b = self.cfg["boxes"]
        out = []
        for k, ts in self.times.items():
            if np.min(np.abs(ts - t_us)) > b["max_gap_s"] * 1e6:
                continue
            rows = self.rows[k]
            i1 = int(np.searchsorted(ts, t_us, side="right"))
            r0 = rows[i1 - 1] if i1 >= 1 else None
            r1 = rows[i1] if i1 < len(rows) else None
            if r0 is not None and r1 is not None and r1["t"] - r0["t"] <= b["max_gap_s"] * 1e6 and r0["t"] <= t_us <= r1["t"]:
                lam = (t_us - r0["t"]) / max(r1["t"] - r0["t"], 1)
                a0, a1 = box_to_angles(r0["x1"], r0["y1"], r0["x2"], r0["y2"]), box_to_angles(r1["x1"], r1["y1"], r1["x2"], r1["y2"])
                out.append((k, self._abox(k, t_us, (1 - lam) * r0["pw"] + lam * r1["pw"], (1 - lam) * a0[3] + lam * a1[3], (1 - lam) * a0[4] + lam * a1[4],
                                          min(r0["conf"], r1["conf"]), (1 - lam) * r0["dist_m"] + lam * r1["dist_m"], r0["trunc"] or r1["trunc"], "interp", r0, r1)))
                continue
            near = [r for r in (r0, r1) if r is not None and abs(r["t"] - t_us) <= b["single_dt_s"] * 1e6]
            if not near:
                continue
            r = min(near, key=lambda r: abs(r["t"] - t_us))
            a = box_to_angles(r["x1"], r["y1"], r["x2"], r["y2"])
            out.append((k, self._abox(k, t_us, r["pw"], a[3], a[4], r["conf"], r["dist_m"], r["trunc"], "single", r, r)))
        return out

    def key_track(self, key: str) -> int:
        return int(key[1:]) if key.startswith("t") else -1
