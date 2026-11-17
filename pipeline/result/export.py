"""Speed classes of the 3D tracks and one combined table per v3 box (stage result, 2026-11-17). Reads 1_detection, 2_tracking and the
pose correction; writes only 4_result/.

Classes (configs/pipeline.yaml result.speed_class): 저속 = speed < slow_max (2.78 m/s = 10 km/h), 고속 = slow_max <= speed <= fast_max
(5.56 m/s = 20 km/h), 초고속 = speed > fast_max, 미정 = no speed (mostly tracks with one observation, never confirmed: no velocity and no median speed).
  speed_class        class of the track's median speed (tracks.csv speed_med, over its observed and interpolated frames): one per id
  speed_class_frame  class of the filtered speed in that frame (track_frames.csv speed)
Speeds and headings stay in the tracking (legacy odometry) frame: the pose correction is built for the map and changed no class over the
17 recordings (7,063 confirmed frames, 2026-11-17). Box centre and box yaw are also given in the corrected map frame of map_static.ply
(x_map, y_map, z_map, yaw_map_deg; correction field at the image time, the same field the map uses), and the track heading rotated
into it (heading_map_deg).
"""
from __future__ import annotations

import csv
import math
import os
from collections import Counter

import numpy as np

from pipeline.common.pose import Odometry
from pipeline.mapping.pose_refine import PoseCorrection

SLOW, FAST, VERY_FAST, UNKNOWN = "저속", "고속", "초고속", "미정"
CLASSES = [SLOW, FAST, VERY_FAST, UNKNOWN]
BOX_FIELDS = ["frame", "t_img_us", "key", "track_id", "track_id_2d", "track_confirmed", "valid", "l", "w", "h",
              "cx_odom", "cy_odom", "cz_odom", "yaw_odom_deg", "yaw_source", "yaw_reliable", "x_map", "y_map", "z_map", "yaw_map_deg",
              "cx", "cy", "cz", "yaw_deg", "speed", "speed_lo95", "speed_hi95", "vx", "vy", "heading_deg", "heading_map_deg",
              "track_status", "segment", "speed_class", "speed_class_frame"]


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(x: float) -> str:
    return f"{x:.6f}" if math.isfinite(x) else "nan"


def speed_class(speed, slow_max: float, fast_max: float) -> str:
    x = _num(speed)
    if not math.isfinite(x):
        return UNKNOWN
    if x < slow_max:
        return SLOW
    return FAST if x <= fast_max else VERY_FAST


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def _write(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _yaw_map(R, deg):
    """direction angle deg (odometry frame, about +z) rotated by the correction R (N,3,3) -> deg in the map frame (nan stays nan)."""
    a = np.radians(np.asarray(deg, float))
    d = np.einsum("nij,nj->ni", R, np.c_[np.cos(a), np.sin(a), np.zeros(len(a))])
    out = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    out[~np.isfinite(a)] = np.nan
    return out


def build_result(det_dir: str, track_dir: str, correction_csv: str, odometry_csv: str, out_dir: str, cfg: dict) -> dict:
    lo, hi = cfg["speed_class"]["slow_max"], cfg["speed_class"]["fast_max"]
    os.makedirs(out_dir, exist_ok=True)
    tr_fields, tracks = _read(os.path.join(track_dir, "tracks.csv"))
    track_cls = {r["track_id"]: speed_class(r["speed_med"], lo, hi) for r in tracks}
    for r in tracks:
        r["speed_class"] = track_cls[r["track_id"]]
    _write(os.path.join(out_dir, "tracks.csv"), tr_fields + ["speed_class"], tracks)

    tf_fields, frames = _read(os.path.join(track_dir, "track_frames.csv"))
    for r in frames:
        vx, vy = _num(r["vx"]), _num(r["vy"])
        r["heading_deg"] = _fmt(math.degrees(math.atan2(vy, vx))) if math.isfinite(vx) and math.isfinite(vy) else "nan"
        r["speed_class"] = track_cls.get(r["track_id"], UNKNOWN)
        r["speed_class_frame"] = speed_class(r["speed"], lo, hi)
    _write(os.path.join(out_dir, "track_frames.csv"), tf_fields + ["heading_deg", "speed_class", "speed_class_frame"], frames)

    frame_row = {(r["frame"], r["track_id"]): r for r in frames}
    id_row = {(r["frame"], r["key"]): r for r in _read(os.path.join(track_dir, "rows.csv"))[1]}
    persons = _read(os.path.join(det_dir, "persons.csv"))[1]
    n = len(persons)
    Xm, yaw_m, head_m = np.full((n, 3), np.nan), np.full(n, np.nan), np.full(n, np.nan)
    if n:
        corr, odo = PoseCorrection.from_csv(correction_csv), Odometry(odometry_csv)
        t = np.array([int(p["t_img_us"]) for p in persons], np.int64)
        X = np.array([[_num(p["cx_odom"]), _num(p["cy_odom"]), _num(p["cz_odom"])] for p in persons])
        Xm = corr.apply(X, t, odo.at(t))
        R = corr.at(t)[0]
        yaw_m = _yaw_map(R, [_num(p["yaw_odom_deg"]) for p in persons])
        heading = [_num(frame_row[(p["frame"], id_row[(p["frame"], p["key"])]["track_id"])]["heading_deg"])
                   if (p["frame"], p["key"]) in id_row and (p["frame"], id_row[(p["frame"], p["key"])]["track_id"]) in frame_row
                   else float("nan") for p in persons]
        head_m = _yaw_map(R, heading)
    boxes = []
    for i, p in enumerate(persons):
        ir = id_row.get((p["frame"], p["key"]), {})
        tid = ir.get("track_id", "")
        fr = frame_row.get((p["frame"], tid), {})
        b = {k: p[k] for k in ("frame", "t_img_us", "key", "valid", "l", "w", "h", "cx_odom", "cy_odom", "cz_odom", "yaw_odom_deg",
                               "yaw_source", "yaw_reliable", "cx", "cy", "cz", "yaw_deg")}
        b.update(track_id=tid, track_id_2d=p["track_id"], track_confirmed=ir.get("track_confirmed", ""),
                 x_map=_fmt(Xm[i, 0]), y_map=_fmt(Xm[i, 1]), z_map=_fmt(Xm[i, 2]), yaw_map_deg=_fmt(yaw_m[i]),
                 speed=fr.get("speed", "nan"), speed_lo95=fr.get("speed_lo95", "nan"), speed_hi95=fr.get("speed_hi95", "nan"),
                 vx=fr.get("vx", "nan"), vy=fr.get("vy", "nan"), heading_deg=fr.get("heading_deg", "nan"), heading_map_deg=_fmt(head_m[i]),
                 track_status=fr.get("status", ""), segment=fr.get("segment", ""),
                 speed_class=track_cls.get(tid, UNKNOWN), speed_class_frame=speed_class(fr.get("speed"), lo, hi))
        boxes.append(b)
    _write(os.path.join(out_dir, "boxes.csv"), BOX_FIELDS, boxes)

    def count(xs):
        c = Counter(xs)
        return {k: c.get(k, 0) for k in CLASSES}
    conf = [r for r in tracks if r.get("confirmed") == "1"]
    by_track = {}
    for r in frames:
        if r["segment"] in ("obs", "interp"):
            by_track.setdefault(r["track_id"], set()).add(r["speed_class_frame"])
    return dict(thresholds_mps=dict(slow_max=lo, fast_max=hi), tracks=len(tracks), tracks_by_class=count(r["speed_class"] for r in tracks),
                confirmed_tracks_by_class=count(r["speed_class"] for r in conf), frames=len(frames),
                frames_by_class_frame=count(r["speed_class_frame"] for r in frames),
                tracks_changing_class=sum(1 for s in by_track.values() if len(s - {UNKNOWN}) > 1),
                boxes=len(boxes), boxes_valid=sum(1 for b in boxes if b["valid"] == "1"),
                boxes_by_class=count(b["speed_class"] for b in boxes), boxes_by_class_frame=count(b["speed_class_frame"] for b in boxes))
