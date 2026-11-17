"""Tracking visuals (2026-11-16 user request): per camera frame only the v3 boxes of that frame, coloured and labelled by their 3D track id
and speed. Predicted positions of tracks without a v3 box and position tails are not drawn. Reads the saved track CSVs, so the pictures can
be redrawn without re-running the tracker."""
from __future__ import annotations

import csv
import os

import cv2
import numpy as np

from pipeline.common.angbox import PAL_TRACK
from pipeline.detect import vis
from pipeline.common.io import read_ply_fields


def _num(s) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def det_labels(persons: list, rows: list, frame_rows: list) -> dict:
    """v3 boxes (persons.csv rows) -> {frame: [dict(key, track_id, speed, row)]}. Speed = the track's smoothed speed at that frame, taken
    only from track_frames rows that have a v3 box (has_det 1); rows of predicted positions are never used."""
    tid = {(int(r["frame"]), r["key"]): int(r["track_id"]) for r in rows}
    spd = {(int(r["frame"]), int(r["track_id"])): _num(r["speed"]) for r in frame_rows if str(r["has_det"]) == "1"}
    out = {}
    for p in persons:
        f = int(p["frame"])
        t = tid.get((f, p["key"]), -1)
        out.setdefault(f, []).append(dict(key=p["key"], track_id=t, speed=spd.get((f, t), float("nan")), row=p))
    return out


def label_text(track_id: int, speed: float) -> str:
    sp = f"{speed:.1f}" if np.isfinite(speed) else "-"
    return f"T{track_id} {sp} m/s" if track_id >= 0 else f"T? {sp} m/s"


def put_label(img, txt: str, x: int, y: int, col) -> None:
    """Readable label above the 2D box top-left corner: filled dark background, coloured text."""
    (w, h), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(y - 4, h + 4)
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + base - 2), (0, 0, 0), -1)
    cv2.putText(img, txt, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)


def plane_lookup(fr_all: list, pose):
    """frames.csv rows -> plane_of(frame): the frame's ground plane (nearest frame with a plane if it had none) with the device pose at
    the frame time. Same rule as scripts/run_tracking.py."""
    fr = {int(r["frame"]): r for r in fr_all if r.get("t_img_us")}
    ok = sorted((int(r["t_img_us"]), r) for r in fr.values() if r.get("R00"))

    def plane_of(frame):
        t = int(fr[frame]["t_img_us"]) if frame in fr else int(ok[0][0])
        src = fr[frame] if frame in fr and fr[frame].get("R00") else min(ok, key=lambda e: abs(e[0] - t))[1]
        R_c, p_c = pose.at(t)
        return dict(R_level=np.array([[float(src[f"R{i}{j}"]) for j in range(3)] for i in range(3)]), ground_z=float(src["ground_z"]), R_c=R_c, p_c=p_c)
    return plane_of


def render(track_dir: str, det: str, ds: dict, geom, plane_of, frame_list: list, out_dir: str, video: bool = False) -> int:
    """Draw out_dir/bev/NNNN.png and out_dir/img/NNNN.jpg for every frame of frame_list (+ bev.mp4, img.mp4). Returns frames written."""
    persons = list(csv.DictReader(open(os.path.join(det, "persons.csv"))))
    lab = det_labels(persons, list(csv.DictReader(open(os.path.join(track_dir, "rows.csv")))),
                     list(csv.DictReader(open(os.path.join(track_dir, "track_frames.csv")))))
    os.makedirs(os.path.join(out_dir, "bev"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    imgs = {f[:4]: f for f in os.listdir(ds["sidebyside"]) if f.endswith(".jpg")}
    for f in frame_list:
        pl = plane_of(f)
        bev = vis.canvas(f"frame {f:04d}: v3 boxes of this frame only, label T<3D id> speed m/s")
        p = imgs.get(f"{f:04d}")
        img = cv2.imread(os.path.join(ds["sidebyside"], p), cv2.IMREAD_REDUCED_COLOR_4) if p else np.full((750, 2000, 3), 20, np.uint8)
        for d in lab.get(f, []):
            r, tid = d["row"], d["track_id"]
            col = tuple(int(c) for c in PAL_TRACK[tid % 1024]) if tid >= 0 else (150, 150, 150)
            path = os.path.join(det, "clouds", f"{f:04d}_{r['key']}.ply")
            if os.path.exists(path):
                pc = read_ply_fields(path)
                m = np.asarray(pc["in_main"]) == 1
                vis.put_points(bev, np.stack([pc["x"], pc["y"], pc["z"]], 1)[m].astype(float), col)
            b = dict(cx=float(r["cx"]), cy=float(r["cy"]), cz=float(r["cz"]), l=float(r["l"]), w=float(r["w"]), h=float(r["h"]), yaw=np.radians(float(r["yaw_deg"])))
            txt = label_text(tid, d["speed"])
            vis.draw_box_bev(bev, b, col, 2, txt)
            vis.draw_box_img(img, geom, pl, b, col, 2)
            x1, y1, x2, y2 = (int(float(r[k]) / 4) for k in ("x1", "y1", "x2", "y2"))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
            put_label(img, txt, x1, y1, col)
        cv2.putText(img, f"frame {f:04d}: v3 boxes of this frame only, label T<3D id> speed m/s", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(os.path.join(out_dir, "bev", f"{f:04d}.png"), bev)
        cv2.imwrite(os.path.join(out_dir, "img", f"{f:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if video:
        vis.make_video(os.path.join(out_dir, "bev"), os.path.join(out_dir, "bev.mp4"))
        vis.make_video(os.path.join(out_dir, "img"), os.path.join(out_dir, "img.mp4"))
    return len(frame_list)
