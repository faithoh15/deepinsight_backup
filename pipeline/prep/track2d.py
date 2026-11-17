"""2D person tracking of the YOLO boxes, polar mode (proposed_pipeline/track_2d.py --mode polar, verbatim: the polar branch of main()
became run_track2d(), finish() unchanged; the other modes (plain / gmc / gyro / world BoT-SORT) are not used and were left out; the
device pose and time of each frame come from 0_prep/masks/frame_poses.csv instead of the 1 GB LiDAR windows).

Input: detections.csv of the yolo + operator stages; boxes with operator = 1 are excluded.
Output: tracks.csv (frame, track_id, det, cam, x1 y1 x2 y2, conf, dist_m, u_s v_s), summary.json, args.json (vis/ with debug figures).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from pipeline.common.cam_geom import CamGeom
from pipeline.prep import track_polar
from pipeline.prep.masks import load_frame_poses

TRACK_FIELDS = ["frame", "track_id", "det", "cam", "x1", "y1", "x2", "y2", "conf", "dist_m", "u_s", "v_s"]   # track_polar make_row


def finish(a, out, trows, frames, backfilled, t0, extra=None):
    """tracks.csv, summary.json, video (all modes)."""
    trows.sort(key=lambda t: (t["frame"], t["track_id"]))
    with open(os.path.join(out, "tracks.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(trows[0].keys()) if trows else TRACK_FIELDS)        # no confirmed track: header only
        wr.writeheader()
        wr.writerows(trows)
    # summary
    ids = {}
    for t in trows:
        ids.setdefault(t["track_id"], []).append(t["frame"])
    lengths = np.array([len(v) for v in ids.values()])
    spans = np.array([max(v) - min(v) + 1 for v in ids.values()])
    # next-frame continuity: fraction of tracked boxes whose id is tracked again in the next camera frame
    nxt = {}
    for t in trows:
        nxt.setdefault(t["frame"], set()).add(t["track_id"])
    cont = [tid in nxt.get(f + 1, set()) for f, s in nxt.items() for tid in s if f + 1 in nxt or (f + 1) in frames]
    segs = {"companion 142-168 cam1": (142, 169, 4000, 8000), "board person 115-119": (115, 120, 1000, 2000), "woman 225-237": (225, 238, 0, 3900),
            "man 347-364": (347, 365, 0, 7800)}
    seg_ids = {}
    for name, (lo, hi, xlo, xhi) in segs.items():
        ts = [t for t in trows if lo <= t["frame"] < hi and xlo <= (t["x1"] + t["x2"]) / 2 < xhi and t["dist_m"] < 12]
        c = {}
        for t in ts:
            c[t["track_id"]] = c.get(t["track_id"], 0) + 1
        seg_ids[name] = dict(ids=len(c), frames_tracked=len({t["frame"] for t in ts}), of=hi - lo, main_id_frames=max(c.values()) if c else 0)
    summary = dict(mode=a.mode, frames=len(frames), tracked_boxes=len(trows), tracks=len(ids), mean_len=round(float(lengths.mean()), 2),
                   median_len=float(np.median(lengths)), mean_span=round(float(spans.mean()), 2), tracks_ge5=int((lengths >= 5).sum()),
                   next_frame_continuity=round(float(np.mean(cont)), 3) if cont else None, backfilled_first_frames=backfilled, segments=seg_ids,
                   seconds=round(time.time() - t0, 1), **(extra or {}))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))
    if a.video:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "4", "-pattern_type", "glob", "-i", os.path.join(out, "vis", "*.jpg"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", os.path.join(out, "vis.mp4")], check=False)


def run_track2d(det_dir: str, images_dir: str, data_dir: str, frame_poses_csv: str, out_dir: str, params: dict,
                frames: str | None = None, figure: bool = False) -> dict:
    a = SimpleNamespace(**params, det=det_dir, images=images_dir, data=data_dir, frame_poses=frame_poses_csv, out=out_dir, frames=frames,
                        video=figure, vis_every=1 if figure else 0)
    out = out_dir
    os.makedirs(os.path.join(out, "vis"), exist_ok=True)
    rows = [r for r in csv.DictReader(open(os.path.join(a.det, "detections.csv"))) if int(r.get("operator", r["operator_flag"])) == 0]
    for r in rows:
        r["frame"] = int(r["frame"])
        for k in ("x1", "y1", "x2", "y2", "conf", "dist_m"):
            r[k] = float(r[k])
    by = {}
    for r in rows:
        by.setdefault(r["frame"], []).append(r)
    files = sorted(glob.glob(os.path.join(a.images, "[0-9]" * 4 + "_*.jpg")))
    frames = [int(os.path.basename(f)[:4]) for f in files]
    fpath = dict(zip(frames, files))
    if a.frames:
        lo, hi = map(int, a.frames.split(":"))
        frames = [f for f in frames if lo <= f < hi]
    geom = CamGeom(a.data)
    poses, times = load_frame_poses(a.frame_poses)
    json.dump(dict(vars(a), out=out), open(os.path.join(out, "args.json"), "w"), indent=1, ensure_ascii=False)
    trows, backfilled, t0, stats = track_polar.run(a, out, by, frames, fpath, poses, times, geom)
    finish(a, out, trows, frames, backfilled, t0, extra=dict(polar=stats))
    return dict(tracked_boxes=len(trows), tracks=len({t["track_id"] for t in trows}), seconds=round(time.time() - t0, 1))
