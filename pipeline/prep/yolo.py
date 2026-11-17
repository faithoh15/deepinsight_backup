"""YOLO person detection on the side-by-side images, one CSV for the whole run (proposed_pipeline/detect_persons.py, verbatim; main()
became run_yolo(), values from configs/pipeline.yaml prep.yolo; the optional --quads pass (off in every run) was left out; the image
clock offset is passed in instead of read from tweaks/; vis images only with debug figures).

Scheme: 1. full 8000x3000 image at imgsz full (1280); 2. cam0 tile and cam1 tile (4000x3000 each) at imgsz tile (1920); person class
only, conf >= conf (0.15); merged with NMS (IoU nms 0.5), then a box lying >= iom (0.6) inside a higher-confidence box is dropped
unless that box is operator-like. Per box: cam, distance from the box height (1.7 m person), operator_flag by the edge rule.
Output: detections.csv (frame, cam0_stamp_us, t_img_us, det, cam, x1, y1, x2, y2, w, h, conf, source, dist_m, operator_flag,
touches_bottom, truncated_top), frames.csv (per frame counts), args.json.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

from pipeline.common.config import FINAL

for _k, _v in {"YOLO_CONFIG_DIR": "ultralytics", "XDG_CACHE_HOME": "xdg", "MPLCONFIGDIR": "mpl", "TORCH_HOME": "torch"}.items():
    _p = os.path.join(FINAL, "output", ".cache", _v)                 # keep library caches out of the home folder
    os.makedirs(_p, exist_ok=True)
    os.environ.setdefault(_k, _p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.ops import nms  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DET_FIELDS = ["frame", "cam0_stamp_us", "t_img_us", "det", "cam", "x1", "y1", "x2", "y2", "w", "h", "conf", "source", "dist_m", "operator_flag",
              "touches_bottom", "truncated_top"]
FRAME_FIELDS = ["frame", "cam0_stamp_us", "n_raw", "n_det", "n_operator", "n_person"]


def person_class_ids(model):
    names = model.names
    ids = [i for i, n in names.items() if n.lower() in ("person", "pedestrian")]
    if not ids:
        raise SystemExit(f"no person class in the model: {names}")
    return ids


def run_pass(model, img, imgsz, conf, classes, x_off, source):
    r = model.predict(img, imgsz=imgsz, conf=conf, classes=classes, verbose=False, half=True)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    cf = r.boxes.conf.cpu().numpy()
    out = []
    for (x1, y1, x2, y2), c in zip(xyxy, cf):
        out.append([x1 + x_off, y1, x2 + x_off, y2, float(c), source])
    return out


def merge_contained(dets, iom, W, edge):
    """Drop a box whose intersection with a higher-confidence box is >= iom of the smaller box's area (full-body box
    vs a part box of the same person, or the two tile boxes of a person on the cam0/cam1 seam): NMS by IoU misses these.
    Never drop a box because it lies inside an OPERATOR-like box (touches the outer image edge and is tall): a person
    standing in front of / beside the device holder is a different person."""
    dets = sorted(dets, key=lambda d: -d[4])
    out = []

    def operator_like(b):
        return (b[0] < edge or b[2] > W - edge) and (b[3] - b[1]) > 0.35 * 3000

    for d in dets:
        ax1, ay1, ax2, ay2 = d[:4]
        aa = max(ax2 - ax1, 1) * max(ay2 - ay1, 1)
        dup = False
        for k in out:
            ix = max(0, min(ax2, k[2]) - max(ax1, k[0]))
            iy = max(0, min(ay2, k[3]) - max(ay1, k[1]))
            ka = max(k[2] - k[0], 1) * max(k[3] - k[1], 1)
            if ix * iy / min(aa, ka) >= iom and not (operator_like(k) and not operator_like(d)):
                dup = True
                break
        if not dup:
            out.append(d)
    return out


def dist_from_height(h_px, H):
    dtheta = np.radians(h_px / (H - 1) * 120.0)
    return 1.7 / (2.0 * np.tan(dtheta / 2.0)) if dtheta > 0 else float("inf")


def run_yolo(images_dir: str, out_dir: str, model_path: str, params: dict, image_offset_us: int, frames: str | None = None,
             figure: bool = False) -> dict:
    a = SimpleNamespace(**params, images=images_dir, model=model_path, out=out_dir, frames=frames, quads=False, video=figure)
    out = out_dir
    IMAGE_OFFSET_US = int(image_offset_us)
    os.makedirs(os.path.join(out, "vis"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.images, "[0-9]" * 4 + "_*.jpg")))
    if a.frames:
        lo, hi = map(int, a.frames.split(":"))
        files = [f for f in files if lo <= int(os.path.basename(f)[:4]) < hi]
    model = YOLO(a.model)
    classes = person_class_ids(model)
    json.dump(dict(vars(a), out=out, model_names=model.names, person_classes=classes, n_images=len(files)),
              open(os.path.join(out, "args.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{len(files)} images, model {a.model} (person classes {classes}) -> {out}")
    rows, frows = [], []
    t0 = time.time()
    for k, f in enumerate(files):
        fr = int(os.path.basename(f)[:4])
        stamp = int(os.path.basename(f)[5:-4])
        img = cv2.imread(f)
        H, W = img.shape[:2]
        half = W // 2
        dets = run_pass(model, img, a.full, a.conf, classes, 0, "full")
        dets += run_pass(model, img[:, :half], a.tile, a.conf, classes, 0, "cam0")
        dets += run_pass(model, img[:, half:], a.tile, a.conf, classes, half, "cam1")
        kept = []
        if dets:
            b = torch.tensor([d[:4] for d in dets], dtype=torch.float32)
            s = torch.tensor([d[4] for d in dets], dtype=torch.float32)
            keep = nms(b, s, a.nms).numpy()
            kept = merge_contained([dets[i] for i in keep], a.iom, W, a.edge)
        n_op = 0
        for j, (x1, y1, x2, y2, c, src) in enumerate(sorted(kept, key=lambda d: -d[4])):
            w, h = x2 - x1, y2 - y1
            cam = 0 if (x1 + x2) / 2 < half else 1
            dist = dist_from_height(h, H)
            bottom = y2 >= H - 2
            top = y1 <= 1
            outer = (cam == 0 and x1 < a.edge) or (cam == 1 and x2 > W - a.edge)
            op = int(outer and (dist < 2.0 or w / max(h, 1) > 0.5 or bottom))
            n_op += op
            rows.append(dict(frame=fr, cam0_stamp_us=stamp, t_img_us=stamp + IMAGE_OFFSET_US, det=j, cam=cam, x1=round(float(x1), 1), y1=round(float(y1), 1),
                             x2=round(float(x2), 1), y2=round(float(y2), 1), w=round(float(w), 1), h=round(float(h), 1), conf=round(float(c), 4),
                             source=src, dist_m=round(float(min(dist, 99)), 2), operator_flag=op, touches_bottom=int(bottom), truncated_top=int(top)))
        frows.append(dict(frame=fr, cam0_stamp_us=stamp, n_raw=len(dets), n_det=len(kept), n_operator=n_op, n_person=len(kept) - n_op))
        if figure:
            vis = cv2.resize(img, (W // 4, H // 4))
            for r in rows:
                if r["frame"] != fr:
                    continue
                col = (128, 128, 128) if r["operator_flag"] else (0, 255, 0) if r["conf"] >= 0.4 else (0, 200, 255)
                p1, p2 = (int(r["x1"] / 4), int(r["y1"] / 4)), (int(r["x2"] / 4), int(r["y2"] / 4))
                cv2.rectangle(vis, p1, p2, col, 2)
                cv2.putText(vis, f"{r['conf']:.2f} {r['dist_m']:.1f}m", (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            cv2.putText(vis, f"frame {fr:04d}  {a.tag}: green conf>=0.4, yellow 0.15-0.4, grey = operator flag", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(out, "vis", f"{fr:04d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if k % 50 == 0:
            print(f"  {k + 1}/{len(files)} frame {fr:04d}: {len(kept)} boxes ({n_op} operator) ({time.time() - t0:.0f}s)", flush=True)
    for name, rs, fields in (("detections.csv", rows, DET_FIELDS), ("frames.csv", frows, FRAME_FIELDS)):
        with open(os.path.join(out, name), "w", newline="") as fh:
            wr = csv.DictWriter(fh, list(rs[0].keys()) if rs else fields)               # no box at all: header only
            wr.writeheader()
            wr.writerows(rs)
    n_det = sum(r["n_det"] for r in frows)
    n_op = sum(r["n_operator"] for r in frows)
    print(f"done in {time.time() - t0:.0f}s: {n_det} boxes in {len(frows)} frames, {n_op} operator-flagged, "
          f"{n_det - n_op} person boxes ({(n_det - n_op) / max(len(frows), 1):.2f} per frame); conf>=0.4: "
          f"{sum(1 for r in rows if not r['operator_flag'] and r['conf'] >= 0.4)}")
    if a.video:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "4", "-pattern_type", "glob", "-i", os.path.join(out, "vis", "*.jpg"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", os.path.join(out, "vis.mp4")], check=False)
    return dict(images=len(files), boxes=n_det, operator_flagged=n_op, seconds=round(time.time() - t0, 1))
