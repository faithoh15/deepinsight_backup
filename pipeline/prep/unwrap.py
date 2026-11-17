"""Undistorted side-by-side images: cam0 unwrapped on the left, cam1 on the right, one image per cam0 frame
(proposed_pipeline/unwrap_sidebyside.py, verbatim; main() became run_unwrap(), values from configs/pipeline.yaml prep.unwrap).

  * KB4 fisheye model (camera.yaml intrinsic fx fy cx cy, distortion k1..k4) via cv2.fisheye.projectPoints
  * each camera is unwrapped to an equirectangular tile: yaw -90..+90 deg (horizontal), pitch -60..+60 deg (vertical)
      camera frame (x right, y down, z forward): yaw = atan2(x, z), pitch = atan2(y, hypot(x, z))
      u = (yaw + 90) / 180 * (W - 1)   (+ W for cam1),   v = (pitch + 60) / 120 * (H - 1)
    with W = width (4000) and H = W * 3 / 4 (3000) -> final image 2W x H (8000 x 3000)
Frame pairing: for every cam0 image the cam1 image with the nearest stamp. Output file name = cam0 stamp, index = cam0 frame number.
By default the frames before the first full 20-scan LiDAR window are skipped (same frame set as the masks stage).
Output: <out>/NNNN_<cam0 stamp>.jpg, frames.csv (frame, cam0 stamp, cam1 stamp, |dt| ms, t_img_us), unwrap.json (geometry).
"""
from __future__ import annotations

import csv
import json
import os
import time
from multiprocessing import Pool
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from pipeline.common.recording import Run


G = {}


def load_cameras(data_dir):
    y = yaml.safe_load(open(os.path.join(data_dir, "calib", "config", "camera.yaml"), encoding="utf-8-sig"))
    cams = {}
    for k, name in (("camera_0", "cam0"), ("camera_1", "cam1")):
        c = y[k]
        assert c["camera_model"] == "kb4", c["camera_model"]
        fx, fy, cx, cy = c["intrinsic"]
        cams[name] = dict(K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64), D=np.array(c["distortion"], np.float64),
                          w=int(c["image_width"]), h=int(c["image_height"]))
    return cams


def unwrap_maps(cam, W, H, yaw_fov=180.0, pitch_fov=120.0):
    """remap tables (fisheye pixel for every unwrapped pixel) and the valid mask."""
    yaw = np.linspace(-np.radians(yaw_fov) / 2, np.radians(yaw_fov) / 2, W)
    pitch = np.linspace(-np.radians(pitch_fov) / 2, np.radians(pitch_fov) / 2, H)
    yg, pg = np.meshgrid(yaw, pitch)
    x = np.cos(pg) * np.sin(yg)
    y = np.sin(pg)
    z = np.cos(pg) * np.cos(yg)
    pts = np.stack([x, y, z], -1).reshape(-1, 1, 3)
    img, _ = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3), cam["K"], cam["D"])
    mx = img[:, 0, 0].reshape(H, W).astype(np.float32)
    my = img[:, 0, 1].reshape(H, W).astype(np.float32)
    theta = np.degrees(np.arccos(np.clip(z, -1, 1)))
    valid = (mx >= 0) & (mx < cam["w"] - 1) & (my >= 0) & (my < cam["h"] - 1) & (z > 0) & (theta <= 100.0)
    return mx, my, valid.astype(np.uint8)


def _init(cfg):
    run = Run(cfg["data"], cfg["offset"])                 # only the file lists are used here
    cams = load_cameras(cfg["data"])
    maps = {c: unwrap_maps(cams[c], cfg["W"], cfg["H"]) for c in ("cam0", "cam1")}
    G.update(cfg=cfg, run=run, maps=maps)


def process(task):
    fr, j = task
    cfg, run, maps = G["cfg"], G["run"], G["maps"]
    tiles = []
    for cam, idx in (("cam0", fr), ("cam1", j)):
        im = cv2.imread(run.cam_files[cam][idx])
        mx, my, valid = maps[cam]
        t = cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        t[valid == 0] = 0
        tiles.append(t)
    out = np.hstack(tiles)
    stamp = int(run.cam_stamp["cam0"][fr])
    path = os.path.join(cfg["out"], f"{fr:04d}_{stamp}.jpg")
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, cfg["quality"]])
    enc.tofile(path)
    return fr


def run_unwrap(run, out_dir: str, params: dict, frames: str | None = None) -> dict:
    a = SimpleNamespace(**params, frames=frames)
    out = out_dir
    os.makedirs(out, exist_ok=True)
    W, H = a.width, a.width * 3 // 4
    frames = range(len(run.cam_stamp["cam0"]))
    if not a.all_frames and not a.frames:
        # same frames as build_windows.py: skip cam0 frames that have no full n-scan LiDAR window before them
        first = next(fr for fr in frames if int(np.searchsorted(run.lidar_ts, run.image_time(fr), side="right")) - 1 >= a.n_scans - 1)
        frames = range(first, len(run.cam_stamp["cam0"]))
    if a.frames:
        lo, hi = map(int, a.frames.split(":"))
        frames = range(lo, min(hi, len(run.cam_stamp["cam0"])))
    s1 = run.cam_stamp["cam1"]
    tasks, rows, skipped = [], [], 0
    for fr in frames:
        s0 = int(run.cam_stamp["cam0"][fr])
        j = int(np.clip(np.searchsorted(s1, s0), 0, len(s1) - 1))
        if j > 0 and abs(int(s1[j - 1]) - s0) < abs(int(s1[j]) - s0):
            j -= 1
        dt = abs(int(s1[j]) - s0) / 1000.0
        if dt > a.max_dt_ms:
            skipped += 1
            continue
        tasks.append((fr, j))
        rows.append(dict(frame=fr, cam0_stamp_us=s0, cam1_stamp_us=int(s1[j]), cam1_index=j, dt_ms=round(dt, 3), t_img_us=run.image_time(fr),
                         file=f"{fr:04d}_{s0}.jpg"))
    cfg = dict(data=run.data_dir, out=out, W=W, H=H, quality=a.quality)
    json.dump(dict(cfg, yaw_fov_deg=180.0, pitch_fov_deg=120.0, final_w=2 * W, final_h=H, cam1_x_offset=W, image_offset_us=run.image_offset_us,
                   u="(yaw_deg + 90) / 180 * (W - 1) + (W if cam1)", v="(pitch_deg + 60) / 120 * (H - 1)",
                   yaw="atan2(x, z)", pitch="atan2(y, hypot(x, z))", frame="camera frame x right, y down, z forward",
                   theta_limit_deg=100.0, frames=len(tasks), skipped=skipped), open(os.path.join(out, "unwrap.json"), "w"), indent=1, ensure_ascii=False)
    print(f"{len(tasks)} frame pairs ({skipped} skipped, cam1 farther than {a.max_dt_ms} ms); cam0|cam1 {2 * W}x{H} -> {out}")
    dts = [r["dt_ms"] for r in rows]
    print(f"  cam0-cam1 stamp difference: median {np.median(dts):.1f} ms, max {max(dts):.1f} ms")
    t0 = time.time()
    _init(dict(cfg, offset=run.image_offset_us))      # fail fast here (missing calibration...): a raising Pool initializer hangs
    with Pool(a.workers, initializer=_init, initargs=(dict(cfg, offset=run.image_offset_us),)) as pool:
        for i, fr in enumerate(pool.imap_unordered(process, tasks, chunksize=2)):
            if i % 50 == 0:
                print(f"  {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(out, "frames.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"done in {time.time() - t0:.0f}s: {len(tasks)} images in {out}")
    return dict(frames=len(tasks), skipped=skipped, seconds=round(time.time() - t0, 1))
