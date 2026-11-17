"""Operator (device holder) pixel mask per frame, and pixel-based operator flag for the YOLO boxes (proposed_pipeline/
flag_operator_pixels.py, verbatim; main() became run_operator(), values from configs/pipeline.yaml prep.operator; the masks are always
recomputed (the original appended to masks.csv when it skipped existing masks); vis images only with debug figures).

A box is the operator if >= cover (40 %) of its area is covered by the operator mask of that frame. Operator mask = union of
  1. static mask: temporal MAD of the grey image over the run (1/4 scale) < static_mad inside the outer static_zone of each tile,
     connected to the outer border;
  2. segmentation instances (yolo26l-seg, person) touching the outer edge that are large or overlap the static mask;
  3. GrabCut seeded with the edge-flagged operator boxes;
  only components touching the left/right image border are kept; other segmented people are carved out (NNNN_others.png).
Masks: operator_mask/NNNN.png (1/4 scale), masks.csv. detections.csv gains op_cover, operator_pixel, operator; frames.csv recounted.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import subprocess
import time
from types import SimpleNamespace

from pipeline.common.config import FINAL

for _k, _v in {"YOLO_CONFIG_DIR": "ultralytics", "XDG_CACHE_HOME": "xdg", "MPLCONFIGDIR": "mpl", "TORCH_HOME": "torch"}.items():
    _p = os.path.join(FINAL, "output", ".cache", _v)
    os.makedirs(_p, exist_ok=True)
    os.environ.setdefault(_k, _p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402


SCALE = 4   # masks are stored at 1/4 of the image size


def static_mask(files, a, mask_dir):
    """Temporal MAD at 1/4 scale over every 3rd frame -> static operator / mount mask connected to the outer borders."""
    acc = []
    for f in files[::3]:
        acc.append(cv2.imread(f, cv2.IMREAD_REDUCED_GRAYSCALE_4).astype(np.float32))
    A = np.stack(acc)
    med = np.median(A, 0)
    mad = np.median(np.abs(A - med), 0)
    H, W = mad.shape
    zone = np.zeros((H, W), bool)
    z = a.static_zone // SCALE
    zone[:, :z] = True
    zone[:, W - z:] = True
    cand = ((mad < a.static_mad) & zone).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab = cv2.connectedComponents(cand)
    keep = np.zeros((H, W), bool)
    for c in range(1, n):
        m = lab == c
        if m[:, 0].any() or m[:, W - 1].any():      # touches the outer border of the image
            keep |= m
    cv2.imwrite(os.path.join(mask_dir, "static.png"), keep.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(mask_dir, "static_mad.png"), cv2.applyColorMap(np.clip(mad * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET))
    return keep


def grabcut_operator(img, boxes, static, scale, iters):
    """GrabCut on a downscaled image: operator boxes = probable foreground, static mask inside = sure foreground,
    everything outside the boxes = background. -> mask at the 1/SCALE grid."""
    Hf, Wf = img.shape[:2]
    small = cv2.resize(img, (Wf // scale, Hf // scale))
    st = cv2.resize(static.astype(np.uint8), (Wf // scale, Hf // scale), interpolation=cv2.INTER_NEAREST) > 0
    m = np.zeros(st.shape, bool)
    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(np.ceil(x2 / scale)), int(np.ceil(y2 / scale))
        gc = np.full(st.shape, cv2.GC_BGD, np.uint8)
        gc[y1:y2, x1:x2] = cv2.GC_PR_FGD
        gc[st & (gc == cv2.GC_PR_FGD)] = cv2.GC_FGD
        bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
        try:
            cv2.grabCut(small, gc, None, bg, fg, iters, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            gc[y1:y2, x1:x2] = cv2.GC_PR_FGD
        m |= (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    return cv2.resize(m.astype(np.uint8), (Wf // SCALE, Hf // SCALE), interpolation=cv2.INTER_NEAREST) > 0


def border_components(mask):
    """Keep only the connected components that touch the left or right image border (the operator always does)."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8))
    keep = np.zeros_like(mask)
    W = mask.shape[1]
    for c in range(1, n):
        m = lab == c
        if m[:, 0].any() or m[:, W - 1].any():
            keep |= m
    return keep


def load_operator_boxes(path, min_h_frac):
    boxes = {}
    if not path or not os.path.exists(path):
        return boxes
    for r in csv.DictReader(open(path)):
        if int(r["operator_flag"]) and float(r["h"]) >= min_h_frac * 3000:
            boxes.setdefault(int(r["frame"]), []).append(tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))
    return boxes


def build_masks(files, a, mask_dir):
    from ultralytics import YOLO
    model = YOLO(a.seg_model)
    static = static_mask(files, a, mask_dir)
    H, W = static.shape
    half = W // 2
    op_boxes = load_operator_boxes(a.boxes_from, a.min_h)
    stats = []
    t0 = time.time()
    for k, f in enumerate(files):
        fr = int(os.path.basename(f)[:4])
        out_path = os.path.join(mask_dir, f"{fr:04d}.png")
        if os.path.exists(out_path) and not a.redo:
            continue
        img = cv2.imread(f)
        Hf, Wf = img.shape[:2]
        mask = static.copy()
        others = np.zeros((H, W), bool)      # segmented people that do NOT touch the outer edge: never operator
        n_inst = 0
        for cam, x0 in ((0, 0), (1, Wf // 2)):
            tile = img[:, x0:x0 + Wf // 2]
            r = model.predict(tile, imgsz=a.seg_imgsz, conf=0.15, classes=[0], verbose=False, half=True)[0]
            if r.masks is None:
                continue
            ms = r.masks.data.cpu().numpy()
            bx = r.boxes.xyxy.cpu().numpy()
            tw, th = Wf // 2, Hf
            for m, (x1, y1, x2, y2) in zip(ms, bx):
                outer = (cam == 0 and x1 < a.edge) or (cam == 1 and x2 > tw - a.edge)
                m4 = cv2.resize(m.astype(np.uint8), (tw // SCALE, th // SCALE), interpolation=cv2.INTER_NEAREST).astype(bool)
                full = np.zeros((H, W), bool)
                full[:, x0 // SCALE:x0 // SCALE + tw // SCALE] = m4
                if not outer:
                    others |= full
                    continue
                big = (y2 - y1) >= a.min_h * th
                overlap = (full & static).sum() / max(full.sum(), 1)
                if big or overlap >= 0.3:
                    mask |= full
                    n_inst += 1
        boxes = op_boxes.get(fr, [])
        if boxes and a.grabcut_iters > 0:
            mask |= grabcut_operator(img, boxes, static, a.grabcut_scale, a.grabcut_iters)
        mask = border_components(mask) | static
        if others.any():                      # carve other people out of the operator mask (GrabCut / mask bleed)
            others = cv2.dilate(others.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            mask &= ~others
        cv2.imwrite(out_path, mask.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(mask_dir, f"{fr:04d}_others.png"), others.astype(np.uint8) * 255)
        stats.append(dict(frame=fr, n_operator_instances=n_inst, n_operator_boxes=len(boxes), mask_fraction=round(float(mask.mean()), 4)))
        if k % 50 == 0:
            print(f"  mask {k + 1}/{len(files)} frame {fr:04d}: {n_inst} operator instances, mask {mask.mean() * 100:.1f}% ({time.time() - t0:.0f}s)", flush=True)
    if stats:
        with open(os.path.join(mask_dir, "masks.csv"), "a" if not a.redo else "w", newline="") as fh:
            wr = csv.DictWriter(fh, list(stats[0].keys()))
            if fh.tell() == 0:
                wr.writeheader()
            wr.writerows(stats)


def apply(det_dir, files, a, mask_dir):
    rows = list(csv.DictReader(open(os.path.join(det_dir, "detections.csv"))))
    if not rows:                                     # no YOLO box at all: header-only tables with the operator columns
        head = next(csv.reader(open(os.path.join(det_dir, "detections.csv"))), [])
        head = [h for h in head if h not in ("op_cover", "operator_pixel", "operator")] + ["op_cover", "operator_pixel", "operator"]
        with open(os.path.join(det_dir, "detections.csv"), "w", newline="") as fh:
            csv.writer(fh).writerow(head)
        with open(os.path.join(det_dir, "frames.csv"), "w", newline="") as fh:
            csv.writer(fh).writerow(["frame", "cam0_stamp_us", "n_det", "n_operator", "n_operator_edge", "n_operator_pixel_only", "n_person"])
        print(f"{os.path.basename(det_dir.rstrip('/'))}: 0 boxes")
        return
    by = {}
    for r in rows:
        by.setdefault(int(r["frame"]), []).append(r)
    masks = {}
    for fr, rs in by.items():
        m = cv2.imread(os.path.join(mask_dir, f"{fr:04d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        masks[fr] = m > 0
        for r in rs:
            x1, y1, x2, y2 = (int(float(r[k]) / SCALE) for k in ("x1", "y1", "x2", "y2"))
            sub = masks[fr][y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
            cover = float(sub.mean()) if sub.size else 0.0
            r["op_cover"] = round(cover, 3)
            r["operator_pixel"] = int(cover >= a.cover)
            r["operator"] = int(int(r["operator_flag"]) or r["operator_pixel"])
    for r in rows:                                   # columns of the removed reflection experiment, if present
        for k in ("top_elev_deg", "bot_elev_deg", "reflection"):
            r.pop(k, None)
    with open(os.path.join(det_dir, "detections.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    frows = []
    for fr in sorted(by):
        rs = by[fr]
        n_op = sum(int(r.get("operator", r["operator_flag"])) for r in rs)
        frows.append(dict(frame=fr, cam0_stamp_us=rs[0]["cam0_stamp_us"], n_det=len(rs), n_operator=n_op, n_operator_edge=sum(int(r["operator_flag"]) for r in rs),
                          n_operator_pixel_only=sum(1 for r in rs if int(r.get("operator_pixel", 0)) and not int(r["operator_flag"])), n_person=len(rs) - n_op))
    with open(os.path.join(det_dir, "frames.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, list(frows[0].keys()))
        wr.writeheader()
        wr.writerows(frows)
    tag = os.path.basename(det_dir.rstrip("/"))
    for f in (files if a.figure else []):       # redraw (debug figure)
        fr = int(os.path.basename(f)[:4])
        img = cv2.imread(f, cv2.IMREAD_REDUCED_COLOR_4)
        if fr in masks:
            ov = img.copy()
            ov[masks[fr]] = (ov[masks[fr]] * 0.6 + np.array([60, 60, 200]) * 0.4).astype(np.uint8)
            img = ov
        for r in by.get(fr, []):
            op = int(r.get("operator", r["operator_flag"]))
            conf = float(r["conf"])
            col = (128, 128, 128) if op else (0, 255, 0) if conf >= 0.4 else (0, 200, 255)
            p1, p2 = (int(float(r["x1"]) / SCALE), int(float(r["y1"]) / SCALE)), (int(float(r["x2"]) / SCALE), int(float(r["y2"]) / SCALE))
            cv2.rectangle(img, p1, p2, col, 2)
            cv2.putText(img, f"{conf:.2f} {float(r['dist_m']):.1f}m" + (f" op{float(r.get('op_cover', 0)):.1f}" if op else ""), (p1[0], max(p1[1] - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.putText(img, f"frame {fr:04d}  {tag}: green conf>=0.4, yellow 0.15-0.4, grey = operator (edge rule or pixel mask, red tint)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(det_dir, "vis", f"{fr:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    n_px = sum(f["n_operator_pixel_only"] for f in frows)
    print(f"{tag}: {len(rows)} boxes, operator by edge rule {sum(f['n_operator_edge'] for f in frows)}, by pixel mask only {n_px}, "
          f"person boxes left {sum(f['n_person'] for f in frows)} ({sum(f['n_person'] for f in frows) / len(frows):.2f} per frame)")
    if a.video:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "4", "-pattern_type", "glob", "-i", os.path.join(det_dir, "vis", "*.jpg"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", os.path.join(det_dir, "vis.mp4")], check=False)


def run_operator(images_dir: str, det_dir: str, mask_dir: str, seg_model: str, params: dict, figure: bool = False) -> dict:
    a = SimpleNamespace(**params, images=images_dir, det=[det_dir], mask_dir=mask_dir, seg_model=seg_model,
                        boxes_from=os.path.join(det_dir, "detections.csv"), redo=True, video=figure, figure=figure)
    t0 = time.time()
    os.makedirs(mask_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.images, "[0-9]" * 4 + "_*.jpg")))
    json.dump(dict(vars(a), mask_dir=mask_dir, n_images=len(files)), open(os.path.join(mask_dir, "args.json"), "w"), indent=1, ensure_ascii=False)
    build_masks(files, a, mask_dir)
    for d in a.det:
        if os.path.exists(os.path.join(d, "detections.csv")):
            apply(d, files, a, mask_dir)
    return dict(images=len(files), seconds=round(time.time() - t0, 1))
