"""Figures of the selective-accumulation detector (cv2 only).

  prep      : preprocessing stages (raw -> FOV -> 10 m -> ground/ceiling -> inside 2D boxes -> selected pieces), BEV + image
  scans     : one person: selected points per candidate scan (bar chart by status) and the points coloured by scan time
  accum     : selected pieces of all persons coloured by scan time (BEV + side view + image), main-component boxes
  boxes_bev / boxes_img : final boxes (thick = accumulated box_acc, thin = nearest-burst box_burst), heading arrow, 2D boxes
BEV: leveled device frame at the image time, x forward = up, y left = left, 40 px/m, 10 m radius.
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np

from pipeline.common.angbox import PAL_TRACK, TILE_H
from pipeline.common.box_fit import EDGES, corners, heading_arrow

SIZE, PPM = 800, 40.0
TURBO = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]


def key_colour(key: str):
    tid = int(key[1:]) if key.startswith("t") else 300 + sum(ord(ch) * (i + 1) for i, ch in enumerate(key)) % 500   # deterministic
    return tuple(int(c) for c in PAL_TRACK[tid % 1024])


def time_colour(dt_s, span=1.05):
    a = np.clip((np.asarray(dt_s, float) + span) / (2 * span), 0, 1)       # -1 s .. +1 s -> blue .. red
    return TURBO[(a * 255).astype(int)]


def canvas(title, size=SIZE):
    img = np.full((size, size, 3), 18, np.uint8)
    c = size // 2
    for r in (5, 10):
        cv2.circle(img, (c, c), int(r * PPM * size / SIZE), (0, 110, 130), 1, cv2.LINE_AA)
    cv2.circle(img, (c, c), 4, (255, 255, 255), -1)
    cv2.putText(img, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return img


def bev_px(xyz_lv, size=SIZE):
    s = PPM * size / SIZE
    u = (size / 2 - xyz_lv[:, 1] * s).astype(int)
    v = (size / 2 - xyz_lv[:, 0] * s).astype(int)
    ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    return u, v, ok


def put_points(img, xyz_lv, col, size=SIZE):
    u, v, ok = bev_px(xyz_lv, size)
    if np.ndim(col) == 1 and len(col) == 3 and not isinstance(col, np.ndarray):
        img[v[ok], u[ok]] = col
    else:
        img[v[ok], u[ok]] = np.asarray(col)[ok]


def side_canvas(title, w=SIZE, h=260):
    img = np.full((h, w, 3), 18, np.uint8)
    for z, c in ((0.0, (0, 200, 0)), (2.5, (0, 0, 255))):
        y = int(230 - z * 50)
        cv2.line(img, (0, y), (w - 1, y), c, 1)
    cv2.putText(img, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return img


def put_side(img, xyz_lv, col, w=SIZE):
    u = (w / 2 + xyz_lv[:, 0] * PPM).astype(int)      # x forward -> right
    v = (230 - xyz_lv[:, 2] * 50).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < img.shape[0])
    if np.ndim(col) == 1 and len(col) == 3 and not isinstance(col, np.ndarray):
        img[v[ok], u[ok]] = col
    else:
        img[v[ok], u[ok]] = np.asarray(col)[ok]


def box_corners_lv(b):
    return corners(b["cx"], b["cy"], b["cz"], b["l"], b["w"], b["h"], b["yaw"])


def draw_box_bev(img, b, col, thick, label=None, size=SIZE):
    if not np.isfinite(b["cx"]):
        return
    C = box_corners_lv(b)
    u, v, _ = bev_px(C[:4], size)
    cv2.polylines(img, [np.stack([u, v], 1).astype(np.int32)], True, col, thick, cv2.LINE_AA)
    s0, s1 = heading_arrow(C)
    (u0, v0, _), (u1, v1, _) = bev_px(np.array([s0]), size), bev_px(np.array([s1]), size)
    cv2.arrowedLine(img, (int(u0[0]), int(v0[0])), (int(u1[0]), int(v1[0])), (0, 220, 255), thick, cv2.LINE_AA, tipLength=0.3)
    if label:
        cv2.putText(img, label, (int(u.max()) + 4, int(v.min()) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)


def project_lv(geom, plane, xyz_lv):
    """Leveled points -> device frame at t_c -> side-by-side pixels (1/4 scale) with validity."""
    dev = (np.asarray(xyz_lv, float) + np.array([0, 0, plane["ground_z"]])) @ plane["R_level"]      # R_level^T (x + gz)
    cam, u, v, z = geom.project(dev)
    ok = (z > 0) & (v >= 0) & (v < TILE_H)
    return cam, (u / 4).astype(int), (v / 4).astype(int), ok


def draw_box_img(img, geom, plane, b, col, thick, label=None):
    if not np.isfinite(b["cx"]):
        return
    C = box_corners_lv(b)
    cam, u, v, ok = project_lv(geom, plane, C)
    for i, j in EDGES:
        if ok[i] and ok[j] and cam[i] == cam[j]:
            cv2.line(img, (u[i], v[i]), (u[j], v[j]), col, thick, cv2.LINE_AA)
    if label and ok[4:].any():
        k = 4 + int(np.argmin(np.where(ok[4:], v[4:], 10 ** 6)))
        cv2.putText(img, label, (u[k], max(v[k] - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)


def draw_2d_boxes(img, boxes, keys_col):
    for key, row in boxes:
        col = keys_col.get(key, (160, 160, 160))
        cv2.rectangle(img, (int(row["x1"] / 4), int(row["y1"] / 4)), (int(row["x2"] / 4), int(row["y2"] / 4)), col, 1)
        cv2.putText(img, key, (int(row["x1"] / 4), max(int(row["y1"] / 4) - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)


def put_points_img(img, geom, plane, xyz_lv, col, step=1):
    cam, u, v, ok = project_lv(geom, plane, xyz_lv[::step])
    ok &= (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
    if np.ndim(col) == 1 and len(col) == 3 and not isinstance(col, np.ndarray):
        img[v[ok], u[ok]] = col
    else:
        img[v[ok], u[ok]] = np.asarray(col)[::step][ok]


# ----------------------------------------------------------------------------------------------------------------------
def draw_prep(path, F, persons):
    """F: frame data (see run_detection.process_frame): raw_lv, all_lv, fov, keep, inbox (key -> bool mask over all_lv),
    sel (key -> bool mask), keys_col, image, boxes2d, geom, plane, frame, n_* counts."""
    fr = F["frame"]
    grey = (90, 90, 90)
    p1 = canvas(f"frame {fr:04d} 1 raw: {F['n_raw']} pts ({F.get('n_scans', 41)} scans, device frame at t_img; every 2nd drawn)")
    put_points(p1, F["raw_lv"], (170, 170, 170))
    p2 = canvas(f"2 after mount/operator masks + camera FOV: {int(F['fov'].sum())} pts")
    put_points(p2, F["all_lv"], grey); put_points(p2, F["all_lv"][F["fov"]], (170, 170, 170))
    p3 = canvas(f"3 sensor range <= 10 m (already applied before 2): {len(F['all_lv'])} pts")
    put_points(p3, F["all_lv"][F["fov"]], (170, 170, 170))
    gk = F["fov"] & F["keep"]
    p4 = canvas(f"4 ground/ceiling removed (h {F['plane']['source']}, ceiling {F['plane']['ceiling_h']:.2f} m): {int(gk.sum())} pts")
    put_points(p4, F["all_lv"][F["fov"] & ~F["keep"]], (50, 50, 50)); put_points(p4, F["all_lv"][gk], (170, 170, 170))
    p5 = canvas("5 inside the 2D person boxes (interpolated to each scan time)")
    put_points(p5, F["all_lv"][gk], (60, 60, 60))
    for key, m in F["inbox"].items():
        put_points(p5, F["all_lv"][m], F["keys_col"][key])
    p6 = canvas("6 selected pieces (per-scan clustering + distance score); dark = in box, not selected")
    for key, m in F["inbox"].items():
        put_points(p6, F["all_lv"][m], (70, 70, 70))
    for key, m in F["sel"].items():
        put_points(p6, F["all_lv"][m], F["keys_col"][key])
    top = np.hstack([p1, p2, p3]); bot = np.hstack([p4, p5, p6])
    img = F["image"].copy()
    put_points_img(img, F["geom"], F["plane"], F["all_lv"][gk], (90, 90, 90), step=3)
    for key, m in F["sel"].items():
        put_points_img(img, F["geom"], F["plane"], F["all_lv"][m], F["keys_col"][key])
    draw_2d_boxes(img, F["boxes2d"], F["keys_col"])
    cv2.putText(img, f"frame {fr:04d}: points after stage 4 (grey) and selected pieces (colour = 2D id), 2D boxes", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    img = cv2.resize(img, (top.shape[1], int(750 * top.shape[1] / 2000)))
    cv2.imwrite(path, np.vstack([top, bot, img]))


def draw_scans(path, F, key, cands, person):
    """One person: bar chart of selected points per candidate scan (colour = status) + BEV/side of the selected points by time."""
    col = F["keys_col"][key]
    st_col = dict(selected=(80, 220, 80), ambiguous=(0, 200, 255), ambiguous_multi=(0, 140, 255), taken_by_other=(0, 80, 255),
                  none=(110, 110, 110), no_box=(60, 60, 60), rejected_consistency=(0, 0, 230), rejected_range=(0, 0, 150))
    W, H = SIZE * 2, 300
    bar = np.full((H, W, 3), 18, np.uint8)
    n = max(len(cands), 1)
    bw = max(int((W - 40) / n), 2)
    vmax = max([max(c["n_in_box"], c["n_sel"]) for c in cands] + [1])
    for k, c in enumerate(cands):
        x0 = 20 + k * bw
        hb = int((H - 60) * c["n_in_box"] / vmax)
        cv2.rectangle(bar, (x0, H - 30 - hb), (x0 + bw - 2, H - 30), (70, 70, 70), -1)
        hs = int((H - 60) * (c.get("n_sel_rejected", 0) or c["n_sel"]) / vmax)
        cv2.rectangle(bar, (x0, H - 30 - hs), (x0 + bw - 2, H - 30), st_col.get(c["status"], (200, 200, 200)), -1)
        if abs(c["dt_cam_s"]) < 0.03:
            cv2.line(bar, (x0 + bw // 2, 20), (x0 + bw // 2, H - 28), (255, 255, 255), 1)
        if k % 5 == 0:
            cv2.putText(bar, f"{c['dt_cam_s']:+.2f}", (x0, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    cv2.putText(bar, f"frame {F['frame']:04d} {key}: points inside the 2D box per candidate scan (grey) and selected piece (green); "
                     f"orange = ambiguous, red = rejected by consistency / burst range, dark = no box / none; white line = image time", (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.putText(bar, f"max {vmax} pts; x = scan time - image time (s)", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    bev = canvas(f"{key}: selected points coloured by scan time (blue -1 s .. red +1 s)")
    side = side_canvas("side view (x forward vs height), same colours")
    if person is not None:
        P = person["xyz_lv"]
        tc = time_colour((person["t_pts"] - person["t_c"]) / 1e6)
        put_points(bev, P, tc); put_side(side, P, tc)
        draw_box_bev(bev, person["box_acc"], col, 2, f"{key} acc {person['box_acc']['l']:.2f}x{person['box_acc']['w']:.2f}x{person['box_acc']['h']:.2f}")
        draw_box_bev(bev, person["box_burst"], col, 1)
        # zoom panel around the person
        c = person["box_acc"]
        if np.isfinite(c["cx"]):
            z = canvas("zoom 3 x 3 m around box_acc", size=SIZE)
            Pz = (P - np.array([c["cx"], c["cy"], 0])) * (10 / 1.5)
            put_points(z, Pz, tc)
            bz = dict(c, cx=0.0, cy=0.0, l=c["l"] * 10 / 1.5, w=c["w"] * 10 / 1.5)
            draw_box_bev(z, bz, col, 2)
            bb = person["box_burst"]
            if np.isfinite(bb["cx"]):
                draw_box_bev(z, dict(bb, cx=(bb["cx"] - c["cx"]) * 10 / 1.5, cy=(bb["cy"] - c["cy"]) * 10 / 1.5, l=bb["l"] * 10 / 1.5, w=bb["w"] * 10 / 1.5), col, 1)
            side = np.vstack([side, cv2.resize(z, (SIZE, SIZE - 260))])
        else:
            side = np.vstack([side, np.full((SIZE - 260, SIZE, 3), 18, np.uint8)])
    else:
        side = np.vstack([side, np.full((SIZE - 260, SIZE, 3), 18, np.uint8)])
    cv2.imwrite(path, np.vstack([bar, np.hstack([bev, side])]))


def draw_accum(path, F, persons):
    bev = canvas(f"frame {F['frame']:04d} selected pieces by scan time (blue = -1 s, red = +1 s); thick box = accumulated, thin = nearest burst")
    side = side_canvas("side view")
    put_points(bev, F["all_lv"][F["fov"] & F["keep"]], (45, 45, 45))
    img = F["image"].copy()
    for q in persons:
        tc = time_colour((q["t_pts"] - q["t_c"]) / 1e6)
        put_points(bev, q["xyz_lv"], tc); put_side(side, q["xyz_lv"], tc)
        put_points_img(img, F["geom"], F["plane"], q["xyz_lv"], tc)
        col = F["keys_col"][q["key"]]
        draw_box_bev(bev, q["box_acc"], col, 2, f"{q['key']} main {q['n_scans']}sc {q['n_pts']}pts {q['span_s']:.2f}s; all {q['n_scans_all']}sc {q['n_pts_all']}pts {q['n_components']}comp")
        draw_box_bev(bev, q["box_burst"], col, 1)
        draw_box_img(img, F["geom"], F["plane"], q["box_acc"], col, 2, q["key"])
    draw_2d_boxes(img, F["boxes2d"], F["keys_col"])
    img = cv2.resize(img, (SIZE * 2, int(750 * SIZE * 2 / 2000)))
    cv2.imwrite(path, np.vstack([np.hstack([bev, np.vstack([side, np.full((SIZE - 260, SIZE, 3), 18, np.uint8)])]), img]))


def draw_boxes(path_img, path_bev, F, persons):
    bev = canvas(f"frame {F['frame']:04d} final boxes: thick = box_acc (accumulated), thin = box_burst (nearest burst), arrow = yaw")
    put_points(bev, F["all_lv"][F["fov"] & F["keep"]], (45, 45, 45))
    img = F["image"].copy()
    for q in persons:
        col = F["keys_col"][q["key"]]
        put_points(bev, q["xyz_lv"][q["in_main"]], col)
        put_points_img(img, F["geom"], F["plane"], q["xyz_lv"][q["in_main"]], col)
        fl = "".join(k[0] for k, v in q["flags"].items() if v)
        lab = f"{q['key']} {q['box_acc']['l']:.2f}x{q['box_acc']['w']:.2f}x{q['box_acc']['h']:.2f} {q['yaw_source']}{' ' + fl if fl else ''}{'' if q['valid'] else ' INVALID'}"
        draw_box_bev(bev, q["box_acc"], col, 2, lab)
        draw_box_bev(bev, q["box_burst"], col, 1)
        draw_box_img(img, F["geom"], F["plane"], q["box_acc"], col, 2, q["key"])
        draw_box_img(img, F["geom"], F["plane"], q["box_burst"], col, 1)
    draw_2d_boxes(img, F["boxes2d"], F["keys_col"])
    cv2.putText(img, f"frame {F['frame']:04d}: 2D boxes, accumulated person points, 3D boxes (thick acc, thin burst)", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(bev, "flags: l lower_only, u upper_only, s single_scan, f fragmented, s smeared", (6, SIZE - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    cv2.imwrite(path_bev, bev)
    cv2.imwrite(path_img, img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def make_video(folder, out_mp4, fps=3):
    import subprocess
    files = sorted(f for f in os.listdir(folder) if f.endswith((".png", ".jpg")))
    if not files:
        return
    ext = files[0][-4:]
    lst = os.path.join(folder, "_list.txt")
    with open(lst, "w") as f:
        for name in files:
            f.write(f"file '{name}'\nduration {1 / fps}\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", out_mp4], check=False)
    os.remove(lst)
