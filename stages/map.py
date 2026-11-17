"""Stage map: the final static map in the pose-corrected frame -> 3_map/ (3d_detection/scripts/build_map.py with --correction,
verbatim; main() became run(); paths of this recording; settings = configs/pipeline.yaml map / tracking / detection).

Map domain = points v3 could see (camera FOV, 0.25-10 m, mount / operator masks, a v3 camera frame nearby). Every tracked person is
removed by a cylinder around its smoothed track at the point's own time (legacy frame: same removal decision); voxels with non-person
hits spanning > 3 s are protected as structure; every voxel step and output uses the corrected coordinates (pose_refine).
Outputs: map_static.ply (final map), map_before.ply, map_removed.ply, map_protected.ply, map_residual.ply, removed_by_track.csv,
top_before.png, top_static.png, top_overlay.png, top_residual.png, summary.json. No ground truth: every number is a description.
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

from pipeline.common.cam_geom import CamGeom
from pipeline.common.io import read_ply_fields, write_csv
from pipeline.common.pose import PoseInterp
from pipeline.common.recording import Run, write_ply
from pipeline.detect.clean import scan_fov_mask
from pipeline.detect.ground import to_level
from pipeline.detect.scans import ScanStore, load_masks
from pipeline.mapping.pose_refine import PoseCorrection
from pipeline.mapping.static_map import (VoxelSpan, VoxelSum, build_tubes, in_tubes, obs_from_csv, protect_mask, read_csv,
                                         isolated_from, scan_in_frames, structure_keys, tube_distance, voxel_keys)
from pipeline.prep.clock import read_clock
from pipeline.track.track_vis import plane_lookup
from stages._base import stage_main


TRACK_FIELDS = ["track_id", "n_obs", "t_first_us", "t_last_us", "n_spans", "span_s", "speed_med", "speed_max", "n_removed_pts",
                "n_protected_pts", "n_removed_vox"]


def save_map(path, xyz, cols, n, extra=None):
    f = dict(x=xyz[:, 0].astype(np.float32), y=xyz[:, 1].astype(np.float32), z=xyz[:, 2].astype(np.float32),
             h=np.asarray(cols.get("h", np.zeros(len(xyz))), np.float32), n=np.asarray(n, np.int32))
    for k, v in (extra or {}).items():
        f[k] = np.asarray(v, np.int32)
    write_ply(path, f)


def top_view(layers, extent, title, px=0.05, h_lo=0.1, h_hi=2.0, max_px=4000):
    """layers: [(xyz, h, colour or None = height colour)] drawn in order; only heights in (h_lo, h_hi]. extent = (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = extent
    px = max(px, max(x1 - x0, y1 - y0) / max_px)
    W, H = int((x1 - x0) / px) + 1, int((y1 - y0) / px) + 1
    img = np.full((H + 40, W, 3), 18, np.uint8)
    for xyz, h, col in layers:
        m = (h > h_lo) & (h <= h_hi)
        if not m.any():
            continue
        u = ((xyz[m, 0] - x0) / px).astype(int)
        v = 40 + ((y1 - xyz[m, 1]) / px).astype(int)
        ok = (u >= 0) & (u < W) & (v >= 40) & (v < H + 40)
        if col is None:
            o = np.argsort(h[m][ok])
            c = cv2.applyColorMap(((h[m][ok][o] - h_lo) / (h_hi - h_lo) * 255).clip(0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
            img[v[ok][o], u[ok][o]] = c
        else:
            img[v[ok], u[ok]] = col
    cv2.putText(img, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    bar = int(5.0 / px)
    cv2.line(img, (W - bar - 12, H + 30), (W - 12, H + 30), (255, 255, 255), 2)
    cv2.putText(img, "5 m", (W - bar - 12, H + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def run(P, cfg, scans=None, **kw):
    """scans = "START:STOP" scan indices (default all). The pose correction of the pose stage is always applied."""
    a = type("A", (), dict(dataset=P.name, scans=scans, correction=os.path.join(P.pose, "pose_correction.csv")))
    t_start = time.time()
    cfg_det, tcfg, mcfg = cfg["detection"], cfg["tracking"], cfg["map"]
    det = P.detection
    track = P.tracking
    out = P.mapping
    corr = PoseCorrection.from_csv(a.correction) if a.correction else None
    os.makedirs(out, exist_ok=True)
    run = Run(P.data, read_clock(P.clock))
    pose = PoseInterp(run, os.path.join(P.odometry, "odometry.csv"))
    geom = CamGeom(P.data)
    store = ScanStore(run, pose, load_masks(P.masks), cfg_det, cache=2)     # v3 scan rules: 0.25-10 m, mount + operator masks
    fr_all = [r for r in read_csv(os.path.join(det, "frames.csv")) if r.get("t_img_us")]
    if not any(r.get("R00") for r in fr_all):
        raise RuntimeError("no floor plane: v3 fits the floor only in camera frames with a 2D person box and this recording has none, "
                           "so the height above the floor (map domain and person cylinders) cannot be computed")
    f_ids = np.array([int(r["frame"]) for r in fr_all])
    f_t = np.array([int(r["t_img_us"]) for r in fr_all], np.int64)
    plane_of = plane_lookup(fr_all, pose)
    obs = obs_from_csv(read_csv(os.path.join(track, "observations.csv")))
    tubes = build_tubes(obs, tcfg, mcfg)
    lo, hi = map(int, a.scans.split(":")) if a.scans else (0, run.n_scans())
    hi = min(hi, run.n_scans())

    vox, pv = mcfg["out"]["voxel"], mcfg["protect"]["voxel"]
    before, static, span = VoxelSum(vox), VoxelSum(vox), VoxelSpan()
    cand = dict(X=[], t=[], tid=[], h=[])
    res = dict(X=[], h=[], d=[])                 # kept points in the residue height band, with the distance to the nearest active cylinder
    cnt = dict(raw=0, near=0, self_mask=0, operator=0, far=0, in_range=0, no_frame=0, out_of_fov=0, domain=0)
    n_scan_used = 0
    for i in range(lo, hi):
        s = store.get(i)
        cnt["raw"] += s["n_raw"]
        cnt["near"] += s["n_near"]
        cnt["self_mask"] += s["n_self"]
        cnt["operator"] += s["n_oper"]
        cnt["far"] += s["n_far"]
        cnt["in_range"] += s["n_range"]
        t_s = int(np.median(s["t_point_us"])) if len(s["t_point_us"]) else s["t_scan_us"] + 25_000
        if not scan_in_frames(np.array([t_s]), f_t, mcfg["domain"]["frame_half_s"], mcfg["domain"]["frame_gap_max_s"])[0]:
            cnt["no_frame"] += s["n_range"]
            continue
        fov = scan_fov_mask(s, pose, geom, cfg_det)
        cnt["out_of_fov"] += int((~fov).sum())
        X, t = s["xyz_odom"][fov], s["t_point_us"][fov]
        if len(X) == 0:
            continue
        n_scan_used += 1
        cnt["domain"] += len(X)
        pl = plane_of(int(f_ids[np.abs(f_t - t_s).argmin()]))
        h = to_level((X - pl["p_c"]) @ pl["R_c"], pl)[:, 2]
        tid = in_tubes(tubes, X, t, pl, mcfg)                        # legacy frame: tracks and planes live there
        non = tid < 0
        Xc = corr.apply(X, t, pose.odo.at(t)) if corr is not None else X
        before.add(Xc, dict(h=h))
        static.add(Xc[non], dict(h=h[non]))
        span.add(voxel_keys(Xc[non], pv), t[non])
        for k, v in (("X", Xc[~non]), ("t", t[~non]), ("tid", tid[~non]), ("h", h[~non])):
            cand[k].append(v)
        band = non & (h > mcfg["vis"]["h_lo"]) & (h <= mcfg["vis"]["h_hi"])
        res["X"].append(Xc[band].astype(np.float32))
        res["h"].append(h[band].astype(np.float32))
        res["d"].append(tube_distance(tubes, X[band], t[band], pl).astype(np.float32))
        if (i - lo) % 500 == 0:
            print(f"scan {i}/{hi}: domain {cnt['domain']} candidates {sum(len(v) for v in cand['t'])}", flush=True)

    C = {k: (np.concatenate(v) if v else np.zeros((0, 3) if k == "X" else 0)) for k, v in cand.items()}
    prot = protect_mask(voxel_keys(C["X"], pv), span, mcfg["protect"]["span_s"]) if len(C["t"]) else np.zeros(0, bool)
    static.add(C["X"][prot], dict(h=C["h"][prot]))

    def per_track(mask):
        xs, hs, ns, ts = [], [], [], []
        for tid in np.unique(C["tid"][mask]):
            m = mask & (C["tid"] == tid)
            v = VoxelSum(vox)
            v.add(C["X"][m], dict(h=C["h"][m]))
            x, c, n = v.result()
            xs.append(x), hs.append(c["h"]), ns.append(n), ts.append(np.full(len(n), tid))
        if not xs:
            return np.zeros((0, 3)), dict(h=np.zeros(0)), np.zeros(0, int), np.zeros(0, int)
        return np.concatenate(xs), dict(h=np.concatenate(hs)), np.concatenate(ns), np.concatenate(ts)

    bx, bc, bn = before.result()
    sx, sc, sn = static.result()
    rx, rc, rn, rt = per_track(~prot)
    px_, pc_, pn_, pt_ = per_track(prot)
    save_map(os.path.join(out, "map_before.ply"), bx, bc, bn)
    save_map(os.path.join(out, "map_static.ply"), sx, sc, sn)
    save_map(os.path.join(out, "map_removed.ply"), rx, rc, rn, dict(track_id=rt))
    save_map(os.path.join(out, "map_protected.ply"), px_, pc_, pn_, dict(track_id=pt_))

    # ---- residue diagnosis: kept points of voxels seen only briefly (span of the kept points <= transient_s)
    rc_ = mcfg["residual"]
    RX = np.concatenate(res["X"]).astype(float) if res["X"] else np.zeros((0, 3))
    RH = np.concatenate(res["h"]).astype(float) if res["h"] else np.zeros(0)
    RD = np.concatenate(res["d"]).astype(float) if res["d"] else np.zeros(0)
    pkeys = voxel_keys(C["X"][prot], pv) if prot.any() else np.zeros(0, np.int64)
    rk = voxel_keys(RX, pv)
    trans0 = (span.span_s(rk) <= rc_["transient_s"]) & ~np.isin(rk, pkeys)    # protected voxels are structure by definition
    trans = trans0 & isolated_from(rk, np.r_[structure_keys(span, mcfg["protect"]["span_s"]), pkeys])   # not touching long-observed structure
    cause = np.where(RD[trans] <= rc_["near_m"], 1, 2)
    qx, qc, qn, qt = [], [], [], []
    for c_ in (1, 2):
        v = VoxelSum(vox)
        v.add(RX[trans][cause == c_], dict(h=RH[trans][cause == c_]))
        x, c, n = v.result()
        qx.append(x), qc.append(c.get("h", np.zeros(0))), qn.append(n), qt.append(np.full(len(n), c_))   # no point of this cause: empty
    qx, qn, qt = np.concatenate(qx), np.concatenate(qn), np.concatenate(qt)
    qh = dict(h=np.concatenate(qc))
    save_map(os.path.join(out, "map_residual.ply"), qx, qh, qn, dict(cause=qt))
    residual = dict(transient_points=int(trans0.sum()), isolated_points=int(trans.sum()), near_track=int((cause == 1).sum()), no_track=int((cause == 2).sum()),
                    band_points=int(len(RX)), note="diagnosis only; nothing removed")

    # ---- per-track table
    trk = {int(r["track_id"]): r for r in read_csv(os.path.join(track, "tracks.csv"))}
    rows = []
    for tb in tubes:
        o = obs[tb.track_id]
        m = C["tid"] == tb.track_id
        rows.append(dict(track_id=tb.track_id, n_obs=len(o), t_first_us=min(x.t_us for x in o), t_last_us=max(x.t_us for x in o),
                         n_spans=len(tb.spans), span_s=round(sum(b - a for a, b in tb.spans) / 1e6, 2),
                         speed_med=trk.get(tb.track_id, {}).get("speed_med", ""), speed_max=trk.get(tb.track_id, {}).get("speed_max", ""),
                         n_removed_pts=int((m & ~prot).sum()), n_protected_pts=int((m & prot).sum()), n_removed_vox=int((rt == tb.track_id).sum())))
    write_csv(os.path.join(out, "removed_by_track.csv"), rows, TRACK_FIELDS)

    # ---- self-check: v3 box points (main component) inside the cylinder of their own 3D track at their own time
    t_lo, t_hi = int(run.lidar_ts[lo]), int(run.lidar_ts[hi - 1]) + 50_000
    tid_of = {(int(r["frame"]), r["key"]): int(r["track_id"]) for r in read_csv(os.path.join(track, "rows.csv"))}
    n_main = n_in = n_own = 0
    for r in read_csv(os.path.join(det, "persons.csv")):
        f = int(r["frame"])
        t_img = int(f_t[f_ids == f][0]) if (f_ids == f).any() else None
        path = os.path.join(det, "clouds", f"{f:04d}_{r['key']}.ply")
        if t_img is None or not (t_lo <= t_img <= t_hi) or not os.path.exists(path):
            continue
        d = read_ply_fields(path)
        m = np.asarray(d["in_main"]) == 1
        pl = plane_of(f)
        lv = np.stack([d["x"], d["y"], d["z"]], 1).astype(float)[m]
        Xo = ((lv + [0, 0, pl["ground_z"]]) @ pl["R_level"]) @ pl["R_c"].T + pl["p_c"]
        tp = t_img + (np.asarray(d["time_rel_s"], float)[m] * 1e6).astype(np.int64)
        got = in_tubes(tubes, Xo, tp, pl, mcfg)
        n_main += len(got)
        n_in += int((got >= 0).sum())
        n_own += int((got == tid_of.get((f, r["key"]), -2)).sum())
    box_check = dict(n_main_points=n_main, frac_inside=round(n_in / n_main, 4) if n_main else 0.0,
                     frac_inside_own_track=round(n_own / n_main, 4) if n_main else 0.0)

    # ---- top views
    vc = mcfg["vis"]
    if len(bx):
        e = (bx[:, 0].min() - 1, bx[:, 1].min() - 1, bx[:, 0].max() + 1, bx[:, 1].max() + 1)
        kw = dict(h_lo=vc["h_lo"], h_hi=vc["h_hi"])
        rng = f"heights {vc['h_lo']}-{vc['h_hi']} m"
        name = a.dataset.split("_")[-1]                        # ASCII part of the recording name (cv2 cannot draw Hangul)
        cv2.imwrite(os.path.join(out, "top_before.png"), top_view([(bx, bc["h"], None)], e, f"{name} map before removal ({rng}, colour = height)", **kw))
        cv2.imwrite(os.path.join(out, "top_static.png"), top_view([(sx, sc["h"], None)], e, f"{name} final static map ({rng}, colour = height)", **kw))
        cv2.imwrite(os.path.join(out, "top_overlay.png"), top_view([(sx, sc["h"], (150, 150, 150)), (rx, rc["h"], (40, 40, 235)), (px_, pc_["h"], (0, 220, 255))], e,
                                                                   f"{name} grey = final map, red = removed persons, yellow = protected", **kw))
        cv2.imwrite(os.path.join(out, "top_residual.png"), top_view([(sx, sc["h"], (110, 110, 110)), (qx[qt == 2], qh["h"][qt == 2], (255, 200, 0)),
                                                                    (qx[qt == 1], qh["h"][qt == 1], (255, 0, 255))], e,
                                                                   f"{name} kept points seen <= 1 s, apart from structure: magenta = near a tracked person, cyan = none", **kw))

    n_rem, n_prot = int((~prot).sum()), int(prot.sum())
    summary = dict(dataset=a.dataset, det=det, track=track, correction=a.correction, scans=[lo, hi], n_scans_used=n_scan_used, n_tracks=len(tubes), config=mcfg,
                   points=dict(cnt, static=cnt["domain"] - n_rem, removed=n_rem, protected=n_prot),
                   voxels=dict(before=int(len(bn)), static=int(len(sn)), removed=int(len(rn)), protected=int(len(pn_))),
                   v3_box_check=box_check, residual=residual, seconds=round(time.time() - t_start, 1))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps(dict(points=summary["points"], voxels=summary["voxels"], v3_box_check=box_check, residual=residual), ensure_ascii=False))
    return dict(points=summary["points"], voxels=summary["voxels"], seconds=summary["seconds"])



if __name__ == "__main__":
    stage_main(run)
