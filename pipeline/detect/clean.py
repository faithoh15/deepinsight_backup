"""Residual (ghost) removal of the selective-accumulation results, as a post-processing of the saved person PLYs.

A frame's window (anchor +-20 scans) shows the same person at every pass of the nodding LiDAR; the detector keeps one main
component (in_main == 1, the boxed cluster) and records the rest of the points it attributed to that person (in_main == 0).
Those are the residuals to remove from the frame: OTHER_BURST = another pass (past / future position), TRIMMED = the same pass
but outside the motion window. Rule fixed by the user (2026-11-16): every window point inside a person's 3D box (box_acc, any scan)
stays untouched; only the object's points OUTSIDE the boxes are removed. With the v3 boxes (motion-compensated, whole pass kept as
main component) the main-component points outside the box are the smear of the same pass: removed as SAME_PASS. Points the detector did not attribute to a person (ambiguous / rejected scans, persons without
a 2D box) are not touched by the basic removal; the extended removal (`predicted_ghosts`) also removes, in every other scan of
the window, person-shaped clusters near the position predicted from the selected pieces (constant velocity in time; PREDICTED).
Window points are matched by scan id and nearest neighbour (the PLY stores the same leveled
coordinates in float32), so nothing is recomputed from the detector.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from pipeline.common.clusters import euclid_clusters
from pipeline.detect.project import fov_mask, project_points

OTHER_BURST, SAME_PASS, PREDICTED = 1, 2, 3
TRIMMED = SAME_PASS                                   # v2 name: the same pass, outside the motion window
REASON_NAME = {OTHER_BURST: "other_burst", SAME_PASS: "same_pass", PREDICTED: "predicted"}


def _bursts_by_time(scan_ids: np.ndarray, t_scan: dict, gap_s: float) -> list:
    """Scans sorted by time, split where consecutive scan times are more than gap_s apart -> list of lists of scan ids."""
    order = sorted(scan_ids, key=lambda s: t_scan[s])
    out = []
    for s in order:
        if out and t_scan[s] - t_scan[out[-1][-1]] <= gap_s:
            out[-1].append(s)
        else:
            out.append([s])
    return out


def ghost_points(d: dict, burst_gap_s: float) -> tuple[dict, np.ndarray]:
    """d: fields of one person PLY (x, y, z, scan_id, time_rel_s, in_main, ...). -> (fields of the in_main == 0 points, reason
    per point: TRIMMED if the scan belongs to the pass of a main-component scan, else OTHER_BURST)."""
    sid = np.asarray(d["scan_id"]).astype(int)
    main = np.asarray(d["in_main"]) == 1
    ghost = ~main
    out = {k: np.asarray(v)[ghost] for k, v in d.items()}
    if not ghost.any():
        return out, np.zeros(0, np.uint8)
    t_rel = np.asarray(d["time_rel_s"], float)
    t_scan = {int(s): float(np.median(t_rel[sid == s])) for s in np.unique(sid)}
    main_scans = set(int(s) for s in np.unique(sid[main]))
    same_pass = set()
    for b in _bursts_by_time(np.unique(sid), t_scan, burst_gap_s):
        if main_scans & set(b):
            same_pass |= set(b)
    reason = np.where(np.isin(out["scan_id"].astype(int), list(same_pass)), TRIMMED, OTHER_BURST).astype(np.uint8)
    return out, reason


def match_points(W: np.ndarray, w_scan: np.ndarray, G: np.ndarray, g_scan: np.ndarray, tol: float = 0.002) -> tuple[np.ndarray, int]:
    """Mark the window points W (N,3) that are the ghost points G (M,3): same scan id, nearest neighbour within tol (m).
    -> (mask over W, number of ghost points without a match)."""
    mask = np.zeros(len(W), bool)
    n_un = 0
    w_scan, g_scan = np.asarray(w_scan).astype(int), np.asarray(g_scan).astype(int)
    for s in np.unique(g_scan):
        wi = np.nonzero(w_scan == s)[0]
        gi = np.nonzero(g_scan == s)[0]
        if len(wi) == 0:
            n_un += len(gi)
            continue
        dist, j = cKDTree(W[wi]).query(np.asarray(G[gi], float), k=1)
        ok = dist <= tol
        mask[wi[j[ok]]] = True
        n_un += int((~ok).sum())
    return mask, n_un


def inside_boxes(W: np.ndarray, boxes: list, margin: float = 0.05) -> np.ndarray:
    """Mask of the points inside any of the boxes (dict cx cy cz l w h yaw, leveled frame), each side enlarged by margin (m)."""
    m = np.zeros(len(W), bool)
    for b in boxes:
        if not np.isfinite(b["cx"]):
            continue
        c, s_ = math.cos(b["yaw"]), math.sin(b["yaw"])
        dx, dy = W[:, 0] - b["cx"], W[:, 1] - b["cy"]
        u, v = dx * c + dy * s_, -dx * s_ + dy * c
        m |= (np.abs(u) <= b["l"] / 2 + margin) & (np.abs(v) <= b["w"] / 2 + margin) & (np.abs(W[:, 2] - b["cz"]) <= b["h"] / 2 + margin)
    return m


def frame_ghosts(persons: list, W: np.ndarray, w_scan: np.ndarray, burst_gap_s: float, tol: float = 0.002, boxes: list | None = None,
                 margin: float = 0.05) -> dict:
    """persons: [(key, track_id, ply fields)] of one frame. W / w_scan: the frame's window points (leveled frame at the image
    time) and their scan ids. boxes: the frame's 3D boxes (box_acc of every person): points inside any of them are fixed.
    -> dict(removed = mask over W of the residual points, main = mask over W of the main-component points, fixed = mask over W of the
    points inside a box, keep_ply = per person mask over its PLY points that stay (main or inside a box), fields = merged removed
    points (x, y, z, scan_id, time_rel_s, track_id, reason), stats = per person dict)."""
    removed = np.zeros(len(W), bool)
    main = np.zeros(len(W), bool)
    fixed = inside_boxes(W, boxes or [], margin)
    fields = dict(x=[], y=[], z=[], scan_id=[], time_rel_s=[], track_id=[], reason=[])
    stats, keep_ply = [], []
    for key, tid, d in persons:
        is_main = np.asarray(d["in_main"]) == 1
        g0, reason0 = ghost_points(d, burst_gap_s)                      # in_main == 0 points with their pass reason
        P_all = np.stack([np.asarray(d["x"]), np.asarray(d["y"]), np.asarray(d["z"])], 1).astype(float)
        in_box_all = inside_boxes(P_all, boxes or [], margin)
        if boxes is None:
            in_box_all |= is_main                                         # no box information: the main component is the detection
        kp = in_box_all.copy()                                            # every PLY point inside a box stays, main or not
        keep_ply.append(kp)
        # ghosts = PLY points outside every box: in_main == 0 with the pass reason, in_main == 1 as the smear of the same pass
        reason_all = np.zeros(len(P_all), np.uint8)
        reason_all[~is_main] = reason0
        reason_all[is_main] = SAME_PASS
        out_idx = np.nonzero(~in_box_all)[0]
        g = {k: np.asarray(v)[out_idx] for k, v in d.items()}
        reason = reason_all[out_idx]
        G = P_all[out_idx]
        m, n_un = match_points(W, w_scan, G, g["scan_id"], tol)
        m &= ~fixed
        removed |= m
        keep = is_main & in_box_all
        M = P_all[keep]
        mm, n_un_main = match_points(W, w_scan, M, np.asarray(d["scan_id"])[keep], tol)
        main |= mm
        in_box = in_box_all & ~is_main                                    # other-scan points kept because they are inside a box
        for k in ("x", "y", "z", "scan_id", "time_rel_s"):
            fields[k].append(np.asarray(g[k]))
        fields["track_id"].append(np.full(len(reason), int(tid), np.int32))
        fields["reason"].append(reason)
        stats.append(dict(key=key, track_id=int(tid), n_main=int(is_main.sum()), n_kept_in_box=int(in_box.sum()), n_ghost=int(len(reason)),
                          n_ghost_other_burst=int((reason == OTHER_BURST).sum()), n_ghost_same_pass=int((reason == SAME_PASS).sum()),
                          n_ghost_main_outside=int((reason == SAME_PASS)[is_main[out_idx]].sum()),
                          n_ghost_scans=int(len(np.unique(g["scan_id"]))), n_unmatched=int(n_un), n_unmatched_main=int(n_un_main)))
    out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in fields.items()}
    return dict(removed=removed, main=main, fixed=fixed, keep_ply=keep_ply, fields=out, stats=stats)


def _trajectory(d: dict):
    """Per-scan centroids (x, y) and times of all points of one person PLY -> (t (K,), xy (K,2)) sorted by time."""
    sid = np.asarray(d["scan_id"]).astype(int)
    t_rel = np.asarray(d["time_rel_s"], float)
    xy = np.stack([np.asarray(d["x"], float), np.asarray(d["y"], float)], 1)
    scans = np.unique(sid)
    t = np.array([np.median(t_rel[sid == s]) for s in scans])
    c = np.array([xy[sid == s].mean(0) for s in scans])
    o = np.argsort(t)
    return t[o], c[o]


def predict_xy(t: np.ndarray, xy: np.ndarray, t_q: float, extrap_max_s: float):
    """Position at t_q from the known (t, xy): constant-velocity least squares when the known span is >= 0.1 s, else the mean;
    None when t_q is more than extrap_max_s outside the known time range."""
    if t_q < t.min() - extrap_max_s or t_q > t.max() + extrap_max_s:
        return None
    if t.max() - t.min() >= 0.1:
        A = np.stack([t, np.ones_like(t)], 1)
        coef, *_ = np.linalg.lstsq(A, xy, rcond=None)
        return np.array([t_q, 1.0]) @ coef
    return xy.mean(0)


def scan_fov_mask(scan: dict, pose, geom, cfg: dict) -> np.ndarray:
    """Camera-FOV mask of one preprocessed scan, computed exactly as in extract_scan: device frame at the median point time of the
    scan, projection into the nearer camera, depth > 0, |pitch| <= pitch_max_deg, inside the tile."""
    t_s = int(np.median(scan["t_point_us"])) if len(scan["t_point_us"]) else scan["t_scan_us"] + 25_000
    R, p = pose.at(t_s)
    cam, yaw, pitch, depth, u, v = project_points(geom, (scan["xyz_odom"] - p) @ R)
    return fov_mask(cam, pitch, depth, u, cfg)


def predicted_ghosts(d: dict, W: np.ndarray, w_scan: np.ndarray, w_t: np.ndarray, exclude: np.ndarray, radius_m: float, search_m: float,
                     fp_max: float, tol: float, tol_per_m: float, extrap_max_s: float, min_pts: int, z_min: float = 0.1, z_max: float = 2.3,
                     allowed: np.ndarray | None = None):
    """Extended removal for one person. d: its PLY fields (selected pieces, leveled frame). W / w_scan / w_t: window points, scan
    ids, times relative to the image. exclude: window points that must stay (main components, already removed points).
    allowed: window points that may be clustered and removed at all (the camera-FOV points, 2026-11-16); None = all.
    In every window scan that holds none of this person's selected points: points within search_m (horizontal) of the predicted
    position and z in (z_min, z_max) are clustered (tol = max(tol, tol_per_m * horizontal distance)); a cluster with >= min_pts
    points, footprint <= fp_max and centre within radius_m of the prediction is removed. -> (mask over W, [dict(scan_id, n)])."""
    t_k, xy_k = _trajectory(d)
    known = set(int(s) for s in np.unique(np.asarray(d["scan_id"])))
    mask = np.zeros(len(W), bool)
    info = []
    w_scan = np.asarray(w_scan).astype(int)
    for s in np.unique(w_scan):
        if int(s) in known:
            continue
        wi = np.nonzero((w_scan == s) & ~exclude & (allowed if allowed is not None else True))[0]
        if len(wi) == 0:
            continue
        c = predict_xy(t_k, xy_k, float(np.median(w_t[wi])), extrap_max_s)
        if c is None:
            continue
        P = W[wi]
        near = (np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) <= search_m) & (P[:, 2] > z_min) & (P[:, 2] < z_max)
        idx = wi[near]
        if len(idx) < min_pts:
            continue
        Q = W[idx]
        lab = euclid_clusters(Q, np.maximum(tol, tol_per_m * np.hypot(Q[:, 0], Q[:, 1])))
        n_rem = 0
        for l in np.unique(lab):
            m = lab == l
            if m.sum() < min_pts:
                continue
            ext = Q[m, :2].max(0) - Q[m, :2].min(0)
            if np.hypot(*ext) > fp_max or np.linalg.norm(Q[m, :2].mean(0) - c) > radius_m:
                continue
            mask[idx[m]] = True
            n_rem += int(m.sum())
        if n_rem:
            info.append(dict(scan_id=int(s), n=n_rem))
    return mask, info


def ext_params(cfg: dict) -> dict:
    """Extended-removal values from the detector config (no new tuned numbers): radius = accumulate.burst_res, footprint / clustering /
    minimum points = extract.*; search radius 2.0 m and extrapolation 0.6 s (burst_res at walking speed ~1.3 m/s) are fixed here."""
    e, a = cfg["extract"], cfg["accumulate"]
    return dict(radius_m=a["burst_res"], search_m=2.0, fp_max=e["person_fp_max"], tol=e["tol"], tol_per_m=e["tol_per_m"], extrap_max_s=0.6, min_pts=e["min_cluster_pts"])


def extend_frame(res: dict, persons: list, W: np.ndarray, w_scan: np.ndarray, w_t: np.ndarray, prm: dict, allowed: np.ndarray | None = None) -> None:
    """Apply predicted_ghosts for every person on top of a frame_ghosts result (in place): removed mask, fields (reason PREDICTED)
    and stats (n_ghost_predicted, n_scans_predicted) are extended. Main components, points inside a box and already removed points
    are excluded (the box interior is fixed before clustering, so a cluster can never reach into a box). allowed: see predicted_ghosts."""
    for (key, tid, d), st in zip(persons, res["stats"]):
        exclude = res["main"] | res["removed"] | res["fixed"]
        m, info = predicted_ghosts(d, W, w_scan, w_t, exclude, allowed=allowed, **prm)
        res["removed"] |= m
        idx = np.nonzero(m)[0]
        add = dict(x=W[idx, 0], y=W[idx, 1], z=W[idx, 2], scan_id=np.asarray(w_scan)[idx], time_rel_s=np.asarray(w_t)[idx],
                   track_id=np.full(len(idx), int(tid), np.int32), reason=np.full(len(idx), PREDICTED, np.uint8))
        for k in res["fields"]:
            res["fields"][k] = np.concatenate([np.asarray(res["fields"][k]), add[k]]) if len(res["fields"][k]) else add[k]
        st.update(n_ghost_predicted=int(len(idx)), n_scans_predicted=len(info))
