"""Selective accumulation of one person's selected pieces (one camera frame): cross-scan consistency check, transform to the
leveled device frame at the image time, final Euclidean clustering, accumulated box and nearest-burst box.

No object motion compensation: a walking person leaves a smear along its path (recorded by the `smeared` flag and by the
difference between box_all / box_burst and box_acc). Since 2026-11-11 box_acc is limited to the main-component scans within
motion_window_m of travel (at the robust constant-velocity speed of the main component) around the scan nearest to the image
time, and the per-burst hypothesis (resolve_bursts) removes selected pieces inconsistent with the other scans of the burst.
"""
from __future__ import annotations

import math

import numpy as np

from pipeline.common.box_fit import fit_box, pca_yaw
from pipeline.common.clusters import euclid_clusters
from pipeline.detect.ground import to_level


def _theil_sen(t, x):
    """Robust line x = a + b t (median of pairwise slopes)."""
    n = len(t)
    i, j = np.triu_indices(n, 1)
    dt = t[j] - t[i]
    ok = dt > 1e-6
    b = float(np.median((x[j] - x[i])[ok] / dt[ok])) if ok.any() else 0.0
    a = float(np.median(x - b * t))
    return a, b


def consistency_filter(cands: list, cfg: dict) -> list:
    """Selected candidates whose centre is farther than consistency_res from the robust constant-velocity fit of all selected
    centres become rejected_consistency (only when at least consistency_min are selected)."""
    a = cfg["accumulate"]
    sel = [c for c in cands if c["status"] == "selected"]
    if len(sel) < a["consistency_min"]:
        return cands
    t = np.array([c["t_scan_us"] for c in sel], float) / 1e6
    t -= t.mean()
    P = np.array([c["cxy_lv"] if c.get("cxy_lv") is not None else c["cxyz_odom"][:2] for c in sel], float)   # leveled horizontal centres
    ax, bx = _theil_sen(t, P[:, 0])
    ay, by = _theil_sen(t, P[:, 1])
    res = np.hypot(P[:, 0] - (ax + bx * t), P[:, 1] - (ay + by * t))
    for c, r in zip(sel, res):
        c["consistency_res"] = float(r)
        if r > a["consistency_res"]:
            c["status"] = "rejected_consistency"
            c["n_sel_rejected"] = c["n_sel"]
            c["n_sel"] = 0
    return cands


def _wmedian(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][int(np.searchsorted(cw, 0.5 * cw[-1]))])


def _clear(c, status, rejected=False):
    if rejected:
        c["n_sel_rejected"] = c["n_sel"]
    c.update(status=status, n_sel=0, idx=np.zeros(0, np.int64), point_id=np.zeros(0, np.int64))


def _take(c, q, how):
    c.update(status="selected", n_sel=q["n"], range_m=q["range_m"], footprint=q["footprint"], zmin_h=q["zmin_h"], zmax_h=q["zmax_h"],
             cxyz_odom=q["cxyz_odom"], cxyz_dev=q["cxyz_dev"], cxy_lv=q["cxy_lv"], idx=q["idx"], point_id=q["point_id"], score=q["score"], resolved=how)


def resolve_bursts(cands_by_key: dict, cfg: dict) -> None:
    """Per person and burst (>= 2 rows of status selected / ambiguous): hypothesis = score*points weighted median of the centres
    of all candidate pieces of those rows. Selected pieces farther than burst_res from it -> rejected_range. An ambiguous row
    whose candidate pieces within burst_res are one piece plus fragments (< absorb_frac of it) -> resolved to it (resolved =
    "burst"), unless that piece shares points
    with another person's selected piece of the same scan (-> ambiguous_multi). In place."""
    a = cfg["accumulate"]
    gap = a["burst_gap_s"]
    tol = a["burst_res"]
    frac = cfg["extract"]["absorb_frac"]
    taken = {}                                                     # (scan_id, key) -> point ids selected before the resolution
    for key, rows in cands_by_key.items():
        for c in rows:
            if c["status"] == "selected":
                taken[(c["scan_id"], key)] = c["point_id"]
    resolved = []
    for key, rows in cands_by_key.items():
        use = sorted([c for c in rows if c["status"] in ("selected", "ambiguous") and c.get("pieces")], key=lambda c: c["t_scan_us"])
        bursts, cur = [], []
        for c in use:
            if cur and (c["t_scan_us"] - cur[-1]["t_scan_us"]) / 1e6 > gap:
                bursts.append(cur); cur = []
            cur.append(c)
        if cur:
            bursts.append(cur)
        for b in bursts:
            if len(b) < 2:
                continue
            P = np.array([q["cxy_lv"] for c in b for q in c["pieces"]], float)
            w = np.array([q["score"] * q["n"] for c in b for q in c["pieces"]], float)
            if len(P) < 2 or w.sum() <= 0:
                continue
            hyp = np.array([_wmedian(P[:, 0], w), _wmedian(P[:, 1], w)])
            for c in b:
                if c["status"] == "selected":
                    if np.linalg.norm(np.asarray(c["cxy_lv"]) - hyp) > tol:
                        _clear(c, "rejected_range", rejected=True)
                        c["burst_hyp_m"] = float(np.linalg.norm(np.asarray(c["cxy_lv"]) - hyp))
                else:
                    near = sorted([q for q in c["pieces"] if np.linalg.norm(q["cxy_lv"] - hyp) <= tol], key=lambda q: -q["n"])
                    dominant = len(near) >= 1 and all(q["n"] < frac * near[0]["n"] for q in near[1:])   # the others are fragments
                    if dominant:
                        q = near[0]
                        clash = any(len(np.intersect1d(q["point_id"], pid)) for (sid, k2), pid in taken.items() if sid == c["scan_id"] and k2 != key)
                        if clash:
                            _clear(c, "ambiguous_multi")
                        else:
                            _take(c, q, "burst")
                            resolved.append((key, c))
    # two persons resolved to overlapping pieces in the same scan -> both back to ambiguous_multi
    for i in range(len(resolved)):
        for j in range(i + 1, len(resolved)):
            (ka, ca), (kb, cb) = resolved[i], resolved[j]
            if ka != kb and ca["scan_id"] == cb["scan_id"] and ca["status"] == cb["status"] == "selected" and len(np.intersect1d(ca["point_id"], cb["point_id"])):
                _clear(ca, "ambiguous_multi"); _clear(cb, "ambiguous_multi")


def _pair_velocity(xyz_lv, sid, in_comp, scans, t_scan, band_min_m):
    """Horizontal velocity of the main component from scan pairs: for each pair of scans the centroids of the points in the
    height band both scans share (band >= band_min_m) are compared, so a different body part per scan (the nodding sweep)
    is not read as motion; velocity = median of the pairwise slopes. None without a usable pair."""
    pts = {s: xyz_lv[in_comp & (sid == s)] for s in scans}
    zr = {s: (pts[s][:, 2].min(), pts[s][:, 2].max()) for s in scans if len(pts[s])}
    vx, vy = [], []
    for i in range(len(scans)):
        for j in range(i + 1, len(scans)):
            si, sj = scans[i], scans[j]
            if si not in zr or sj not in zr:
                continue
            lo, hi = max(zr[si][0], zr[sj][0]), min(zr[si][1], zr[sj][1])
            dt = (t_scan[sj] - t_scan[si]) / 1e6
            if hi - lo < band_min_m or dt <= 1e-6:
                continue
            A, B = pts[si], pts[sj]
            ca = A[(A[:, 2] >= lo) & (A[:, 2] <= hi)][:, :2].mean(0)
            cb = B[(B[:, 2] >= lo) & (B[:, 2] <= hi)][:, :2].mean(0)
            vx.append((cb[0] - ca[0]) / dt); vy.append((cb[1] - ca[1]) / dt)
    if not vx:
        return None
    return np.array([float(np.median(vx)), float(np.median(vy))])


def _bursts(scan_ids: np.ndarray, t_scan: dict, gap_s: float) -> list:
    """Groups of scans whose start times are at most gap_s apart (one pass of the nodding LiDAR)."""
    ids = sorted(int(s) for s in np.unique(scan_ids))
    out, cur = [], [ids[0]]
    for s in ids[1:]:
        if (t_scan[s] - t_scan[cur[-1]]) / 1e6 <= gap_s:
            cur.append(s)
        else:
            out.append(cur)
            cur = [s]
    out.append(cur)
    return out


def _box(P, yaw):
    cx, cy, cz, l, w, h = fit_box(P, yaw)
    return dict(cx=cx, cy=cy, cz=cz, l=l, w=w, h=h, yaw=float(yaw))


def _ref_time(tp: np.ndarray, sid: np.ndarray, t_scan: dict, t_c: int, gap_s: float) -> int:
    """Reference time of the compensated box: the image time clipped to the measured time range of the pass (burst) of these points
    nearest to it. Between two passes the box therefore stops at the nearest measured instant instead of being extrapolated to a
    place where the LiDAR saw nothing (2026-11-16)."""
    best = None
    for b in _bursts(sid, t_scan, gap_s):
        m = np.isin(sid, b)
        t0, t1 = int(tp[m].min()), int(tp[m].max())
        d = 0 if t0 <= t_c <= t1 else min(abs(t_c - t0), abs(t_c - t1))
        if best is None or d < best[0]:
            best = (d, int(np.clip(t_c, t0, t1)))
    return best[1]


def _comp_box(P, tp, vel, t_c, yaw):
    """Motion-compensated box (motion_mode box, 2026-11-16): the box of the points shifted along the fitted velocity to the
    reference time t_c (the image time clipped by _ref_time), P_i - vel * (t_i - t_c). The points themselves are not moved. The shift
    is along yaw (= motion direction), so only the length changes and the centre moves to the position at t_c. If the compensation
    does not shorten the box (velocity inconsistent with the smear) the uncompensated box is returned. -> (box, applied)."""
    Pc = np.array(P, float)
    Pc[:, :2] -= np.asarray(vel, float)[None, :] * ((np.asarray(tp, float) - t_c) / 1e6)[:, None]
    bu, bc = _box(P, yaw), _box(Pc, yaw)
    return (bc, True) if bc["l"] < bu["l"] else (bu, False)


def _nan_box():
    return dict(cx=np.nan, cy=np.nan, cz=np.nan, l=np.nan, w=np.nan, h=np.nan, yaw=np.nan)


def _to_odom(box, plane, R, p):
    """Box in the leveled device frame -> odometry frame (adds cx_o, cy_o, cz_o, yaw_o)."""
    if not np.isfinite(box["cx"]):
        box.update(cx_o=np.nan, cy_o=np.nan, cz_o=np.nan, yaw_o=np.nan)
        return box
    c_lv = np.array([box["cx"], box["cy"], box["cz"] + plane["ground_z"]])
    c_dev = plane["R_level"].T @ c_lv
    c_o = R @ c_dev + p
    d_lv = np.array([math.cos(box["yaw"]), math.sin(box["yaw"]), 0.0])
    d_o = R @ (plane["R_level"].T @ d_lv)
    box.update(cx_o=float(c_o[0]), cy_o=float(c_o[1]), cz_o=float(c_o[2]), yaw_o=float(math.atan2(d_o[1], d_o[0])))
    return box


def accumulate_person(cands: list, store, pose, plane: dict, t_c: int, cfg: dict) -> dict | None:
    """cands: this person's candidates (only status == selected are used). plane: frame plane (R_level, ground_z) in the device
    frame at t_c. -> Person dict (None when nothing was selected)."""
    a = cfg["accumulate"]
    sel = [c for c in cands if c["status"] == "selected" and len(c["idx"])]
    if not sel:
        return None
    R, p = pose.at(t_c)
    xyz_o, pid, sid, tp, rng = [], [], [], [], []
    for c in sel:
        s = store.get(c["scan_id"])
        xyz_o.append(s["xyz_odom"][c["idx"]])
        pid.append(s["point_id"][c["idx"]])
        sid.append(np.full(len(c["idx"]), c["scan_id"], np.int32))
        tp.append(s["t_point_us"][c["idx"]])
        rng.append(s["range"][c["idx"]])
    xyz_o, pid, sid, tp, rng = (np.concatenate(v) for v in (xyz_o, pid, sid, tp, rng))
    assert len(np.unique(pid)) == len(pid), "a point was selected twice for one person"
    xyz_dev = (xyz_o - p) @ R
    xyz_lv = to_level(xyz_dev, plane)
    lab = euclid_clusters(xyz_lv, a["tol"])
    labels, counts = np.unique(lab, return_counts=True)
    # bursts over ALL selected points (one pass of the nodding LiDAR each); the burst nearest to the image time defines the
    # instantaneous box, and the main component is the component holding that burst (if it is big enough), else the largest
    t_scan = {int(s): int(np.median(tp[sid == s])) for s in np.unique(sid)}
    bursts_all = _bursts(sid, t_scan, a["burst_gap_s"])
    burst_t = [float(np.mean([t_scan[s] for s in b])) for b in bursts_all]
    nearest = int(np.argmin([abs(bt - t_c) for bt in burst_t]))
    in_near = np.isin(sid, bursts_all[nearest])
    lab_near, cnt_near = np.unique(lab[in_near], return_counts=True)
    cand_lab = lab_near[int(np.argmax(cnt_near))]
    main_lab = cand_lab if counts[labels == cand_lab][0] >= a["valid_min_pts"] else labels[int(np.argmax(counts))]
    in_comp = lab == main_lab                                     # the whole main component (box_burst, untrimmed counts)
    in_main = in_comp.copy()
    # speed of the main component from the selected piece centres (scan-pair velocity, >= 2 scans). final keeps the v3 motion mode
    # 'box' only (the v2 'window' trimming branch of 3d_detection/src/accumulate.py was removed; trimmed stays False)
    scans_untrimmed = sorted(int(s) for s in np.unique(sid[in_main]))
    v_mps, trimmed, vel, travel_m = float("nan"), False, None, 0.0
    mode = a.get("motion_mode", "window")
    if mode != "box":
        raise ValueError(f"accumulate.motion_mode {mode!r}: final implements the v3 mode 'box' only")
    if len(scans_untrimmed) >= 2:
        vel = _pair_velocity(xyz_lv, sid, in_comp, scans_untrimmed, t_scan, a["speed_band_min_m"])
    if vel is not None:
        v_mps = float(np.hypot(*vel))
        tt = np.array([t_scan[s] for s in scans_untrimmed], float) / 1e6
        travel_m = v_mps * (tt.max() - tt.min())
    P = xyz_lv[in_main]
    scans_main = sid[in_main]
    bursts = _bursts(scans_main, t_scan, a["burst_gap_s"])
    # yaw: motion between the first and last burst centres of all selected points, else PCA of the main component
    yaw_source, yaw_reliable = "pca", 1
    if len(bursts_all) >= 2:
        c0 = xyz_lv[np.isin(sid, bursts_all[0])][:, :2].mean(0)
        c1 = xyz_lv[np.isin(sid, bursts_all[-1])][:, :2].mean(0)
        if np.linalg.norm(c1 - c0) >= a["motion_min_m"]:
            yaw, yaw_source = float(math.atan2(*(c1 - c0)[::-1])), "motion"
    if yaw_source == "pca" and vel is not None and travel_m >= a["motion_min_m"]:
        yaw, yaw_source = float(math.atan2(vel[1], vel[0])), "motion"     # one pass only: direction of the fitted velocity (2026-11-11)
    if yaw_source == "pca":
        yaw = float(pca_yaw(P)) if len(P) >= 3 else 0.0
        xy = P[:, :2] - P[:, :2].mean(0)
        ev = np.linalg.eigvalsh(xy.T @ xy) if len(P) >= 3 else np.array([1.0, 1.0])
        yaw_reliable = int(ev[1] / max(ev[0], 1e-9) >= a["pca_ratio_min"] ** 2)      # axis-length ratio >= pca_ratio_min
    compensated, box_dt_s = False, float("nan")
    if mode == "box" and vel is not None and travel_m >= a["motion_min_m"]:
        t_ref = _ref_time(tp[in_main], sid[in_main], t_scan, t_c, a["burst_gap_s"])
        box_dt_s = (t_ref - t_c) / 1e6
        box_lv, compensated = _comp_box(P, tp[in_main], vel, t_ref, yaw)  # whole pass kept; box moved to t_ref and shortened by the travel
    else:
        box_lv = _box(P, yaw)
    box_acc = _to_odom(box_lv, plane, R, p)
    box_all = _box(xyz_lv, yaw)                                   # every selected point (all components): the full smear
    Pb = xyz_lv[in_near & in_comp] if (in_near & in_comp).any() else xyz_lv[in_near]     # nearest pass, not trimmed
    box_burst = _to_odom(_box(Pb, yaw) if len(Pb) else _nan_box(), plane, R, p)
    zmin_h, zmax_h = float(P[:, 2].min()), float(P[:, 2].max())
    fp_all, fp_burst = max(box_all["l"], box_all["w"]), max(box_burst["l"], box_burst["w"]) if len(Pb) else np.nan
    flags = dict(lower_only=zmax_h < a["lower_only_top"], upper_only=zmin_h > a["upper_only_bottom"], single_scan=len(np.unique(scans_main)) == 1,
                 fragmented=len(labels) > 1, smeared=bool(np.isfinite(fp_burst) and fp_all - fp_burst > a["smear_extra"]), trimmed=trimmed,
                 compensated=compensated)
    valid = bool(len(P) >= a["valid_min_pts"] and (zmax_h - zmin_h) >= a["valid_min_h"])
    t_first, t_last = int(tp[in_main].min()), int(tp[in_main].max())
    return dict(key=sel[0]["key"], t_c=int(t_c), n_pts=int(len(P)), n_pts_all=int(len(pid)), n_scans=int(len(np.unique(scans_main))),
                scan_ids=[int(s) for s in np.unique(scans_main)], n_bursts=len(bursts), nearest_burst=bursts_all[nearest],
                t_first_us=t_first, t_last_us=t_last, span_s=(t_last - t_first) / 1e6, n_components=int(len(labels)), b_dt_s=(burst_t[nearest] - t_c) / 1e6,
                xyz_lv=xyz_lv, xyz_odom=xyz_o, point_id=pid, scan_id_pts=sid, t_pts=tp, range_pts=rng, in_main=in_main,
                box_acc=box_acc, box_burst=box_burst, box_all=box_all, n_scans_all=int(len(np.unique(sid))), n_bursts_all=len(bursts_all), yaw_source=yaw_source, yaw_reliable=yaw_reliable, zmin_h=zmin_h, zmax_h=zmax_h,
                flags={k: bool(v) for k, v in flags.items()}, valid=valid, v_mps=v_mps, n_scans_untrimmed=len(scans_untrimmed), box_dt_s=box_dt_s)


def assert_unique_points(persons: list) -> None:
    seen = set()
    for q in persons:
        ids = set(int(v) for v in q["point_id"])
        dup = seen & ids
        assert not dup, f"{len(dup)} points assigned to two persons (e.g. {q['key']})"
        seen |= ids
