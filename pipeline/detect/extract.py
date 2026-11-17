"""Per-scan person piece extraction: points of one preprocessed scan -> (ground/ceiling cut, device frame at the scan time,
projection) -> points inside each person's angular box -> Euclidean clustering -> piece selection with a distance score.

Statuses of a candidate (one per person key per scan):
  selected          one piece chosen (score >= select_min_score and >= runner-up + select_margin)
  ambiguous         two or more pieces fit the box about equally -> nothing accumulated, all recorded in `alt`
  ambiguous_multi   the chosen piece also fits another person's box about equally -> nothing accumulated
  taken_by_other    the piece fits another person's box clearly better
  none              no piece (no points in the box, or all pieces tiny / rejected by shape)
  (pieces of status `absorbed` in `alt`: small fragments of the best-scoring big piece merged into it before the selection)
  no_box            (made by the caller) the person's 2D box does not exist at this scan time
"""
from __future__ import annotations

import numpy as np

from pipeline.common.clusters import euclid_clusters
from pipeline.detect.ground import height_odom
from pipeline.detect.project import project_points, fov_mask, in_box_mask


def cluster_in_box(xyz: np.ndarray, rng: np.ndarray, cfg: dict) -> np.ndarray:
    e = cfg["extract"]
    tol = np.maximum(e["tol"], e["tol_per_m"] * rng)
    return euclid_clusters(xyz, tol)


def plane_basis(n):
    """Two unit vectors spanning the ground plane (n = unit upward normal)."""
    n = np.asarray(n, float) / max(np.linalg.norm(n), 1e-9)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def _piece(idx, xyz_dev, xyz_odom, rng, h, point_id, basis):
    xy = xyz_odom[idx] @ np.stack(basis, 1)                       # horizontal (leveled) coordinates in the odometry frame
    ext = xy.max(0) - xy.min(0)
    return dict(idx=idx, n=int(len(idx)), range_m=float(np.median(rng[idx])), footprint=float(np.hypot(*ext)),
                zmin_h=float(h[idx].min()), zmax_h=float(h[idx].max()), cxyz_odom=xyz_odom[idx].mean(0), cxyz_dev=xyz_dev[idx].mean(0),
                cxy_lv=xy.mean(0), point_id=point_id[idx])


def score_pieces(pieces: list, abox, cfg: dict) -> None:
    """Adds score / status (rejected_shape | candidate) to each piece in place."""
    e = cfg["extract"]
    ok = []
    for p in pieces:
        if p["footprint"] > e["person_fp_max"] or p["zmax_h"] > e["person_top_max"]:
            p["status"], p["score"] = "rejected_shape", 0.0
        else:
            p["status"] = "candidate"
            ok.append(p)
    if not ok:
        return
    if abox.truncated:
        r0 = min(p["range_m"] for p in ok)                       # box distance unreliable (cut box): nearest piece preferred
        for p in ok:
            p["score"] = float(np.exp(-0.5 * ((p["range_m"] - r0) / 1.0) ** 2))
    else:
        d_box = float(np.clip(abox.dist_m, 0.5, 15.0))           # a tiny far box (> 15 m) keeps a far estimate: near pieces score ~0
        for p in ok:
            p["score"] = float(np.exp(-0.5 * (np.log(p["range_m"] / d_box) / e["dist_sigma_ln"]) ** 2))


def _min_gap(A: np.ndarray, B: np.ndarray) -> float:
    """Smallest point-to-point distance between two point sets."""
    return float(np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1).min()))


def absorb_fragments(pieces: list, xyz: np.ndarray, rebuild, cfg: dict) -> list:
    """Merge small candidate pieces into a bigger candidate piece of the same person: smallest point gap between the pieces
    <= absorb_dist, n < absorb_frac * n_big, merged piece still within person_fp_max / person_top_max. Big pieces are processed
    largest first; absorbed pieces stay in the returned list with status `absorbed`. rebuild(idx) -> piece dict."""
    e = cfg["extract"]
    cands = sorted([q for q in pieces if q["status"] == "candidate"], key=lambda q: -q["n"])
    others = [q for q in pieces if q["status"] != "candidate"]
    out = []
    while cands:
        big = cands.pop(0)
        merged = [big]
        rest = []
        for q in cands:
            if q["n"] < e["absorb_frac"] * big["n"] and _min_gap(xyz[q["idx"]], xyz[np.concatenate([m["idx"] for m in merged])]) <= e["absorb_dist"]:
                trial = rebuild(np.concatenate([m["idx"] for m in merged] + [q["idx"]]))
                if trial["footprint"] <= e["person_fp_max"] and trial["zmax_h"] <= e["person_top_max"]:
                    merged.append(q)
                    continue
            rest.append(q)
        if len(merged) > 1:
            new = rebuild(np.concatenate([m["idx"] for m in merged]))
            new["status"] = "candidate"
            out.append(new)
            for q in merged[1:]:
                q["status"] = "absorbed"
                out.append(q)
        else:
            out.append(big)
        cands = rest
    return out + others


def extract_scan(scan: dict, plane_odom: dict, boxes_at_ts: list, geom, pose, cfg: dict, masks_out: dict | None = None) -> list:
    """boxes_at_ts: [(key, AngBox)] at the scan's time. plane_odom: dict(n, d, ceiling_h) in the odometry frame.
    -> one dict per key: key, scan_id, t_scan_us, n_fov, n_in_box, n_clusters, n_sel, status, score, score2, range_m, footprint,
       zmin_h, zmax_h, cxyz_odom, cxyz_dev, idx (point indices into the scan), point_id, alt (list of the other pieces)."""
    e = cfg["extract"]
    g = cfg["ground"]
    t_s = int(np.median(scan["t_point_us"])) if len(scan["t_point_us"]) else scan["t_scan_us"] + 25_000
    h = height_odom(scan["xyz_odom"], plane_odom)
    keep = (h > g["ground_h"]) & (h < plane_odom["ceiling_h"])
    R, p = pose.at(t_s)
    xyz_dev = (scan["xyz_odom"] - p) @ R
    cam, yaw, pitch, depth, u, v = project_points(geom, xyz_dev)
    fov0 = fov_mask(cam, pitch, depth, u, cfg)
    fov = fov0 & keep
    if masks_out is not None:
        masks_out.update(t_s=t_s, fov=fov0, keep=keep, inbox={}, h=h)
    if not boxes_at_ts:
        return []
    out = []
    per_key = {}
    basis = plane_basis(plane_odom["n"])
    for key, abox in boxes_at_ts:
        inb = np.nonzero(fov & in_box_mask(cam, yaw, pitch, abox, cfg["boxes"]["margin"]))[0]
        if masks_out is not None:
            masks_out["inbox"][key] = inb
        c = dict(key=key, scan_id=scan["scan_id"], t_scan_us=t_s, n_fov=int(fov.sum()), n_in_box=int(len(inb)), n_clusters=0, n_sel=0,
                 status="none", score=0.0, score2=0.0, range_m=float("nan"), footprint=float("nan"), zmin_h=float("nan"), zmax_h=float("nan"),
                 cxyz_odom=None, cxyz_dev=None, cxy_lv=None, idx=np.zeros(0, np.int64), point_id=np.zeros(0, np.int64), alt=[], abox=abox)
        pieces = []
        if len(inb) >= e["min_cluster_pts"]:
            lab = cluster_in_box(xyz_dev[inb], scan["range"][inb], cfg)
            for l in np.unique(lab):
                idx = inb[lab == l]
                if len(idx) >= e["min_cluster_pts"]:
                    pieces.append(_piece(idx, xyz_dev, scan["xyz_odom"], scan["range"], h, scan["point_id"], basis))
            c["n_clusters"] = len(pieces)
            score_pieces(pieces, abox, cfg)
            pieces = absorb_fragments(pieces, xyz_dev, lambda idx: _piece(idx, xyz_dev, scan["xyz_odom"], scan["range"], h, scan["point_id"], basis), cfg)
            score_pieces([q for q in pieces if q["status"] == "candidate"], abox, cfg)     # merged pieces re-scored
        cands = sorted([q for q in pieces if q["status"] == "candidate"], key=lambda q: -q["score"])
        c["alt"] = [dict(n=q["n"], range_m=q["range_m"], footprint=q["footprint"], zmax_h=q["zmax_h"], score=q["score"], status=q["status"])
                    for q in pieces]
        c["pieces"] = cands                                        # full candidate pieces (idx, point_id, centres) for the burst resolution
        if cands and cands[0]["score"] >= e["select_min_score"]:
            best = cands[0]
            second = cands[1]["score"] if len(cands) > 1 else 0.0
            c["score"], c["score2"] = best["score"], second
            if best["score"] - second >= e["select_margin"]:
                c.update(status="selected", n_sel=best["n"], range_m=best["range_m"], footprint=best["footprint"], zmin_h=best["zmin_h"],
                         zmax_h=best["zmax_h"], cxyz_odom=best["cxyz_odom"], cxyz_dev=best["cxyz_dev"], cxy_lv=best["cxy_lv"], idx=best["idx"], point_id=best["point_id"])
            else:
                c["status"] = "ambiguous"
        per_key[key] = c
        out.append(c)
    # the same piece selected for two persons: clear winner keeps it, otherwise both ambiguous_multi
    sel = [c for c in out if c["status"] == "selected"]
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            a, b = sel[i], sel[j]
            if a["status"] != "selected" or b["status"] != "selected":
                continue
            shared = np.intersect1d(a["point_id"], b["point_id"])
            if len(shared) == 0:
                continue
            if abs(a["score"] - b["score"]) >= e["select_margin"]:
                loser = a if a["score"] < b["score"] else b
                loser.update(status="taken_by_other", n_sel=0, idx=np.zeros(0, np.int64), point_id=np.zeros(0, np.int64))
            else:
                for c in (a, b):
                    c.update(status="ambiguous_multi", n_sel=0, idx=np.zeros(0, np.int64), point_id=np.zeros(0, np.int64))
    return out


def no_box_cand(key, scan, t_s, n_fov):
    return dict(key=key, scan_id=scan["scan_id"], t_scan_us=t_s, n_fov=n_fov, n_in_box=0, n_clusters=0, n_sel=0, status="no_box", score=0.0, score2=0.0,
                range_m=np.nan, footprint=np.nan, zmin_h=np.nan, zmax_h=np.nan, cxyz_odom=None, cxyz_dev=None, cxy_lv=None, idx=np.zeros(0, np.int64),
                point_id=np.zeros(0, np.int64), alt=[], abox=None)


def extract_frame(scans: list, keys: list, boxes_at, plane_odom: dict, geom, pose, cfg: dict, t_c: int):
    """All candidate scans of one frame. keys = persons with a 2D box in this frame (they get a candidate row per scan); every
    person whose box exists at a scan time competes for the pieces (a person missed by YOLO in this frame but tracked around it
    must not hand its piece to a neighbour). -> (cands: key -> [cand per scan], masks: per-scan dict from extract_scan)."""
    cands = {k: [] for k in keys}
    masks = []
    for s in scans:
        mo = {}
        t_s = int(np.median(s["t_point_us"])) if len(s["t_point_us"]) else s["t_scan_us"] + 25_000
        res = extract_scan(s, plane_odom, boxes_at(t_s), geom, pose, cfg, masks_out=mo)
        masks.append(mo)
        got = set()
        for r in res:
            if r["key"] in cands:
                r["dt_cam_s"] = (r["t_scan_us"] - t_c) / 1e6
                cands[r["key"]].append(r)
                got.add(r["key"])
        for k in keys:
            if k not in got:
                c = no_box_cand(k, s, mo["t_s"], int(mo["fov"].sum()))
                c["dt_cam_s"] = (c["t_scan_us"] - t_c) / 1e6
                cands[k].append(c)
    return cands, masks
