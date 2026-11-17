"""Per-pass observations for the tracker, built from the v3 detection rows (read only).

One observation = one 2D key x one nodding pass: rows of the same key whose pass time (t_img + b_dt_s, the mean time of the nearest
burst) are within same_key_gap_s. Rows of different keys are merged only if the pass times agree AND the two observations share at
least merge_point_share of their points AND their pass boxes overlap by merge_box_overlap (design §2); otherwise a dup_unmerged
event is recorded and both stay separate.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_track_config(path: str | None = None) -> dict:
    if path:
        return yaml.safe_load(open(path, encoding="utf-8"))
    from pipeline.common.config import load_config
    return load_config()["tracking"]          # configs/pipeline.yaml tracking (= track.yaml)


@dataclass
class Obs:
    obs_id: str
    t_us: int
    keys: list
    frames: list
    rows: list
    x: float
    y: float
    z_c: float
    sigma: float
    quality: str
    range_m: float
    box: dict
    bbox: dict
    v0: np.ndarray = field(default_factory=lambda: np.zeros(2))
    n_pts: int = 0


def pass_time_us(row: dict) -> int:
    return int(row["t_img_us"]) + int(round(float(row["b_dt_s"]) * 1e6))


def group_same_key(rows: list, gap_s: float) -> list:
    by = {}
    for r in rows:
        if r.get("b_cx_odom", "") in ("", "nan"):
            continue
        by.setdefault(r["key"], []).append(r)
    out = []
    for k in sorted(by):
        rs = sorted(by[k], key=pass_time_us)
        cur = [rs[0]]
        for r in rs[1:]:
            if pass_time_us(r) - pass_time_us(cur[-1]) <= gap_s * 1e6:
                cur.append(r)
            else:
                out.append(cur)
                cur = [r]
        out.append(cur)
    return out


def _f(r, k, default=float("nan")):
    v = r.get(k, "")
    return float(v) if v not in ("", None) else default


def make_obs(group: list, cfg: dict) -> Obs:
    o = cfg["obs"]
    rep = min(group, key=lambda r: abs(_f(r, "b_dt_s", 0.0)))
    partial = rep["lower_only"] == "1" or rep["upper_only"] == "1" or rep["single_scan"] == "1" or rep["valid"] == "0"
    v, yaw = _f(rep, "v_mps"), math.radians(_f(rep, "yaw_odom_deg", 0.0))
    v0 = np.array([v * math.cos(yaw), v * math.sin(yaw)]) if rep.get("yaw_source") == "motion" and np.isfinite(v) and v <= o["v0_max"] else np.zeros(2)
    return Obs(obs_id="", t_us=int(np.median([pass_time_us(r) for r in group])), keys=sorted({r["key"] for r in group}),
               frames=sorted(int(r["frame"]) for r in group), rows=list(group),
               x=float(np.median([_f(r, "b_cx_odom") for r in group])), y=float(np.median([_f(r, "b_cy_odom") for r in group])),
               z_c=float(np.median([_f(r, "b_cz_odom", 0.0) for r in group])), sigma=o["sigma_partial"] if partial else o["sigma_full"],
               quality="partial" if partial else "full", range_m=float(math.hypot(_f(rep, "cx", 0.0), _f(rep, "cy", 0.0))),
               box=dict(l=_f(rep, "l"), w=_f(rep, "w"), h=_f(rep, "h"), yaw_deg=_f(rep, "yaw_odom_deg", 0.0)),
               bbox=dict(cx=_f(rep, "b_cx_odom"), cy=_f(rep, "b_cy_odom"), l=_f(rep, "b_l"), w=_f(rep, "b_w"), yaw_deg=_f(rep, "yaw_odom_deg", 0.0)),
               v0=v0, n_pts=int(_f(rep, "n_pts", 0)))


def _rect(b: dict) -> np.ndarray:
    c, s = math.cos(math.radians(b["yaw_deg"])), math.sin(math.radians(b["yaw_deg"]))
    hl, hw = max(b["l"], 1e-3) / 2, max(b["w"], 1e-3) / 2
    loc = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    return (loc @ np.array([[c, s], [-s, c]]) + [b["cx"], b["cy"]]).astype(np.float32)


def bev_overlap_frac(a: dict, b: dict) -> float:
    """Overlap area of two rotated BEV rectangles / area of the smaller one."""
    if not all(np.isfinite([a["cx"], a["cy"], a["l"], a["w"], b["cx"], b["cy"], b["l"], b["w"]])):
        return 0.0
    inter, _ = cv2.intersectConvexConvex(_rect(a), _rect(b))
    small = min(max(a["l"], 1e-3) * max(a["w"], 1e-3), max(b["l"], 1e-3) * max(b["w"], 1e-3))
    return float(max(inter, 0.0) / small)


def point_share_frac(pa: dict, pb: dict, tol: float) -> float:
    """Points of pa that have a point of pb in the same scan within tol / size of the smaller set."""
    na, nb = len(pa["xyz"]), len(pb["xyz"])
    if na == 0 or nb == 0:
        return 0.0
    sa, sb = np.asarray(pa["scan_id"]).astype(int), np.asarray(pb["scan_id"]).astype(int)
    shared = 0
    for s in np.intersect1d(sa, sb):
        d, _ = cKDTree(np.asarray(pb["xyz"], float)[sb == s]).query(np.asarray(pa["xyz"], float)[sa == s])
        shared += int((d <= tol).sum())
    return float(min(shared, min(na, nb)) / min(na, nb))


def merge_cross_key(obs: list, points_of, cfg: dict) -> tuple:
    o = cfg["obs"]
    obs = sorted(obs, key=lambda q: q.t_us)
    events, merged_into = [], {}
    for i in range(len(obs)):
        for j in range(i + 1, len(obs)):
            a, b = obs[i], obs[j]
            if b.t_us - a.t_us > o["same_key_gap_s"] * 1e6:
                break
            if id(a) in merged_into or id(b) in merged_into or set(a.keys) & set(b.keys):
                continue
            overlap = bev_overlap_frac(a.bbox, b.bbox)
            if overlap <= 0 and math.hypot(a.x - b.x, a.y - b.y) > o["dup_candidate_m"]:
                continue                                          # two different people in the same pass: not a duplicate candidate
            pa, pb = points_of(a), points_of(b)
            share = point_share_frac(pa, pb, o["merge_nn_m"]) if pa is not None and pb is not None else float("nan")
            if np.isfinite(share) and share >= o["merge_point_share"] and overlap >= o["merge_box_overlap"]:
                merged_into[id(b)] = a
                rows = a.rows + b.rows
                m = make_obs(rows, cfg)
                a.__dict__.update({k: v for k, v in m.__dict__.items() if k != "obs_id"})
            else:
                events.append(dict(kind="dup_unmerged", obs_a=a, obs_b=b, share=share, overlap=overlap))
    return [q for q in obs if id(q) not in merged_into], events


def build_observations(rows: list, points_of, cfg: dict) -> tuple:
    """-> (observations sorted by time with ids o0, o1, ..., events, row_obs {(frame, key): obs_id})."""
    obs = [make_obs(g, cfg) for g in group_same_key(rows, cfg["obs"]["same_key_gap_s"])]
    obs, events = merge_cross_key(obs, points_of, cfg)
    obs.sort(key=lambda q: (q.t_us, q.keys))
    row_obs = {}
    for k, q in enumerate(obs):
        q.obs_id = f"o{k}"
        for r in q.rows:
            row_obs[(int(r["frame"]), r["key"])] = q.obs_id
    for e in events:
        e["obs_a"], e["obs_b"] = e["obs_a"].obs_id, e["obs_b"].obs_id
    return obs, events, row_obs
