"""Observation opportunity of a track (design §4): did a nodding-LiDAR scan sweep the predicted person volume, and was the person
where the v3 detector could have seen it (camera FOV, <= det_range)?

classify_scan: returns inside the angular window of the person volume (horizontal radius min(core_r + sigma, core_r_max), height
z_lo..z_hi above the floor) seen from the sensor at the scan time, split by horizontal range: at the predicted distance (hit), beyond
(behind = the ray went through the volume), before it (front = occluded). H: >= min_rays hits; b: >= min_rays behind from
>= min_rings rings (strong if sigma <= sigma_strong); f: >= min_rays front; '.': nothing (not swept or no return).
Rays without a return (glass, dark clothes, out of range) cannot be seen: they look like '.', i.e. no opportunity.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from pipeline.detect.project import fov_mask, project_points
from pipeline.detect.scans import ScanStore


def vis_store(run, pose, masks, cfg_det: dict, far_range: float) -> ScanStore:
    """Same preprocessing as the detector (masks, per-point time, odometry frame) without the 10 m limit."""
    c = copy.deepcopy(cfg_det)
    c["scan_prep"]["det_max_range"] = float(far_range)
    return ScanStore(run, pose, masks, c)


def scans_between(run, t0_us: int, t1_us: int) -> list:
    """Scan ids whose start time is in (t0, t1]."""
    ts = np.asarray(run.lidar_ts)
    return [int(i) for i in np.nonzero((ts > t0_us) & (ts <= t1_us))[0]]


def _to_lv(X, plane):
    lv = ((np.atleast_2d(np.asarray(X, float)) - plane["p_c"]) @ plane["R_c"]) @ plane["R_level"].T
    lv[:, 2] -= plane["ground_z"]
    return lv


def classify_scan(scan: dict, pose, centre_odom, frame_plane: dict, sigma_pos: float, vcfg: dict) -> dict:
    tp = scan["t_point_us"]
    t_s = int(np.median(tp)) if len(tp) else 0
    out = dict(cls=".", strong=False, n_hit=0, n_behind=0, n_front=0, rings_behind=0, t_us=t_s)
    if len(tp) == 0:
        return out
    P = _to_lv(scan["xyz_odom"], frame_plane)
    s = _to_lv(pose.odo.at(t_s), frame_plane)[0]
    c = _to_lv(centre_odom, frame_plane)[0]
    d = math.hypot(c[0] - s[0], c[1] - s[1])
    r = min(vcfg["core_r"] + sigma_pos, vcfg["core_r_max"])
    dx, dy = P[:, 0] - s[0], P[:, 1] - s[1]
    rho = np.hypot(dx, dy)
    daz = np.abs((np.arctan2(dy, dx) - math.atan2(c[1] - s[1], c[0] - s[0]) + np.pi) % (2 * np.pi) - np.pi)
    el = np.arctan2(P[:, 2] - s[2], np.maximum(rho, 1e-6))
    el_lo, el_hi = math.atan2(vcfg["z_lo"] - s[2], max(d, 1e-6)), math.atan2(vcfg["z_hi"] - s[2], max(d, 1e-6))
    win = (daz <= math.atan2(r, max(d, 1e-6))) & (el >= el_lo) & (el <= el_hi)
    hit = win & (np.abs(rho - d) <= vcfg["hit_tol"])
    behind = win & (rho > d + vcfg["hit_tol"])
    front = win & (rho < d - vcfg["hit_tol"])
    rings = len(np.unique(np.asarray(scan["ring"])[behind]))
    out.update(n_hit=int(hit.sum()), n_behind=int(behind.sum()), n_front=int(front.sum()), rings_behind=int(rings))
    if hit.sum() >= vcfg["min_rays"]:
        out["cls"] = "H"
    elif behind.sum() >= vcfg["min_rays"] and rings >= vcfg["min_rings"]:
        out.update(cls="b", strong=bool(sigma_pos <= vcfg["sigma_strong"]))
    elif front.sum() >= vcfg["min_rays"]:
        out["cls"] = "f"
    return out


def group_passes(results: list, gap_s: float) -> list:
    """Runs of swept scans (H or b) whose times are at most gap_s apart -> [dict(t0_us, t1_us, has_H, strong_b, weak_b, n_scans)]."""
    sw = sorted((r for r in results if r["cls"] in ("H", "b")), key=lambda r: r["t_us"])
    out, cur = [], []
    for r in sw:
        if cur and r["t_us"] - cur[-1]["t_us"] > gap_s * 1e6:
            out.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return [dict(t0_us=g[0]["t_us"], t1_us=g[-1]["t_us"], has_H=any(r["cls"] == "H" for r in g),
                 strong_b=any(r["cls"] == "b" and r["strong"] for r in g), weak_b=any(r["cls"] == "b" and not r["strong"] for r in g),
                 n_scans=len(g)) for g in out]


def in_detect_region(geom, pose, cfg_det: dict, centre_odom, t_us: int, det_range: float) -> bool:
    R, p = pose.at(int(t_us))
    dev = (np.asarray(centre_odom, float) - p) @ R
    cam, yaw, pitch, depth, u, v = project_points(geom, dev[None])
    return bool(fov_mask(cam, pitch, depth, u, cfg_det)[0]) and math.hypot(dev[0], dev[1]) <= det_range


def pass_outcome(p: dict, opportunity: bool, observed: bool) -> str:
    if observed:
        return "observed"
    if not opportunity:
        return "no_opportunity"
    if p["has_H"]:
        return "present_unconfirmed"
    if p["strong_b"]:
        return "absent_strong"
    if p["weak_b"]:
        return "absent_weak"
    return "no_opportunity"


class TrackPasses:
    """Scan classifications of one track; passes are handed out once a later classified scan shows they ended."""

    def __init__(self):
        self.results = []
        self.t_last_scan = None

    def add(self, results: list) -> None:
        self.results += results
        if results:
            self.t_last_scan = max(r["t_us"] for r in results) if self.t_last_scan is None else max(self.t_last_scan, *(r["t_us"] for r in results))

    def pop_finished(self, t_now_us: int, gap_s: float) -> list:
        """Passes whose last swept scan is more than gap_s before both t_now and the latest classified scan."""
        ps = group_passes(self.results, gap_s)
        lim = min(t_now_us, self.t_last_scan if self.t_last_scan is not None else t_now_us) - gap_s * 1e6
        done = [p for p in ps if p["t1_us"] < lim]
        if done:
            t_cut = done[-1]["t1_us"]
            self.results = [r for r in self.results if r["t_us"] > t_cut]
        return done
