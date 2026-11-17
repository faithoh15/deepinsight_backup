"""Stage pose: map-only pose correction -> 3_map/pose_refine/ (3d_detection/scripts/refine_poses.py, verbatim; main() became run();
paths of this recording instead of the dataset table; settings = configs/pipeline.yaml pose (= 3d_detection/configs/pose_refine.yaml)).

Nodes every 2 s (submap of the scans 0.7-40 m around each node, tracked-person points removed), sequential 6-DOF point-to-plane ICP
edges, a weak legacy prior, gravity priors and loop closures (Cauchy weights, outliers dropped) solved as one sparse least-squares graph;
the correction is interpolated per point time. Outputs: pose_correction.csv, pose_edges.csv, consistency.csv (held-out blocks, a
description), trajectory_top.png, summary.json.
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from pipeline.common.io import write_csv
from pipeline.common.pose import PoseInterp
from pipeline.common.recording import Run
from pipeline.mapping.pose_refine import (Edge, PoseCorrection, _Y, edge_whitener, icp_edge, loop_candidates,
                                          solve_graph)
from pipeline.mapping.static_map import build_tubes, in_tubes, load_map_config, obs_from_csv, read_csv
from pipeline.prep.clock import read_clock
from pipeline.track.track_obs import load_track_config
from pipeline.track.track_vis import plane_lookup
from stages._base import stage_main


EDGE_FIELDS = ["i", "j", "kind", "t_i_s", "t_j_s", "overlap", "rmse", "n_corr", "accepted", "reason", "whitened_after"]
GAP_BINS = [(3.9, 4.1), (4.1, 12), (12, 30), (30, 60), (60, 120), (120, 1e9)]


class PersonFilter:
    """Points inside a tracked person's cylinder at their own time (build_map's removal test, legacy frame)."""

    def __init__(self, det, track, pose):
        fr = [r for r in read_csv(os.path.join(det, "frames.csv")) if r.get("t_img_us")]
        self.f_ids = np.array([int(r["frame"]) for r in fr])
        self.f_t = np.array([int(r["t_img_us"]) for r in fr], np.int64)
        self.plane_of = plane_lookup(fr, pose)
        self.mcfg = load_map_config()
        self.tubes = build_tubes(obs_from_csv(read_csv(os.path.join(track, "observations.csv"))), load_track_config(), self.mcfg)
        self.n_excluded = self.n_used = 0

    def person(self, X, t):
        if not len(X) or not self.tubes:
            return np.zeros(len(X), bool)
        pl = self.plane_of(int(self.f_ids[np.abs(self.f_t - int(np.median(t))).argmin()]))
        return in_tubes(self.tubes, X, t, pl, self.mcfg) >= 0


def scan_points(run, pose, i, rmin, rmax, pf=None, count=False):
    s = run.load_scan(i)
    k = (s["range"] >= rmin) & (s["range"] <= rmax)
    tp = s["t_point_us"][k]
    X = np.einsum("nij,nj->ni", pose.gyro.at(tp), s["xyz"][k]) + pose.odo.at(tp)
    if pf is not None:
        keep = ~pf.person(X, tp)
        if count:
            pf.n_excluded += int((~keep).sum())
            pf.n_used += int(keep.sum())
        X, tp = X[keep], tp[keep]
    return X, tp


def cloud(P, scfg):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(scfg["voxel"])
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=scfg["normal_radius"], max_nn=scfg["normal_nn"]))
    return pc


def up_direction(run, pose, t0, t1):
    """Mean accelerometer direction over [t0, t1) rotated into the legacy frame sample by sample (= up while walking averages out)."""
    m = (run.imu_ts >= t0) & (run.imu_ts < t1)
    if m.sum() < 10:
        return None
    v = np.einsum("nij,nj->ni", pose.gyro.at(run.imu_ts[m]), run.acc[m]).mean(0)
    return v / np.linalg.norm(v)


def misalign(src, dst, radii):
    reg = o3d.pipelines.registration
    T = np.eye(4)
    for d in radii:
        r = reg.registration_icp(src, dst, d, T, reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=d)),
                                 reg.ICPConvergenceCriteria(max_iteration=30))
        T = np.asarray(r.transformation)
    S = np.asarray(src.points)
    disp = np.linalg.norm(S @ T[:3, :3].T + T[:3, 3] - S, axis=1)
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(T[:3, :3]).as_rotvec()))), float(np.median(disp)), float(r.fitness)


def check_consistency(run, pose, corr, t_scan, lo, hi, ccfg, scfg, pf=None):
    """Held-out blocks: misalignment of pairs before (legacy) / after (corrected), same pairs."""
    t0, t1 = t_scan[lo], t_scan[hi - 1]
    blocks = []
    for tb in np.arange(t0 + ccfg["offset_s"] * 1e6, t1 - ccfg["block_s"] * 1e6, ccfg["block_s"] * 1e6):
        idx = [i for i in range(lo, hi) if tb <= t_scan[i] < tb + ccfg["block_s"] * 1e6]
        if not idx:
            continue
        P, T = zip(*[scan_points(run, pose, i, scfg["min_range"], 15.0, pf) for i in idx[::2]])
        P, T = np.concatenate(P), np.concatenate(T)
        Pc = corr.apply(P, T, pose.odo.at(T))
        tm = int(tb + ccfg["block_s"] * 5e5)
        dev = pose.odo.at(tm)
        blocks.append(dict(t=(tm - t0) / 1e6, before=cloud(P, dict(scfg, voxel=0.1, normal_radius=0.4)),
                           after=cloud(Pc, dict(scfg, voxel=0.1, normal_radius=0.4)),
                           c=corr.apply(dev[None], np.array([tm]), dev[None])[0]))
    rows = []
    radii = [1.0, 0.6, 0.3, 0.15]
    for j in range(len(blocks)):
        for i in range(j):
            gap = blocks[j]["t"] - blocks[i]["t"]
            if gap < 3.9 or np.linalg.norm(blocks[j]["c"] - blocks[i]["c"]) > ccfg["max_dist_m"]:
                continue
            ab, db, fb = misalign(blocks[j]["before"], blocks[i]["before"], radii)
            aa, da, fa = misalign(blocks[j]["after"], blocks[i]["after"], radii)
            if min(fb, fa) < 0.4:
                continue
            rows.append(dict(t_i_s=round(blocks[i]["t"], 1), t_j_s=round(blocks[j]["t"], 1), gap_s=round(gap, 1), angle_before_deg=round(ab, 3),
                             shift_before_m=round(db, 4), angle_after_deg=round(aa, 3), shift_after_m=round(da, 4)))
    table = []
    for a_, b_ in GAP_BINS:
        r = [x for x in rows if a_ <= x["gap_s"] < b_]
        if r:
            md = lambda k: round(float(np.median([x[k] for x in r])), 3)  # noqa: E731
            table.append(dict(gap=f"{a_:.0f}-{b_:.0f}s" if b_ < 1e8 else f">{a_:.0f}s", n=len(r), angle_before=md("angle_before_deg"),
                              angle_after=md("angle_after_deg"), shift_before=md("shift_before_m"), shift_after=md("shift_after_m")))
    return rows, table


def draw_traj(path, Pl, Pc, title):
    allp = np.vstack([Pl, Pc])[:, :2]
    lo_, hi_ = allp.min(0) - 2, allp.max(0) + 2
    s = min(20.0, 3000 / max(hi_ - lo_))
    W, H = int((hi_[0] - lo_[0]) * s), int((hi_[1] - lo_[1]) * s) + 40
    img = np.full((H, W, 3), 18, np.uint8)
    uv = lambda P: np.stack([(P[:, 0] - lo_[0]) * s, H - (P[:, 1] - lo_[1]) * s], 1).astype(int)  # noqa: E731
    cv2.polylines(img, [uv(Pl)], False, (120, 120, 120), 1)
    q = uv(Pc)
    col = cv2.applyColorMap(np.linspace(0, 255, len(q)).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    for k in range(len(q) - 1):
        cv2.line(img, tuple(q[k]), tuple(q[k + 1]), tuple(int(v) for v in col[k]), 2)
    cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(path, img)


def run(P, cfg, scans=None, no_check=False, **kw):
    """scans = "START:STOP" scan indices (default all); no_check skips the held-out consistency check."""
    a = type("A", (), dict(dataset=P.name, scans=scans, no_check=no_check))
    t_start = time.time()
    cfg = cfg["pose"]
    scfg, icfg, gcfg, lcfg = cfg["submap"], cfg["icp"], cfg["graph"], cfg["loop"]
    det = P.detection
    out = P.pose
    os.makedirs(out, exist_ok=True)
    run = Run(P.data, read_clock(P.clock))
    pose = PoseInterp(run, os.path.join(P.odometry, "odometry.csv"))
    pf = PersonFilter(det, P.tracking, pose)
    t_scan = run.lidar_ts
    lo, hi = map(int, a.scans.split(":")) if a.scans else (0, run.n_scans())
    hi = min(hi, run.n_scans())
    half = int(scfg["window_s"] * 5e5)

    # ---- nodes and submaps
    t_nodes = np.arange(t_scan[lo] + half, t_scan[hi - 1] + half, 2 * half, dtype=np.int64)
    sub, ups, keep_t = [], [], []
    for tk in t_nodes:
        idx = [i for i in range(lo, hi) if tk - half <= t_scan[i] < tk + half]
        if not idx:
            continue
        P = np.concatenate([scan_points(run, pose, i, scfg["min_range"], scfg["max_range"], pf, count=True)[0] for i in idx])
        ck = pose.odo.at(int(tk))
        sub.append(dict(t=int(tk), c=ck, pc=cloud(P - ck, scfg), n_scans=len(idx)))
        ups.append(up_direction(run, pose, tk - half, tk + half))
        keep_t.append(int(tk))
    K = len(sub)
    c = np.array([s["c"] for s in sub])
    t_s = (np.array(keep_t) - keep_t[0]) / 1e6
    up = np.array(ups) if all(u is not None for u in ups) else None
    print(f"{K} nodes, submaps built ({time.time() - t_start:.0f} s)", flush=True)

    # ---- sequential ICP edges + legacy prior
    Wp = np.diag([1 / np.radians(gcfg["prior_sigma_rot_deg"])] * 3 + [1 / gcfg["prior_sigma_t_m"]] * 3)
    edges, log = [], []
    for k in range(K - 1):
        T0 = np.eye(4)
        T0[:3, 3] = c[k + 1] - c[k]
        edges.append(Edge(k, k + 1, T0, Wp, False, "prior"))
    for g in gcfg["seq_gaps"]:
        for k in range(K - g):
            T0 = np.eye(4)
            T0[:3, 3] = c[k + g] - c[k]
            T, ok, st = icp_edge(sub[k + g]["pc"], sub[k]["pc"], T0, icfg)
            log.append(dict(i=k, j=k + g, kind="icp", t_i_s=round(t_s[k], 1), t_j_s=round(t_s[k + g], 1), overlap=round(st["overlap"], 3),
                            rmse=round(st["rmse"], 4), n_corr=st["n_corr"], accepted=int(ok), reason="" if ok else "overlap/rmse"))
            if ok:
                edges.append(Edge(k, k + g, T, edge_whitener(st["info"], gcfg["icp_n_eff"], gcfg["icp_sigma_p"]), False, "icp"))
    rv, u, st = solve_graph(c, edges, up, gcfg)
    print(f"sequential: {sum(e.kind == 'icp' for e in edges)} ICP edges ({time.time() - t_start:.0f} s)", flush=True)

    # ---- loop closure rounds
    tried = set()
    outliers = []
    for rnd in range(lcfg["rounds"]):
        cand = loop_candidates(t_s, c + u, lcfg["min_gap_s"], lcfg["max_dist_m"], lcfg["max_per_node"], tried)
        n_new = 0
        for i, j in cand:
            tried.add((i, j))
            T0 = np.linalg.inv(_Y(rv[i], u[i], c[i])) @ _Y(rv[j], u[j], c[j])
            T, ok, s_ = icp_edge(sub[j]["pc"], sub[i]["pc"], T0, icfg)
            d = np.linalg.inv(T0) @ T
            jump_m = float(np.linalg.norm(d[:3, 3]))
            jump_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec())))
            reason = "" if ok else "overlap/rmse"
            if ok and (jump_m > lcfg["max_jump_m"] or jump_deg > lcfg["max_jump_deg"]):
                ok, reason = False, f"jump {jump_m:.2f} m {jump_deg:.1f} deg"
            log.append(dict(i=i, j=j, kind=f"loop{rnd + 1}", t_i_s=round(t_s[i], 1), t_j_s=round(t_s[j], 1), overlap=round(s_["overlap"], 3),
                            rmse=round(s_["rmse"], 4), n_corr=s_["n_corr"], accepted=int(ok), reason=reason))
            if ok:
                edges.append(Edge(i, j, T, edge_whitener(s_["info"], gcfg["icp_n_eff"], gcfg["icp_sigma_p"]), True, "loop"))
                n_new += 1
        rv, u, st = solve_graph(c, edges, up, gcfg, x0=st["x"])
        outliers = st["outliers"]
        print(f"loop round {rnd + 1}: {len(cand)} candidates, {n_new} accepted, outliers {len(outliers)} ({time.time() - t_start:.0f} s)", flush=True)
    for r in log:
        key = [k for k in st["edge_norm"] if k[0] == r["i"] and k[1] == r["j"] and (k[2] == r["kind"] or (k[2] == "loop" and r["kind"].startswith("loop")))]
        r["whitened_after"] = round(st["edge_norm"][key[0]], 2) if key and r["accepted"] else ""
        if r["accepted"] and (r["i"], r["j"]) in outliers and r["kind"].startswith("loop"):
            r["reason"], r["whitened_after"] = "dropped as outlier by the graph", ""

    corr = PoseCorrection(np.array(keep_t), c, rv, u)
    corr.to_csv(os.path.join(out, "pose_correction.csv"))
    write_csv(os.path.join(out, "pose_edges.csv"), log, EDGE_FIELDS)

    # ---- trajectory figure
    ts_all = t_scan[lo:hi]
    Pl = pose.odo.at(ts_all)
    Pc = corr.apply(Pl, ts_all, Pl)
    name = a.dataset.split("_")[-1]
    draw_traj(os.path.join(out, "trajectory_top.png"), Pl, Pc, f"{name} device path: grey = legacy, colour = corrected (time)")

    ang = np.degrees(np.linalg.norm(rv, axis=1))
    summary = dict(dataset=a.dataset, scans=[lo, hi], n_nodes=K, config=cfg, points=dict(used=pf.n_used, person_excluded=pf.n_excluded),
                   edges=dict(prior=sum(e.kind == "prior" for e in edges), icp=sum(e.kind == "icp" for e in edges),
                              icp_rejected=sum(1 for r in log if r["kind"] == "icp" and not r["accepted"]),
                              loop_tried=sum(1 for r in log if r["kind"].startswith("loop")),
                              loop_accepted=sum(e.kind == "loop" for e in edges) - len(outliers), loop_dropped_outlier=len(outliers)),
                   correction=dict(max_rot_deg=round(float(ang.max()), 3), end_rot_deg=round(float(ang[-1]), 3),
                                   max_shift_m=round(float(np.linalg.norm(u, axis=1).max()), 3), end_shift_m=round(float(np.linalg.norm(u[-1])), 3)),
                   up=None if st["up"] is None else [round(float(v), 4) for v in st["up"]])
    if not a.no_check:
        rows, table = check_consistency(run, pose, corr, t_scan, lo, hi, cfg["check"], scfg, pf)
        write_csv(os.path.join(out, "consistency.csv"), rows, list(rows[0]) if rows else ["gap_s"])
        summary["consistency"] = table
    summary["seconds"] = round(time.time() - t_start, 1)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps({k: summary[k] for k in ("n_nodes", "points", "edges", "correction") if k in summary}, ensure_ascii=False))
    for r in summary.get("consistency", []):
        print(r)
    return {k: summary[k] for k in ("n_nodes", "points", "edges", "correction", "seconds")}



if __name__ == "__main__":
    stage_main(run)
