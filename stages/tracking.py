"""Stage tracking: 3D person tracking of the v3 boxes -> 2_tracking/ (3d_detection/scripts/run_tracking.py, verbatim; main() became
run(); the dataset table / legacy folder lookup was replaced by the paths of this recording; settings = configs/pipeline.yaml tracking
(= 3d_detection/configs/track.yaml) and detection; tracking video = run.tracking_video).

Per pass observations of the v3 boxes -> constant-velocity Kalman tracker with observation-opportunity passes (LiDAR sweep + camera
field of view) -> RTS-smoothed per-frame state and speed. Outputs: observations.csv, track_frames.csv (per frame: id, position, speed and
its 95 % band), rows.csv (every v3 box -> 3D track id), tracks.csv (per track summary), events.csv, summary.json, vis_det/ (the v3 boxes
of each frame with T<id> and speed).
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np

from pipeline.common.cam_geom import CamGeom
from pipeline.common.io import read_ply_fields, write_csv
from pipeline.common.pose import PoseInterp
from pipeline.common.recording import Run
from pipeline.detect.scans import load_masks
from pipeline.prep.clock import read_clock
from pipeline.track.track_kf import cv_F, cv_Q
from pipeline.track.track_obs import build_observations
from pipeline.track.track_post import consistency_checks, smooth_track, summarize_track
from pipeline.track.tracker import Tracker
from pipeline.track.track_vis import render
from pipeline.track.visibility import TrackPasses, classify_scan, in_detect_region, scans_between, vis_store
from stages._base import stage_main

OBS_FIELDS = ["obs_id", "t_us", "keys", "frames", "x", "y", "z_c", "sigma", "quality", "range_m", "l", "w", "h", "yaw_deg", "v0x", "v0y", "n_pts",
              "track_id", "d2", "v_impl", "reason"]
FRAME_FIELDS = ["frame", "t_img_us", "track_id", "status", "segment", "has_det", "keys", "obs_id", "x", "y", "vx", "vy", "speed", "speed_lo95",
                "speed_hi95", "sigma_x", "sigma_y", "sigma_vx", "sigma_vy", "sigma_speed", "speed_filt", "v_pass_mps", "vis_last"]
ROW_FIELDS = ["frame", "key", "obs_id", "track_id", "track_confirmed", "reason"]
TRACK_FIELDS = ["track_id", "status_final", "confirmed", "t_first_us", "t_last_us", "frame_first", "frame_last", "n_obs", "keys", "n_obs_frames",
                "n_interp_frames", "n_pred_frames", "speed_med", "speed_p90", "speed_max", "end_reason", "n_merge_2d", "n_split_2d"]
EVENT_FIELDS = ["t_us", "frame", "kind", "track_id", "obs_id", "detail"]


def state_at(kf, t_us, sigma_a):
    """Position and covariance of a track state at t (forwards or backwards in time; the track itself is not changed)."""
    dt = (int(t_us) - kf.t_us) / 1e6
    F = cv_F(dt)
    return F @ kf.x, F @ kf.P @ F.T + cv_Q(abs(dt), sigma_a)




def run(P, cfg, frames=None, no_vis=False, **kw):
    """frames = "START:STOP" camera frames (default all)."""
    a = type("A", (), dict(frames=frames, no_vis=no_vis, video=cfg["run"]["tracking_video"]))
    t0 = time.time()
    cfg_det, tcfg = cfg["detection"], cfg["tracking"]
    ds = dict(name=P.name, data=P.data, sidebyside=P.sidebyside)
    det = P.detection
    out = P.tracking
    os.makedirs(out, exist_ok=True)
    lo, hi = map(int, a.frames.split(":")) if a.frames else (-10 ** 9, 10 ** 9)
    rows = [r for r in csv.DictReader(open(os.path.join(det, "persons.csv"))) if lo <= int(r["frame"]) < hi]
    fr_all = [r for r in csv.DictReader(open(os.path.join(det, "frames.csv"))) if r.get("t_img_us")]
    fr_rows = {int(r["frame"]): r for r in fr_all if lo <= int(r["frame"]) < hi}
    ok_planes = sorted((int(r["t_img_us"]), r) for r in fr_all if r.get("R00"))
    run = Run(P.data, read_clock(P.clock))
    pose = PoseInterp(run, os.path.join(P.odometry, "odometry.csv"))
    geom = CamGeom(P.data)
    masks = load_masks(P.masks)
    vc, sa = tcfg["vis"], tcfg["kf"]["sigma_a"]

    def plane_of(frame):
        """Frame ground plane (the nearest frame with a plane if this frame had none) with the device pose at the frame time."""
        t = int(fr_rows[frame]["t_img_us"]) if frame in fr_rows else int(ok_planes[0][0])
        src = fr_rows.get(frame) if frame in fr_rows and fr_rows[frame].get("R00") else min(ok_planes, key=lambda e: abs(e[0] - t))[1]
        R_c, p_c = pose.at(t)
        return dict(R_level=np.array([[float(src[f"R{i}{j}"]) for j in range(3)] for i in range(3)]), ground_z=float(src["ground_z"]), R_c=R_c, p_c=p_c)

    # ---------------------------------------------------------------- observations
    pts_cache = {}

    def row_points(r):
        k = (int(r["frame"]), r["key"])
        if k not in pts_cache:
            path = os.path.join(det, "clouds", f"{k[0]:04d}_{k[1]}.ply")
            if not os.path.exists(path):
                pts_cache[k] = None
            else:
                d = read_ply_fields(path)
                m = np.asarray(d["in_main"]) == 1
                pl = plane_of(k[0])
                lv = np.stack([d["x"], d["y"], d["z"]], 1).astype(float)[m]
                odom = ((lv + [0, 0, pl["ground_z"]]) @ pl["R_level"]) @ pl["R_c"].T + pl["p_c"]
                pts_cache[k] = dict(xyz=odom, scan_id=np.asarray(d["scan_id"])[m].astype(int))
        return pts_cache[k]

    def points_of(o):
        ps = [row_points(r) for r in o.rows]
        if any(p is None for p in ps):
            return None
        return dict(xyz=np.concatenate([p["xyz"] for p in ps]), scan_id=np.concatenate([p["scan_id"] for p in ps]))

    obs, dup_events, row_obs = build_observations(rows, points_of, tcfg)

    # ---------------------------------------------------------------- tracking with the observation-opportunity passes
    store = vis_store(run, pose, masks, cfg_det, vc["far_range"])
    passes_of = {}

    def vis_fn(trk, t_prev, t, frame):
        ids = scans_between(run, t_prev, t)
        if not ids:
            return None                                           # LiDAR gap: cannot judge -> time rule
        pl = plane_of(frame)
        res = []
        for i in ids:
            try:
                s = store.get(i)
            except Exception:                                     # unreadable scan: cannot judge
                return None
            ts = int(np.median(s["t_point_us"])) if len(s["t_point_us"]) else int(run.lidar_ts[i]) + 25_000
            x, P = state_at(trk.kf, ts, sa)
            sig = float(np.sqrt(max(np.linalg.eigvalsh(P[:2, :2]).max(), 0.0)))
            res.append(classify_scan(s, pose, np.array([x[0], x[1], trk.z_c]), pl, sig, vc))
        tp = passes_of.setdefault(trk.id, TrackPasses())
        tp.add(res)
        done = tp.pop_finished(t, vc["pass_gap_s"])
        for p in done:
            tm = (p["t0_us"] + p["t1_us"]) // 2
            x, _ = state_at(trk.kf, tm, sa)
            p["opportunity"] = in_detect_region(geom, pose, cfg_det, np.array([x[0], x[1], trk.z_c]), tm, vc["det_range"])
        return done

    ticks = sorted((f, int(r["t_img_us"])) for f, r in fr_rows.items())
    tr = Tracker(tcfg, vis_fn)
    tr.run(obs, ticks)

    # ---------------------------------------------------------------- per-frame smoothed output
    obs_by_id = {o.obs_id: o for o in obs}
    confirmed = {e["track_id"] for e in tr.events if e["kind"] == "confirm"}
    t_of = dict(ticks)
    row_by_fk = {(int(r["frame"]), r["key"]): r for r in rows}
    pass_ev = {}
    for e in tr.events:
        if e["kind"] == "pass":
            pass_ev.setdefault(e["track_id"], []).append((e["t_us"], e["detail"].split(";")[0].split("=")[1]))
    frame_rows, track_rows = [], []
    for trk in sorted(tr.tracks.values(), key=lambda t: t.id):
        alive = dict((f, st) for f in sorted(tr.alive_at) for i, st in tr.alive_at[f] if i == trk.id)
        for f in {f for o in trk.obs_ids for f in obs_by_id[o].frames if f in t_of}:     # frames whose image precedes the first pass time
            alive.setdefault(f, "tentative")
        alive = sorted(alive.items())
        sm = smooth_track([obs_by_id[o] for o in trk.obs_ids], [(f, t_of[f]) for f, _ in alive], tcfg)
        mine = []
        for (f, st), r in zip(alive, sm):
            keys = [k for (ff, k), oid in row_obs.items() if ff == f and tr.obs_track.get(oid) == trk.id]
            pe = [o for t, o in pass_ev.get(trk.id, []) if t <= r["t_img_us"]]
            r.update(track_id=trk.id, status=st, has_det=int(bool(keys)), keys="|".join(sorted(keys)), obs_id=row_obs.get((f, keys[0]), "") if keys else "",
                     v_pass_mps=float(row_by_fk[(f, keys[0])]["v_mps"]) if keys and row_by_fk[(f, keys[0])]["v_mps"] not in ("", "nan") else float("nan"),
                     vis_last=pe[-1] if pe else "")
            mine.append(r)
        frame_rows += mine
        s = summarize_track(trk, mine, tr.events)
        s["confirmed"] = int(trk.id in confirmed)
        track_rows.append(s)
    frame_rows.sort(key=lambda r: (r["frame"], r["track_id"]))

    # ---------------------------------------------------------------- files
    write_csv(os.path.join(out, "observations.csv"), [dict(obs_id=o.obs_id, t_us=o.t_us, keys="|".join(o.keys), frames="|".join(map(str, o.frames)), x=o.x, y=o.y,
              z_c=o.z_c, sigma=o.sigma, quality=o.quality, range_m=o.range_m, l=o.box["l"], w=o.box["w"], h=o.box["h"], yaw_deg=o.box["yaw_deg"], v0x=float(o.v0[0]),
              v0y=float(o.v0[1]), n_pts=o.n_pts, track_id=tr.obs_track.get(o.obs_id, -1), **tr.obs_info.get(o.obs_id, {})) for o in obs], OBS_FIELDS)
    write_csv(os.path.join(out, "track_frames.csv"), frame_rows, FRAME_FIELDS)
    write_csv(os.path.join(out, "rows.csv"), [dict(frame=f, key=k, obs_id=oid, track_id=tr.obs_track[oid], track_confirmed=int(tr.obs_track[oid] in confirmed),
              reason=tr.obs_info[oid]["reason"]) for (f, k), oid in sorted(row_obs.items())], ROW_FIELDS)
    write_csv(os.path.join(out, "tracks.csv"), track_rows, TRACK_FIELDS)
    ev = tr.events + [dict(t_us=obs_by_id[e["obs_a"]].t_us, frame=-1, kind="dup_unmerged", track_id=-1, obs_id=f"{e['obs_a']}|{e['obs_b']}",
                           detail=f"share={e['share']:.2f};overlap={e['overlap']:.2f}") for e in dup_events]
    write_csv(os.path.join(out, "events.csv"), sorted(ev, key=lambda e: e["t_us"]), EVENT_FIELDS)
    checks = consistency_checks(obs, row_obs, len(rows), tr.obs_track, tr.tracks, frame_rows)
    nis = np.array([n["nis"] for n in tr.nis]) if tr.nis else np.zeros(0)

    def q(v):
        return dict(n=int(len(v)), median=float(np.median(v)) if len(v) else None, p99=float(np.percentile(v, 99)) if len(v) else None,
                    frac_gt_9p21=float(np.mean(v > 9.21)) if len(v) else None)
    kinds = {}
    for e in ev:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    outcomes = {}
    for e in tr.events:
        if e["kind"] == "pass":
            o = e["detail"].split(";")[0].split("=")[1]
            outcomes[o] = outcomes.get(o, 0) + 1
    summary = dict(dataset=P.name, det=det, frames=a.frames, config=tcfg, n_rows=len(rows), n_obs=len(obs), n_tracks=len(tr.tracks),
                   n_tracks_confirmed=len(confirmed), n_keys=len({r["key"] for r in rows}), events=kinds, pass_outcomes=outcomes,
                   nis=dict(all=q(nis), full=q(np.array([n["nis"] for n in tr.nis if n["quality"] == "full"])),
                            partial=q(np.array([n["nis"] for n in tr.nis if n["quality"] == "partial"])),
                            near_lt4m=q(np.array([n["nis"] for n in tr.nis if n["range_m"] < 4])), far_ge4m=q(np.array([n["nis"] for n in tr.nis if n["range_m"] >= 4]))),
                   checks=checks, seconds=round(time.time() - t0, 1))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, ensure_ascii=False, default=float)
    print(json.dumps(dict(n_obs=len(obs), n_tracks=len(tr.tracks), n_tracks_confirmed=len(confirmed), events=kinds, checks=checks), ensure_ascii=False))
    if not a.no_vis and ok_planes:          # no frame with a floor plane (no 2D box anywhere): nothing to draw
        render(out, det, ds, geom, plane_of, sorted(fr_rows), os.path.join(out, "vis_det"), a.video)   # v3 boxes only, id + speed
    return dict(n_obs=len(obs), n_tracks=len(tr.tracks), n_tracks_confirmed=len(confirmed), seconds=round(time.time() - t0, 1))



if __name__ == "__main__":
    stage_main(run)
