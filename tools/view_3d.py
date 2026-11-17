#!/usr/bin/env python
"""Open3D viewer of the selective-accumulation results: accumulated person points (colour = scan time or 2D id), box_acc (thick) and
box_burst (thin), background = all candidate-scan points of the frame (recomputed on the fly), floor grid.

  python view.py <name> det [--frame N] [--scan K] [--screenshot] [--track]      (copy of 3d_detection/scripts/view_3d.py; reads
  1_detection/ and, with --track, 2_tracking/ of final/output/<name>/)
Keys: N / right = next frame, P / left = previous, J = jump (frame number in the terminal), C = colour mode (time / id), B = toggle
background, O = toggle the other components (selected points outside the boxed main component, i.e. the same person at the other
bursts), T = toggle the thin box_burst (off by default: one box_acc per 2D box), F = toggle floor points (< 0.1 m, raw),
] / [ = single-scan view: step through the scans of this frame in which a person piece was selected (background = that one scan only,
person points of that scan in full colour, the other scans dimmed / hidden with O), A = back to the accumulated view,
G = residual removal mode: before (as stored) -> after (every point inside a 3D box stays; the object's points outside the boxes are
removed: attributed points of other scans and, unless --no-predicted, predicted-position clusters of the other scans) -> diff (after +
the removed points in red = other pass / orange = trimmed / magenta = predicted), S = screenshot (<out>/screens/NNNN.png, NNNN_sKKKK.png in single-scan view, suffix _after /
_diff in the removal modes), Q = quit. Mouse: rotate / zoom / Ctrl-drag pan.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys  # noqa: F401

import numpy as np

from pipeline.common.angbox import PAL_TRACK  # noqa: E402
from pipeline.common.box_fit import EDGES, heading_arrow  # noqa: E402
from pipeline.common.cam_geom import CamGeom  # noqa: E402
from pipeline.common.config import load_config, load_datasets  # noqa: E402
from pipeline.common.io import read_ply_fields  # noqa: E402
from pipeline.common.paths import result_paths  # noqa: E402
from pipeline.common.pose import PoseInterp  # noqa: E402
from pipeline.common.recording import Run  # noqa: E402
from pipeline.detect import vis  # noqa: E402
from pipeline.detect.clean import PREDICTED, TRIMMED, ext_params, extend_frame, frame_ghosts, scan_fov_mask  # noqa: E402
from pipeline.detect.ground import fit_frame_plane, to_level  # noqa: E402
from pipeline.detect.scans import ScanStore, anchor_scan, load_masks  # noqa: E402
from pipeline.prep.clock import read_clock  # noqa: E402


def split_main(d: dict) -> tuple[dict, dict]:
    """Split the fields of one person PLY into the main component (in_main == 1, boxed) and the other components."""
    m = np.asarray(d["in_main"]) == 1
    return {k: np.asarray(v)[m] for k, v in d.items()}, {k: np.asarray(v)[~m] for k, v in d.items()}


def split_scan(d: dict, scan_id: int) -> tuple[dict, dict]:
    """Split the fields of one person PLY into the points of one scan (scan_id == scan_id) and the points of all other scans."""
    m = np.asarray(d["scan_id"]) == scan_id
    return {k: np.asarray(v)[m] for k, v in d.items()}, {k: np.asarray(v)[~m] for k, v in d.items()}


def frame_scans(cand_rows: list, frame: int) -> list:
    """Scans of one frame in which at least one person piece was selected (rows of scan_candidates.csv), in time order:
    [{scan_id, dt_cam_s, persons: {key: (n_sel, used)}}]."""
    by_scan = {}
    for r in cand_rows:
        if int(r["frame"]) != frame or r["status"] != "selected":
            continue
        e = by_scan.setdefault(int(r["scan_id"]), dict(scan_id=int(r["scan_id"]), dt_cam_s=float(r["dt_cam_s"]), persons={}))
        e["persons"][r["key"]] = (int(r["n_sel"]), int(r["used"]))
    return [by_scan[k] for k in sorted(by_scan)]


def track_ids_for_frame(rows: list, frame: int) -> dict:
    """Rows of track/rows.csv -> {key: 3D track id} for one camera frame (track id -1 = not assigned, left out)."""
    return {r["key"]: int(r["track_id"]) for r in rows if int(r["frame"]) == frame and int(r["track_id"]) >= 0}


def box_lineset(b, colour, o3d):
    C = vis.box_corners_lv(b)
    s0, s1 = heading_arrow(C)
    pts = np.vstack([C, s0[None], s1[None]])
    lines = list(EDGES) + [(8, 9)]
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(np.array(lines)))
    ls.colors = o3d.utility.Vector3dVector([colour] * len(EDGES) + [[1.0, 0.9, 0.0]])
    return ls


def main(argv=None):
    import open3d as o3d
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--scan", type=int, default=None, help="start in the single-scan view at this scan id of --frame")
    p.add_argument("--screenshot", action="store_true", help="save one screenshot of --frame without a window")
    p.add_argument("--clean", default="before", choices=("before", "after", "diff"), help="start in this residual-removal mode (key G)")
    p.add_argument("--no-predicted", action="store_true", help="removal modes without the predicted-position clusters (as clean_basic/)")
    p.add_argument("--track", action="store_true", help="3D tracking results (<out>/track/): person points coloured by 3D id (key C cycles "
                   "time / 2D id / 3D id), track id, speed and segment printed, dim box at the predicted position of tracks without a detection")
    a = p.parse_args(argv)
    a.dataset = a.name
    full = load_config()
    P = result_paths(full, a.name, load_datasets())
    cfg = full["detection"]
    out = P.detection
    run = Run(P.data, read_clock(P.clock))
    pose = PoseInterp(run, os.path.join(P.odometry, "odometry.csv"))
    store = ScanStore(run, pose, load_masks(P.masks), cfg)
    geom = CamGeom(P.data)                                 # camera FOV of the window points for the predicted-cluster removal
    persons = {}
    for r in csv.DictReader(open(os.path.join(out, "persons.csv"))):
        persons.setdefault(int(r["frame"]), []).append(r)
    frows = {int(r["frame"]): r for r in csv.DictReader(open(os.path.join(out, "frames.csv"))) if r["status"] == "ok"}
    frames = sorted(frows)
    cands = {}
    for r in csv.DictReader(open(os.path.join(out, "scan_candidates.csv"))):
        cands.setdefault(int(r["frame"]), []).append(r)
    state = dict(i=frames.index(a.frame) if a.frame in frames else 0, colour="time", bg=True, floor=False, others=True, burst=False,
                 scan=None, clean="before")                   # scan = index into frame_scans(...) of the current frame, None = accumulated view
    CLEAN_MODES = ("before", "after", "diff")
    gap = cfg["accumulate"]["burst_gap_s"]
    state["clean"] = a.clean
    if a.scan is not None:
        ids = [e["scan_id"] for e in frame_scans(cands.get(frames[state["i"]], []), frames[state["i"]])]
        state["scan"] = ids.index(a.scan) if a.scan in ids else (None if not ids else 0)
    os.makedirs(os.path.join(out, "screens"), exist_ok=True)
    trk_rows, trk_frames, trk_obs = [], {}, {}
    if a.track:
        tdir = P.tracking
        trk_rows = list(csv.DictReader(open(os.path.join(tdir, "rows.csv"))))
        for r in csv.DictReader(open(os.path.join(tdir, "track_frames.csv"))):
            trk_frames.setdefault(int(r["frame"]), []).append(r)
        for r in csv.DictReader(open(os.path.join(tdir, "observations.csv"))):
            trk_obs.setdefault(int(r["track_id"]), []).append(r)
        state["colour"] = "track"
    COLOURS = ("time", "id", "track") if a.track else ("time", "id")

    def frame_geoms(fr):
        fx = frows[fr]
        t_c = int(fx["t_img_us"])
        R_lv = np.array([[float(fx[f"R{i}{j}"]) for j in range(3)] for i in range(3)])
        plane = dict(R_level=R_lv, ground_z=float(fx["ground_z"]))
        geoms = []
        scans = frame_scans(cands.get(fr, []), fr)
        cur = scans[state["scan"]] if state["scan"] is not None and scans else None
        if state["scan"] is not None and not scans:
            state["scan"] = None
        plys = {r["key"]: read_ply_fields(os.path.join(out, "clouds", f"{fr:04d}_{r['key']}.ply")) for r in persons.get(fr, [])}
        res = None
        if state["bg"]:
            a0 = int(fx["anchor_scan"])
            ids = [i for i in range(a0 - 20, a0 + 21) if 0 <= i < run.n_scans()]
            if cur is not None:
                ids = [cur["scan_id"]]                            # single-scan view: the background is that one scan only
            R, pp = pose.at(t_c)
            X = np.concatenate([store.get(i)["xyz_odom"] for i in ids])
            sid = np.concatenate([np.full(len(store.get(i)["xyz_odom"]), i) for i in ids])
            trel = np.concatenate([(store.get(i)["t_point_us"] - t_c) / 1e6 for i in ids])
            lv = to_level((X - pp) @ R, plane)
            keep = np.ones(len(lv), bool) if state["floor"] else lv[:, 2] > 0.1
            if state["clean"] != "before":                       # removal modes: the residual points leave the background
                plist = [(r["key"], int(r["track_id"]), plys[r["key"]]) for r in persons.get(fr, [])]
                boxes = [dict(cx=float(r["cx"]), cy=float(r["cy"]), cz=float(r["cz"]), l=float(r["l"]), w=float(r["w"]), h=float(r["h"]),
                              yaw=math.radians(float(r["yaw_deg"]))) for r in persons.get(fr, [])]
                res = frame_ghosts(plist, lv, sid, gap, boxes=boxes)
                if not a.no_predicted:
                    fov = np.concatenate([scan_fov_mask(store.get(i), pose, geom, cfg) for i in ids])
                    extend_frame(res, plist, lv, sid, trel, ext_params(cfg), allowed=fov)
                keep &= ~res["removed"]
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lv[keep]))
            pc.paint_uniform_color([0.35, 0.35, 0.35])
            geoms.append(pc)
            if state["clean"] == "diff" and len(res["fields"]["x"]):
                f = res["fields"]
                pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.stack([f["x"], f["y"], f["z"]], 1).astype(float)))
                colr = np.where((f["reason"] == TRIMMED)[:, None], [[1.0, 0.6, 0.15]], [[0.9, 0.2, 0.2]])
                colr[f["reason"] == PREDICTED] = [0.9, 0.25, 0.9]
                pc.colors = o3d.utility.Vector3dVector(colr)
                geoms.append(pc)
        print(f"frame {fr:04d}  t_img {t_c}  persons {len(persons.get(fr, []))}  selected scans {len(scans)}  removal mode {state['clean']}"
              + (f": removed {int(res['removed'].sum())} pts, unmatched {sum(s['n_unmatched'] for s in res['stats'])}" if res is not None else ""))
        if cur is not None:
            who = ", ".join(f"{k}: {n} pts{' (main)' if u else ''}" for k, (n, u) in sorted(cur["persons"].items()))
            print(f"  single scan {cur['scan_id']} ({state['scan'] + 1}/{len(scans)}), dt_cam {cur['dt_cam_s']:+.2f} s: {who}")
        tid_of = track_ids_for_frame(trk_rows, fr) if a.track else {}
        for pi, r in enumerate(persons.get(fr, [])):
            d = plys[r["key"]]
            if res is not None:                                       # removal modes: only the kept PLY points (main or inside a box) are drawn
                d = {k: np.asarray(v)[res["keep_ply"][pi]] for k, v in d.items()}
            col = np.array(vis.key_colour(r["key"]))[::-1] / 255.0
            if state["colour"] == "track":
                col = np.array(PAL_TRACK[tid_of[r["key"]] % 1024])[::-1] / 255.0 if r["key"] in tid_of else np.array([0.6, 0.6, 0.6])
            if cur is not None:
                main, other = split_scan(d, cur["scan_id"])      # single-scan view: that scan in full colour, the other scans dimmed
            else:
                main, other = split_main(d)
            for part, dim in ((main, 1.0), (other, 0.45)):       # other part drawn dimmer, hidden with O
                if len(part["x"]) == 0 or (dim < 1.0 and not state["others"]):
                    continue
                pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.stack([part["x"], part["y"], part["z"]], 1).astype(float)))
                if state["colour"] == "time":
                    pc.colors = o3d.utility.Vector3dVector(vis.time_colour(part["time_rel_s"])[:, ::-1] / 255.0 * dim)
                else:
                    pc.paint_uniform_color(col * dim)
                geoms.append(pc)
            b = dict(cx=float(r["cx"]), cy=float(r["cy"]), cz=float(r["cz"]), l=float(r["l"]), w=float(r["w"]), h=float(r["h"]), yaw=math.radians(float(r["yaw_deg"])))
            bb = dict(cx=float(r["b_cx"]), cy=float(r["b_cy"]), cz=float(r["b_cz"]), l=float(r["b_l"]), w=float(r["b_w"]), h=float(r["b_h"]), yaw=b["yaw"])
            geoms.append(box_lineset(b, col, o3d))
            if state["burst"] and np.isfinite(bb["cx"]):
                geoms.append(box_lineset(bb, col * 0.6, o3d))
            fl = ",".join(k for k in ("lower_only", "upper_only", "single_scan", "fragmented", "smeared", "trimmed", "compensated") if r[k] == "1")
            print(f"  {r['key']}: {r['n_pts']} pts ({r['n_pts_all']} all), {r['n_scans']} scans, span {float(r['span_s']):.2f} s, "
                  f"box {float(r['l']):.2f}x{float(r['w']):.2f}x{float(r['h']):.2f} m yaw {float(r['yaw_deg']):.0f} ({r['yaw_source']}), "
                  f"burst {float(r['b_l']):.2f}x{float(r['b_w']):.2f}x{float(r['b_h']):.2f}, valid {r['valid']} {fl}")
        if a.track:
            R, pp = pose.at(t_c)
            for tr_ in trk_frames.get(fr, []):
                tid = int(tr_["track_id"])
                sp = float(tr_["speed"]) if tr_["speed"] not in ("", "nan") else float("nan")
                ss = float(tr_["sigma_speed"]) if tr_["sigma_speed"] not in ("", "nan") else float("nan")
                print(f"  T{tid} {tr_['status']} [{tr_['segment']}] keys {tr_['keys'] or '-'} speed {sp:.2f} +- {ss:.2f} m/s, last pass {tr_['vis_last'] or '-'}")
                if tr_["has_det"] == "1" or max(float(tr_["sigma_x"]), float(tr_["sigma_y"])) > 1.0:
                    continue                                      # predicted box only where the position is known within 1 m (gate_sigma_max)
                prev = [o for o in trk_obs.get(tid, []) if int(o["t_us"]) <= t_c] or trk_obs.get(tid, [])[:1]
                if not prev:
                    continue
                o = prev[-1]
                c = to_level((np.array([[float(tr_["x"]), float(tr_["y"]), float(o["z_c"])]]) - pp) @ R, plane)[0]
                d_lv = plane["R_level"] @ (R.T @ np.array([math.cos(math.radians(float(o["yaw_deg"]))), math.sin(math.radians(float(o["yaw_deg"]))), 0.0]))
                h = float(o["h"]) if o["h"] not in ("", "nan") else 1.7
                bpred = dict(cx=c[0], cy=c[1], cz=h / 2, l=float(o["l"]), w=float(o["w"]), h=h, yaw=math.atan2(d_lv[1], d_lv[0]))
                geoms.append(box_lineset(bpred, np.array(PAL_TRACK[tid % 1024])[::-1] / 255.0 * 0.45, o3d))   # predicted position, no detection
        grid = o3d.geometry.LineSet()
        pts, lines = [], []
        for k in range(-10, 11, 2):
            pts += [[k, -10, 0], [k, 10, 0], [-10, k, 0], [10, k, 0]]
            lines += [(len(pts) - 4, len(pts) - 3), (len(pts) - 2, len(pts) - 1)]
        grid.points, grid.lines = o3d.utility.Vector3dVector(np.array(pts, float)), o3d.utility.Vector2iVector(np.array(lines))
        grid.paint_uniform_color([0.2, 0.25, 0.2])
        geoms.append(grid)
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
        return geoms

    def shot_name():
        name = f"{frames[state['i']]:04d}"
        if state["scan"] is not None:
            scans = frame_scans(cands.get(frames[state["i"]], []), frames[state["i"]])
            name += f"_s{scans[state['scan']]['scan_id']:04d}"
        return name + ("" if state["clean"] == "before" else f"_{state['clean']}" + ("_basic" if a.no_predicted else "")) + ".png"

    if a.screenshot:
        v = o3d.visualization.Visualizer()
        v.create_window(width=1400, height=1000)                   # a visible window: offscreen capture returns the desktop on this system
        for g in frame_geoms(frames[state["i"]]):
            v.add_geometry(g)
        v.get_render_option().background_color = np.array([0.06, 0.06, 0.06]); v.get_render_option().point_size = 2.0
        for _ in range(5):
            v.poll_events(); v.update_renderer()
        path = os.path.join(out, "screens", shot_name())
        v.capture_screen_image(path); v.destroy_window()
        print("saved", path)
        return
    v = o3d.visualization.VisualizerWithKeyCallback()
    v.create_window(f"selective accumulation: {a.dataset}", width=1400, height=1000)
    v.get_render_option().background_color = np.array([0.06, 0.06, 0.06]); v.get_render_option().point_size = 2.0

    def load(first=False):
        ctr = v.get_view_control()
        prm = ctr.convert_to_pinhole_camera_parameters() if not first else None
        v.clear_geometries()
        for g in frame_geoms(frames[state["i"]]):
            v.add_geometry(g, reset_bounding_box=first)
        if prm is not None:
            ctr.convert_from_pinhole_camera_parameters(prm)
        return False

    def step(k):
        state["i"] = int(np.clip(state["i"] + k, 0, len(frames) - 1)); state["scan"] = None if state["scan"] is None else 0; return load()

    def step_scan(k):
        n = len(frame_scans(cands.get(frames[state["i"]], []), frames[state["i"]]))
        if n == 0:
            print("  no selected scan in this frame"); return False
        state["scan"] = 0 if state["scan"] is None and k > 0 else n - 1 if state["scan"] is None else int(np.clip(state["scan"] + k, 0, n - 1))
        return load()

    def accumulated(_):
        state["scan"] = None; return load()

    def jump(_):
        try:
            fr = int(input("frame: "))
            state["i"] = frames.index(fr) if fr in frames else min(range(len(frames)), key=lambda i: abs(frames[i] - fr))
        except ValueError:
            pass
        return load()

    def toggle(name):
        def f(_):
            state[name] = (not state[name]) if name != "colour" else COLOURS[(COLOURS.index(state["colour"]) + 1) % len(COLOURS)]; return load()
        return f

    def shot(_):
        path = os.path.join(out, "screens", shot_name()); v.capture_screen_image(path); print("saved", path); return False

    def clean_mode(_):
        state["clean"] = CLEAN_MODES[(CLEAN_MODES.index(state["clean"]) + 1) % len(CLEAN_MODES)]; return load()

    for key, fn in ((ord("N"), lambda _: step(1)), (262, lambda _: step(1)), (ord("P"), lambda _: step(-1)), (263, lambda _: step(-1)), (ord("J"), jump),
                    (ord("C"), toggle("colour")), (ord("B"), toggle("bg")), (ord("O"), toggle("others")), (ord("T"), toggle("burst")),
                    (ord("F"), toggle("floor")), (ord("]"), lambda _: step_scan(1)), (ord("["), lambda _: step_scan(-1)), (ord("A"), accumulated),
                    (ord("G"), clean_mode), (ord("S"), shot), (ord("Q"), lambda _: sys.exit(0))):
        v.register_key_callback(key, fn)
    load(first=True)
    v.run()


if __name__ == "__main__":
    main()
