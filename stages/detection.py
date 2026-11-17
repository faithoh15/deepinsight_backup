"""Stage detection: v3 camera-guided selective LiDAR accumulation, 3D person boxes per camera frame -> 1_detection/
(3d_detection/scripts/run_detection.py, verbatim; main() became run(); the dataset table / legacy folder lookup was replaced by the
paths of this recording; settings = configs/pipeline.yaml detection (= 3d_detection/configs/common.yaml), workers / video = run.*).

Per camera frame c (clock-corrected time t_c): anchor scan = nearest scan centre, candidates = anchor-20..anchor+20 -> single-scan
preprocessing (masks, 10 m, per-point time, odometry frame) -> frame ground plane (RANSAC + gravity check) and ceiling -> per scan: 2D
boxes of every person interpolated to the scan time, projection with the pose at the scan time, inside-box points -> clustering ->
piece selection -> per person: consistency check, selective accumulation in the leveled device frame at t_c, final clustering, box_acc
(accumulated, motion-compensated box mode) and box_burst (nearest burst).
Outputs: frames.csv, scans.csv, scan_candidates.csv, persons.csv, ambiguous.csv, clouds/NNNN_<key>.ply, vis/, config_used.yaml, summary.json.
"""
from __future__ import annotations

import os as _os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from pipeline.common.angbox import TILE_W  # noqa: E402
from pipeline.common.cam_geom import CamGeom  # noqa: E402
from pipeline.common.io import write_csv, write_json, write_person_ply  # noqa: E402
from pipeline.common.pose import PoseInterp  # noqa: E402
from pipeline.common.recording import Run  # noqa: E402
from pipeline.detect import vis  # noqa: E402
from pipeline.detect.accumulate import accumulate_person, assert_unique_points, consistency_filter, resolve_bursts  # noqa: E402
from pipeline.detect.extract import extract_frame  # noqa: E402
from pipeline.detect.ground import fit_frame_plane, plane_to_odom, to_level  # noqa: E402
from pipeline.detect.project import PersonBoxes  # noqa: E402
from pipeline.detect.scans import ScanStore, anchor_scan, load_masks  # noqa: E402
from pipeline.prep.clock import read_clock  # noqa: E402
from stages._base import stage_main  # noqa: E402

G = {}

FRAME_FIELDS = ["frame", "cam0_stamp_us", "t_img_us", "status", "anchor_scan", "anchor_dt_ms", "scan_first", "scan_last", "n_cand_scans", "n_raw",
                "n_range", "n_fov", "n_after_ground", "n_after_ceiling", "plane_source", "ground_z", "plane_nz", "ceiling_h", "n_boxes2d",
                "n_persons", "n_valid", "n_ambiguous_rows", "seconds"] + [f"R{i}{j}" for i in range(3) for j in range(3)]
SCAN_FIELDS = ["scan_id", "t_scan_us", "n_raw", "n_near", "n_self", "n_oper", "n_far", "n_range"]
CAND_FIELDS = ["frame", "key", "track_id", "scan_id", "t_scan_us", "dt_cam_s", "box_mode", "box_dist_m", "n_fov", "n_in_box", "n_clusters", "n_sel",
               "status", "score", "score2", "range_m", "footprint", "zmin_h", "zmax_h", "cx_odom", "cy_odom", "cz_odom", "consistency_res", "used", "resolved", "alt"]
PERSON_FIELDS = ["frame", "t_img_us", "key", "track_id", "source", "cam", "x1", "y1", "x2", "y2", "conf", "dist2d_m", "n_pts", "n_pts_all", "n_scans",
                 "n_bursts", "scan_first", "scan_last", "t_first_us", "t_last_us", "span_s", "n_components",
                 "cx", "cy", "cz", "l", "w", "h", "yaw_deg", "cx_odom", "cy_odom", "cz_odom", "yaw_odom_deg",
                 "b_cx", "b_cy", "b_cz", "b_l", "b_w", "b_h", "b_cx_odom", "b_cy_odom", "b_cz_odom", "b_n_scans",
                 "all_l", "all_w", "all_h", "n_scans_all", "n_bursts_all", "b_dt_s",
                 "yaw_source", "yaw_reliable", "zmin_h", "zmax_h", "lower_only", "upper_only", "single_scan", "fragmented", "smeared", "trimmed", "compensated", "box_dt_s", "v_mps", "n_scans_untrimmed", "valid",
                 "n_cand_scans", "n_selected", "n_ambiguous", "n_taken", "n_none", "n_no_box", "n_rejected"]


def _init(cfg):
    d = cfg["dataset"]                     # paths of this recording (final layout), set by run()
    run = Run(d["data"], d["image_offset_us"])
    pose = PoseInterp(run, d["odometry"])
    geom = CamGeom(d["data"])
    masks = load_masks(d["masks"]) if (cfg["scan_prep"]["self_mask"] or cfg["scan_prep"]["operator_mask"]) else None
    G.update(cfg=cfg, run=run, pose=pose, geom=geom, store=ScanStore(run, pose, masks, cfg),
             boxes=PersonBoxes(d["sidebyside"], d["detections"], d["tracks"], pose, geom, cfg))


def _raw_lv(scan_ids, t_c, plane):
    """All raw points of the candidate scans (before any mask) in the leveled device frame at t_c, for the prep figure."""
    run, pose = G["run"], G["pose"]
    R, p = pose.at(t_c)
    out = []
    for i in scan_ids:
        s = run.load_scan(i)
        m = s["range"] > 0.05
        tp = s["t_point_us"][m]
        x = np.einsum("nij,nj->ni", pose.gyro.at(tp), s["xyz"][m]) + pose.odo.at(tp)
        out.append(((x - p) @ R)[::2])
    return to_level(np.concatenate(out), plane)


def process_frame(fr, t_c, stamp, out, vis_on):
    cfg, run, pose, geom, store, pb = G["cfg"], G["run"], G["pose"], G["geom"], G["store"], G["boxes"]
    t0 = time.time()
    row = dict(frame=fr, cam0_stamp_us=stamp, t_img_us=t_c, status="ok", n_persons=0, n_valid=0, n_ambiguous_rows=0)
    boxes2d = pb.frame_boxes(fr)
    row["n_boxes2d"] = len(boxes2d)
    a = anchor_scan(run, t_c, cfg["candidates"]["anchor_max_dt_ms"])
    if a is None:
        row.update(status="lidar_gap", seconds=time.time() - t0)
        return row, [], [], [], []
    row.update(anchor_scan=a, anchor_dt_ms=(int(run.lidar_ts[a]) + 25_000 - t_c) / 1e3)
    c = cfg["candidates"]
    scan_ids = [i for i in range(a - c["scans_before"], a + c["scans_after"] + 1) if 0 <= i < run.n_scans()]
    row.update(scan_first=scan_ids[0], scan_last=scan_ids[-1], n_cand_scans=len(scan_ids))
    scans = [store.get(i) for i in scan_ids]
    scan_rows = [{k: s[k] for k in SCAN_FIELDS} for s in scans]
    row["n_raw"] = sum(s["n_raw"] for s in scans)
    row["n_range"] = sum(s["n_range"] for s in scans)
    if not boxes2d:
        row.update(status="no_2d_box", seconds=time.time() - t0)
        return row, scan_rows, [], [], []
    # frame ground plane from all candidate scans in the device frame at t_c
    R_c, p_c = pose.at(t_c)
    all_odom = np.concatenate([s["xyz_odom"] for s in scans])
    all_dev = (all_odom - p_c) @ R_c
    k0, k1 = np.searchsorted(run.imu_ts, [int(run.lidar_ts[scan_ids[0]]), int(run.lidar_ts[scan_ids[-1]]) + 50_000])
    acc_up = run.acc[max(k0, 0):max(k1, k0 + 1)].mean(0)
    plane = fit_frame_plane(all_dev, acc_up, cfg)
    plane_odom = plane_to_odom(plane, R_c, p_c)
    plane_odom["ceiling_h"] = plane["ceiling_h"]
    row.update(plane_source=plane["source"], ground_z=plane["ground_z"], plane_nz=float(plane["n"][2]), ceiling_h=plane["ceiling_h"],
               **{f"R{i}{j}": float(plane["R_level"][i, j]) for i in range(3) for j in range(3)})
    keys = [k for k, _ in boxes2d]
    cands, masks = extract_frame(scans, keys, pb.boxes_at, plane_odom, geom, pose, cfg, t_c)
    resolve_bursts(cands, cfg)
    n_fov = sum(int(m["fov"].sum()) for m in masks)
    n_ground = sum(int((m["fov"] & (m["h"] > cfg["ground"]["ground_h"])).sum()) for m in masks)
    n_ceil = sum(int((m["fov"] & m["keep"]).sum()) for m in masks)
    row.update(n_fov=n_fov, n_after_ground=n_ground, n_after_ceiling=n_ceil)
    # per person: consistency + accumulation
    persons, cand_rows, person_rows = [], [], []
    for k in keys:
        cl = consistency_filter(cands[k], cfg)
        q = accumulate_person(cl, store, pose, plane, t_c, cfg)
        used = set(q["scan_ids"]) if q else set()
        for r in cl:
            ab = r.get("abox")
            cand_rows.append(dict(frame=fr, key=k, track_id=pb.key_track(k), scan_id=r["scan_id"], t_scan_us=r["t_scan_us"], dt_cam_s=r["dt_cam_s"],
                                  box_mode=ab.interp_mode if ab else "", box_dist_m=ab.dist_m if ab else np.nan, n_fov=r["n_fov"], n_in_box=r["n_in_box"],
                                  n_clusters=r["n_clusters"], n_sel=r["n_sel"] or r.get("n_sel_rejected", 0), status=r["status"], score=r["score"], score2=r["score2"],
                                  range_m=r["range_m"], footprint=r["footprint"], zmin_h=r["zmin_h"], zmax_h=r["zmax_h"],
                                  cx_odom=r["cxyz_odom"][0] if r["cxyz_odom"] is not None else np.nan, cy_odom=r["cxyz_odom"][1] if r["cxyz_odom"] is not None else np.nan,
                                  cz_odom=r["cxyz_odom"][2] if r["cxyz_odom"] is not None else np.nan, consistency_res=r.get("consistency_res", np.nan),
                                  used=int(r["status"] == "selected" and r["scan_id"] in used), resolved=r.get("resolved", ""),
                                  alt=";".join(f"{x['n']}pts@{x['range_m']:.1f}m/fp{x['footprint']:.2f}/top{x['zmax_h']:.2f}/s{x['score']:.2f}/{x['status']}" for x in r["alt"])))
        st = [r["status"] for r in cl]
        counts = dict(n_cand_scans=len(cl), n_selected=st.count("selected"), n_ambiguous=st.count("ambiguous") + st.count("ambiguous_multi"),
                      n_taken=st.count("taken_by_other"), n_none=st.count("none"), n_no_box=st.count("no_box"),
                      n_rejected=st.count("rejected_consistency") + st.count("rejected_range"))
        if q is None:
            continue
        q["frame"] = fr
        persons.append(q)
        b, bb = q["box_acc"], q["box_burst"]
        r2 = dict(boxes2d)[k]
        person_rows.append(dict(frame=fr, t_img_us=t_c, key=k, track_id=pb.key_track(k), source="track" if k.startswith("t") else "det",
                                cam=0 if r2["x1"] < TILE_W else 1, x1=r2["x1"], y1=r2["y1"], x2=r2["x2"], y2=r2["y2"], conf=r2["conf"], dist2d_m=r2["dist_m"],
                                n_pts=q["n_pts"], n_pts_all=q["n_pts_all"], n_scans=q["n_scans"], n_bursts=q["n_bursts"], scan_first=q["scan_ids"][0],
                                scan_last=q["scan_ids"][-1], t_first_us=q["t_first_us"], t_last_us=q["t_last_us"], span_s=q["span_s"], n_components=q["n_components"],
                                cx=b["cx"], cy=b["cy"], cz=b["cz"], l=b["l"], w=b["w"], h=b["h"], yaw_deg=math.degrees(b["yaw"]), cx_odom=b["cx_o"], cy_odom=b["cy_o"],
                                cz_odom=b["cz_o"], yaw_odom_deg=math.degrees(b["yaw_o"]) if np.isfinite(b["yaw_o"]) else np.nan,
                                b_cx=bb["cx"], b_cy=bb["cy"], b_cz=bb["cz"], b_l=bb["l"], b_w=bb["w"], b_h=bb["h"], b_cx_odom=bb["cx_o"], b_cy_odom=bb["cy_o"],
                                b_cz_odom=bb["cz_o"], b_n_scans=len(q["nearest_burst"]), all_l=q["box_all"]["l"], all_w=q["box_all"]["w"], all_h=q["box_all"]["h"],
                                n_scans_all=q["n_scans_all"], n_bursts_all=q["n_bursts_all"], b_dt_s=q["b_dt_s"], yaw_source=q["yaw_source"], yaw_reliable=q["yaw_reliable"],
                                zmin_h=q["zmin_h"], zmax_h=q["zmax_h"], **{f: int(v) for f, v in q["flags"].items()}, v_mps=q["v_mps"], n_scans_untrimmed=q["n_scans_untrimmed"], box_dt_s=q["box_dt_s"],
                                valid=int(q["valid"]), **counts))
    assert_unique_points(persons)
    row.update(n_persons=len(persons), n_valid=sum(q["valid"] for q in persons),
               n_ambiguous_rows=sum(1 for r in cand_rows if r["status"] in ("ambiguous", "ambiguous_multi")))
    for q in persons:
        write_person_ply(os.path.join(out, "clouds", f"{fr:04d}_{q['key']}.ply"), q)
    if vis_on:
        imgp = [f for f in os.listdir(cfg["dataset"]["sidebyside"]) if f.startswith(f"{fr:04d}_") and f.endswith(".jpg")]
        img = cv2.imread(os.path.join(cfg["dataset"]["sidebyside"], imgp[0]), cv2.IMREAD_REDUCED_COLOR_4) if imgp else np.full((750, 2000, 3), 20, np.uint8)
        all_lv = to_level(all_dev, plane)
        off = np.cumsum([0] + [len(s["point_id"]) for s in scans])
        fov = np.concatenate([m["fov"] for m in masks]); keep = np.concatenate([m["keep"] for m in masks])
        inbox, sel = {}, {}
        for k in keys:
            mi = np.zeros(len(all_lv), bool); ms = np.zeros(len(all_lv), bool)
            for j, m in enumerate(masks):
                if k in m["inbox"]:
                    mi[off[j] + m["inbox"][k]] = True
            for r in cands[k]:
                if r["status"] == "selected" and len(r["idx"]):
                    j = scan_ids.index(r["scan_id"]); ms[off[j] + r["idx"]] = True
            inbox[k], sel[k] = mi, ms
        F = dict(frame=fr, raw_lv=_raw_lv(scan_ids, t_c, plane), n_raw=row["n_raw"], all_lv=all_lv, fov=fov, keep=keep, inbox=inbox, sel=sel, image=img, boxes2d=boxes2d,
                 geom=geom, plane=plane, keys_col={k: vis.key_colour(k) for k in keys})
        vis.draw_prep(os.path.join(out, "vis", "prep", f"{fr:04d}.png"), F, persons)
        for k in keys:
            q = next((q for q in persons if q["key"] == k), None)
            vis.draw_scans(os.path.join(out, "vis", "scans", f"{fr:04d}_{k}.png"), F, k, cands[k], q)
        vis.draw_accum(os.path.join(out, "vis", "accum", f"{fr:04d}.png"), F, persons)
        vis.draw_boxes(os.path.join(out, "vis", "boxes_img", f"{fr:04d}.jpg"), os.path.join(out, "vis", "boxes_bev", f"{fr:04d}.png"), F, persons)
    row["seconds"] = time.time() - t0
    return row, scan_rows, cand_rows, person_rows, [r for r in cand_rows if r["status"] in ("ambiguous", "ambiguous_multi", "taken_by_other")]


def _chunk(task):
    frames, out, vis_every = task
    res = []
    for fr, t_c, stamp in frames:
        try:
            res.append(process_frame(fr, t_c, stamp, out, vis_every > 0 and fr % vis_every == 0))
        except Exception as e:                     # record, do not stop the run
            import traceback
            res.append((dict(frame=fr, cam0_stamp_us=stamp, t_img_us=t_c, status=f"error: {type(e).__name__}: {e}"), [], [], [], []))
            traceback.print_exc()
    return res




def run(P, cfg, frames=None, vis_every=1, **kw):
    """frames = "START:STOP" camera frames (default all); vis_every 0 = no figures."""
    a = type("A", (), dict(frames=frames, workers=cfg["run"]["detection_workers"], vis_every=vis_every, video=cfg["run"]["detection_video"]))
    cfg = dict(cfg["detection"])
    cfg["dataset"] = dict(name=P.name, data=P.data, image_offset_us=read_clock(P.clock), odometry=os.path.join(P.odometry, "odometry.csv"),
                          masks=P.masks, sidebyside=P.sidebyside, detections=os.path.join(P.yolo, "detections.csv"),
                          tracks=os.path.join(P.track2d, "tracks.csv"))
    out = P.detection
    for sub in ("clouds", "vis/prep", "vis/scans", "vis/accum", "vis/boxes_bev", "vis/boxes_img"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
        if not a.frames:                                   # full run: remove stale per-frame files of a previous run of this folder
            for f in os.listdir(os.path.join(out, sub)):
                if f.endswith((".ply", ".png", ".jpg")):
                    os.remove(os.path.join(out, sub, f))
    yaml.safe_dump({k: v for k, v in cfg.items() if k != "dataset"}, open(os.path.join(out, "config_used.yaml"), "w"), sort_keys=False, allow_unicode=True)
    frames = []
    for r in csv.DictReader(open(os.path.join(cfg["dataset"]["sidebyside"], "frames.csv"))):
        frames.append((int(r["frame"]), int(r["t_img_us"]), int(r["cam0_stamp_us"])))
    if a.frames:
        lo, hi = map(int, a.frames.split(":"))
        frames = [f for f in frames if lo <= f[0] < hi]
    t0 = time.time()
    n = max(1, min(a.workers, len(frames)))
    chunks = [(frames[i::n], out, a.vis_every) for i in range(n)]          # interleaved: balanced load, neighbours still share scans
    _init(cfg)       # fail fast in the main process: a Pool initializer that raises is restarted forever and the stage would hang
    if n == 1:
        results = [_chunk(chunks[0])]
    else:
        with Pool(n, initializer=_init, initargs=(cfg,)) as pool:
            results = pool.map(_chunk, chunks)
    rows = sorted((r for res in results for r in res), key=lambda r: r[0]["frame"])
    frame_rows = [r[0] for r in rows]
    scan_rows = {s["scan_id"]: s for r in rows for s in r[1]}
    cand_rows = [c for r in rows for c in r[2]]
    person_rows = [q for r in rows for q in r[3]]
    amb_rows = [q for r in rows for q in r[4]]
    write_csv(os.path.join(out, "frames.csv"), frame_rows, FRAME_FIELDS)
    write_csv(os.path.join(out, "scans.csv"), [scan_rows[k] for k in sorted(scan_rows)], SCAN_FIELDS)
    write_csv(os.path.join(out, "scan_candidates.csv"), cand_rows, CAND_FIELDS)
    write_csv(os.path.join(out, "persons.csv"), person_rows, PERSON_FIELDS)
    write_csv(os.path.join(out, "ambiguous.csv"), amb_rows, CAND_FIELDS)
    st = [c["status"] for c in cand_rows]
    summary = dict(dataset=cfg["dataset"]["name"], frames=len(frame_rows), frames_ok=sum(r["status"] == "ok" for r in frame_rows),
                   frame_status={s: sum(r["status"] == s for r in frame_rows) for s in sorted(set(r["status"] for r in frame_rows))},
                   persons=len(person_rows), persons_valid=sum(q["valid"] for q in person_rows),
                   keys=len(set(q["key"] for q in person_rows)),
                   candidate_status={s: st.count(s) for s in sorted(set(st))},
                   n_scans_used_median=float(np.median([q["n_scans"] for q in person_rows])) if person_rows else None,
                   n_pts_median=float(np.median([q["n_pts"] for q in person_rows])) if person_rows else None,
                   span_s_median=float(np.median([q["span_s"] for q in person_rows])) if person_rows else None,
                   flags={f: int(sum(q[f] for q in person_rows)) for f in ("lower_only", "upper_only", "single_scan", "fragmented", "smeared")},
                   yaw_source={s: sum(q["yaw_source"] == s for q in person_rows) for s in ("motion", "pca")},
                   seconds=round(time.time() - t0, 1))
    write_json(os.path.join(out, "summary.json"), summary)
    if a.video:
        for sub in ("prep", "accum", "boxes_bev", "boxes_img"):
            vis.make_video(os.path.join(out, "vis", sub), os.path.join(out, "vis", f"{sub}.mp4"))
    print(json.dumps(summary, ensure_ascii=False))
    return summary



if __name__ == "__main__":
    stage_main(run)
