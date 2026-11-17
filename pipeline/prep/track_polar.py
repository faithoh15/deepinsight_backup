"""Polar 2D tracker (proposed_pipeline/track_polar.py, verbatim; run() takes the frame times as an argument instead of reading the
LiDAR windows.csv). Each YOLO box -> bearing (camera ray rotated by the device pose at the image time) and a range from the box height;
a constant-velocity Kalman filter per person in the horizontal odometry plane, gated association on the innovation with a bearing /
range cost, tentative / confirmed / lost life cycle, back-filled first rows; returns the track rows for tracks.csv.
"""
from __future__ import annotations

import math
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

from pipeline.prep.kalman_cv import KalmanCV


BIG = 1e6


class Track:
    __slots__ = ("id", "kf", "t", "t_last", "status", "hits", "rows", "last_z", "conf")

    def __init__(self, tid, z, R, t, v0_sigma):
        self.id = tid
        self.kf = KalmanCV(z, 1.0, v0_sigma)
        self.kf.P[:2, :2] = R
        self.t = t                   # time the state refers to (us)
        self.t_last = t              # last match (us)
        self.status = "tentative"    # tentative | confirmed | lost
        self.hits = 1
        self.rows = []               # pending rows of a tentative track (back-filled at confirmation)
        self.last_z = np.asarray(z, float)
        self.conf = 0.0

    @property
    def sigma_pos(self):
        return math.sqrt(max(float(np.linalg.eigvalsh(self.kf.P[:2, :2]).max()), 0.0))


def meas_cov(az, r, frac, sigma_az, floor_r=0.3, floor_t=0.25):
    """radial sigma = max(frac * r, floor_r); tangential sigma^2 = (r * sigma_az)^2 + floor_t^2 (the box centre of a near
    person wanders over the body width)."""
    c, s = math.cos(az), math.sin(az)
    Rot = np.array([[c, -s], [s, c]])
    return Rot @ np.diag([max(frac * r, floor_r) ** 2, (r * sigma_az) ** 2 + floor_t ** 2]) @ Rot.T


def measurements(rs, T, geom, a):
    """Detection rows of one frame -> list of dict(idx, az, r, trunc, conf, z, R) in the odometry plane."""
    R3, pc = T[:3, :3], T[:3, 3]
    out = []
    sig = math.radians(a.pl_sigma_az_deg)
    for i, r in enumerate(rs):
        d = R3 @ geom.ray((r["x1"] + r["x2"]) / 2, (r["y1"] + r["y2"]) / 2)[0]
        az = math.atan2(d[1], d[0])
        rng = float(np.clip(r["dist_m"], a.pl_dist_range[0], a.pl_dist_range[1]))
        trunc = bool(int(r.get("truncated_top", 0) or 0) or int(r.get("touches_bottom", 0) or 0))
        z = np.array([pc[0] + rng * math.cos(az), pc[1] + rng * math.sin(az)])
        out.append(dict(idx=i, az=az, r=rng, trunc=trunc, conf=float(r["conf"]), z=z, R=meas_cov(az, rng, a.pl_range_frac_trunc if trunc else a.pl_range_frac, sig)))
    return out


class PolarTracker:
    def __init__(self, a):
        self.a = a
        self.tracks: list[Track] = []
        self.next_id = 1
        self.backfilled = 0
        self.stats = dict(matched_high=0, matched_low=0, new=0, dup_skipped=0, merged_lost=0, lost=0, deleted_lost=0, deleted_tent=0, reacquired=0)

    # -- association ------------------------------------------------------------------------------------------------
    def _cost(self, tracks, meas, pc, t):
        a = self.a
        sig_az = math.radians(a.pl_sigma_az_deg)
        C = np.full((len(tracks), len(meas)), BIG)
        for i, trk in enumerate(tracks):
            dt = max((t - trk.t_last) / 1e6, 1e-3)
            px, py = trk.kf.x[0] - pc[0], trk.kf.x[1] - pc[1]
            az_pred, r_pred = math.atan2(py, px), math.hypot(px, py)
            for j, m in enumerate(meas):
                daz = (m["az"] - az_pred + math.pi) % (2 * math.pi) - math.pi
                # tangential speed gate, noise-aware: displacement beyond 3 sigma of the bearing noise, per second
                rr = max(r_pred, m["r"])
                sig_t = math.sqrt((rr * sig_az) ** 2 + 0.25 ** 2)
                if max(abs(daz) * rr - 3.0 * sig_t, 0.0) / dt > a.pl_v_tan_max:
                    continue
                if m["trunc"] and r_pred > 1.6 * m["r"]:            # a truncated box only says "not farther than ~r"
                    continue
                _, _, d2 = trk.kf.innovation_xy(m["z"], m["R"])
                if d2 <= a.pl_gate_chi2:
                    C[i, j] = d2
        return C

    def _dup(self, m, q):
        """same bearing (<= --pl-dup-az-deg) and range ratio <= 2: the two boxes are the same person"""
        return abs((m["az"] - q["az"] + math.pi) % (2 * math.pi) - math.pi) <= math.radians(self.a.pl_dup_az_deg) and max(m["r"], q["r"]) <= 2.0 * min(m["r"], q["r"])

    @staticmethod
    def _assign(C):
        if C.size == 0:
            return []
        ri, ci = linear_sum_assignment(C)
        return [(i, j) for i, j in zip(ri, ci) if C[i, j] < BIG]

    # -- one camera frame -------------------------------------------------------------------------------------------
    def step(self, fr, t, pc, meas, make_row):
        """pc = device xy at the image time; meas from measurements(); make_row(track, m) -> output dict.
        Returns the rows of this frame (plus back-filled first rows of newly confirmed tracks)."""
        a = self.a
        for trk in self.tracks:
            dt = (t - trk.t) / 1e6
            if dt > 0:
                if trk.status == "lost":
                    trk.kf.predict(dt, a.pl_sigma_a, model="damped", tau=a.pl_lost_tau)
                else:
                    trk.kf.predict(dt, a.pl_sigma_a)
                trk.t = t
        high = [m for m in meas if m["conf"] >= a.high]
        low = [m for m in meas if a.low <= m["conf"] < a.high]
        matched, used = {}, set()
        pairs = self._assign(self._cost(self.tracks, high, pc, t))
        for i, j in pairs:
            matched[self.tracks[i].id] = high[j]
            used.add(high[j]["idx"])
        self.stats["matched_high"] += len(pairs)
        rest = [trk for trk in self.tracks if trk.id not in matched and trk.status != "tentative"]
        pairs = self._assign(self._cost(rest, low, pc, t))
        for i, j in pairs:
            matched[rest[i].id] = low[j]
            used.add(low[j]["idx"])
        self.stats["matched_low"] += len(pairs)
        out, keep = [], []
        for trk in self.tracks:
            m = matched.get(trk.id)
            if m is not None:
                trk.kf.update_xy(m["z"], m["R"])
                trk.t_last, trk.last_z, trk.conf = t, m["z"], m["conf"]
                trk.hits += 1
                row = make_row(trk, m)
                if trk.status == "tentative":
                    trk.rows.append(row)
                    if trk.hits >= 2:
                        # a tentative track whose box duplicates the box of an older confirmed track (same bearing, the
                        # second tile box of a person on the seam) is dropped instead of confirmed
                        if any(self._dup(m, matched[o.id]) for o in self.tracks if o is not trk and o.status != "tentative" and o.id in matched):
                            self.stats["dup_skipped"] += 1
                            continue
                        trk.status = "confirmed"
                        out += trk.rows
                        self.backfilled += len(trk.rows) - 1
                        trk.rows = []
                else:
                    if trk.status == "lost":
                        self.stats["reacquired"] += 1
                    trk.status = "confirmed"
                    out.append(row)
                keep.append(trk)
                continue
            age = (t - trk.t_last) / 1e6
            if trk.status == "tentative":
                if age <= a.pl_tent_max_s:
                    keep.append(trk)
                else:
                    self.stats["deleted_tent"] += 1
            else:
                if trk.status == "confirmed":
                    trk.status = "lost"
                    self.stats["lost"] += 1
                if age <= a.pl_lost_max_s and trk.sigma_pos <= a.pl_sigma_max:
                    keep.append(trk)
                else:
                    self.stats["deleted_lost"] += 1
        # a lost track whose prediction lies on the bearing of a box matched by another track is the same person
        # (the track was split earlier): drop it instead of letting the two compete for the boxes
        matched_meas = list(matched.values())
        keep2 = []
        for trk in keep:
            if trk.status == "lost":
                px, py = trk.kf.x[0] - pc[0], trk.kf.x[1] - pc[1]
                pred = dict(az=math.atan2(py, px), r=math.hypot(px, py))
                if any(self._dup(pred, q) for q in matched_meas):
                    self.stats["merged_lost"] += 1
                    continue
            keep2.append(trk)
        self.tracks = keep2
        # new tracks from unmatched boxes >= new, unless a duplicate of a matched box of this frame
        for m in meas:
            if m["idx"] in used or m["conf"] < a.new:
                continue
            dup = any(self._dup(m, q) for q in matched_meas)
            if dup:
                self.stats["dup_skipped"] += 1
                continue
            trk = Track(self.next_id, m["z"], m["R"], t, a.pl_v0_sigma)
            trk.conf = m["conf"]
            self.next_id += 1
            trk.rows.append(make_row(trk, m))
            self.tracks.append(trk)
            matched_meas.append(m)
            self.stats["new"] += 1
        return out


# ======================================================================================================================
def run(a, out, by, frames, fpath, poses, times, geom):
    """track_2d.py --mode polar: returns (trows, backfilled, t0, stats). Visualization like the other modes (a.vis_every 0 = none).
    times = {frame: t_img_us} of the frames that have a pose (frame_poses.csv; was windows.csv)."""
    import csv
    import time
    import cv2
    t0 = time.time()
    tracker = PolarTracker(a)
    PAL = np.random.default_rng(3).integers(60, 255, (1024, 3))
    trows, no_pose = [], 0
    for fr_i, fr in enumerate(frames):
        rs = by.get(fr, [])
        img = cv2.imread(fpath[fr], cv2.IMREAD_REDUCED_COLOR_4) if (a.vis_every and fr % a.vis_every == 0) else None
        rows = []
        if fr in poses and fr in times:
            T, t = poses[fr], times[fr]
            meas = measurements(rs, T, geom, a)

            def make_row(trk, m, fr=fr, rs=rs):
                r = rs[m["idx"]]
                return dict(frame=fr, track_id=trk.id, det=int(r["det"]), cam=int(r["cam"]), x1=round(r["x1"], 1), y1=round(r["y1"], 1),
                            x2=round(r["x2"], 1), y2=round(r["y2"], 1), conf=round(r["conf"], 3), dist_m=r["dist_m"],
                            u_s=round(float(trk.kf.x[0] * 100.0 + 5000.0), 1), v_s=round(float(trk.kf.x[1] * 100.0 + 5000.0), 1))
            rows = tracker.step(fr, t, T[:3, 3], meas, make_row)
        elif rs:
            no_pose += 1
        trows += rows
        if img is not None:
            for r in rs:
                cv2.rectangle(img, (int(r["x1"] / 4), int(r["y1"] / 4)), (int(r["x2"] / 4), int(r["y2"] / 4)), (90, 90, 90), 1)
            for tr_ in [x for x in rows if x["frame"] == fr]:
                col = tuple(int(c) for c in PAL[tr_["track_id"] % 1024])
                cv2.rectangle(img, (int(tr_["x1"] / 4), int(tr_["y1"] / 4)), (int(tr_["x2"] / 4), int(tr_["y2"] / 4)), col, 3)
                cv2.putText(img, f"#{tr_['track_id']} {tr_['conf']:.2f} {tr_['dist_m']:.1f}m", (int(tr_["x1"] / 4), max(int(tr_["y1"] / 4) - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.putText(img, f"frame {fr:04d}  polar (world KF, bearing/range cost): colour = track id, grey = unmatched detection", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(out, "vis", f"{fr:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        # back-filled first rows of tracks confirmed in this frame: mark them on the earlier image
        if a.vis_every:
            for tr_ in [x for x in rows if x["frame"] != fr]:
                prev_path = os.path.join(out, "vis", f"{tr_['frame']:04d}.jpg")
                if os.path.exists(prev_path):
                    pim = cv2.imread(prev_path)
                    col = tuple(int(c) for c in PAL[tr_["track_id"] % 1024])
                    cv2.rectangle(pim, (int(tr_["x1"] / 4), int(tr_["y1"] / 4)), (int(tr_["x2"] / 4), int(tr_["y2"] / 4)), col, 3)
                    cv2.putText(pim, f"#{tr_['track_id']} {tr_['conf']:.2f} {tr_['dist_m']:.1f}m (start)", (int(tr_["x1"] / 4), max(int(tr_["y1"] / 4) - 5, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                    cv2.imwrite(prev_path, pim, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if fr_i % 100 == 0:
            print(f"  {fr_i + 1}/{len(frames)} frame {fr:04d}: {len(tracker.tracks)} live tracks ({time.time() - t0:.0f}s)", flush=True)
    if no_pose:
        print(f"  {no_pose} frames with detections but no pose (no LiDAR window): not tracked")
    return trows, tracker.backfilled, t0, tracker.stats
