"""Online 3D person tracker over the per-pass observations (design §3, §4).

Events in time order: observation batches (observations of one nodding pass, <= batch_gap_s apart; one global assignment) and
camera-frame ticks (lifecycle, observation-opportunity passes). The gate is 3D only (Mahalanobis d^2 on the horizontal position,
chi2 2 dof); a physical speed cap rejects pairs whose implied speed minus v_sigma_k sigma exceeds v_max; the 2D key changes the cost
of gated pairs only (link bonus / penalty). Each observation updates at most one track once (spawned tracks are initialised from it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from pipeline.track.track_kf import KF
from pipeline.track.track_obs import bev_overlap_frac
from pipeline.track.visibility import pass_outcome

LIVE = ("tentative", "confirmed", "lost")
INF = 1e9


def make_batches(obs: list, gap_s: float) -> list:
    """One batch = one nodding pass: observations within gap_s of the batch's FIRST observation (no chaining) and no 2D key twice
    (a repeated key is the next pass of the same person)."""
    obs = sorted(obs, key=lambda o: o.t_us)
    out = []
    for o in obs:
        if out and o.t_us - out[-1][0].t_us <= gap_s * 1e6 and not set(o.keys) & {k for q in out[-1] for k in q.keys}:
            out[-1].append(o)
        else:
            out.append([o])
    return out


@dataclass
class Track:
    id: int
    kf: KF
    status: str
    keys: list
    obs_ids: list
    t_first_us: int
    t_last_obs_us: int
    z_c: float
    box: dict
    last_z: np.ndarray
    last_sigma: float
    misses: int = 0
    lost_misses: int = 0
    t_last_eval_us: int = 0
    n_updates: int = 0
    end_reason: str = ""
    last_pass: str = ""
    obs_times: list = field(default_factory=list)
    frame_obs: dict = field(default_factory=dict)          # camera frame -> [(keys, xy, bbox)] of this track's observations


class Tracker:
    def __init__(self, cfg: dict, vis_fn=None):
        self.cfg, self.vis_fn = cfg, vis_fn
        self.tracks, self.events, self.obs_track, self.obs_info, self.links, self.nis, self.alive_at = {}, [], {}, {}, {}, [], {}
        self.next_id, self.t_prev, self.frame = 1, None, -1

    # ------------------------------------------------------------------ bookkeeping
    def _log(self, t_us, kind, track_id=-1, obs_id="", detail=""):
        self.events.append(dict(t_us=int(t_us), frame=self.frame, kind=kind, track_id=int(track_id), obs_id=obs_id, detail=detail))

    def _live(self):
        return [t for t in self.tracks.values() if t.status in LIVE]

    def _delete(self, trk, t_us, reason):
        trk.status, trk.end_reason = "deleted", reason
        self._log(t_us, "delete", trk.id, detail=f"reason={reason}")

    def _expire(self, t_us):
        lf = self.cfg["life"]
        for trk in self._live():
            if trk.status == "tentative" and t_us - trk.t_first_us > lf["tentative_max_s"] * 1e6:
                self._delete(trk, t_us, "tentative_timeout")
            elif t_us - trk.t_last_obs_us > lf["max_age_s"] * 1e6:
                self._delete(trk, t_us, "max_age")

    # ------------------------------------------------------------------ association
    def _d2(self, trk, o, sigma_a, cap=True):
        """Gating d^2; the regular hypothesis caps the predicted position std at gate_sigma_max (long unseen gaps), the manoeuvre
        hypothesis is not capped (it is already limited to manoeuvre_max_dt_s)."""
        kp = trk.kf.copy()
        kp.predict_to(o.t_us, sigma_a)
        return kp.d2(np.array([o.x, o.y]), o.sigma, self.cfg["assoc"]["gate_sigma_max"] if cap else None)

    def _pair(self, trk, o):
        """-> (cost, d2, v_impl, manoeuvre); cost INF if the pair is not allowed. Gate with sigma_a; if that fails and the last
        observation is at most manoeuvre_max_dt_s old, gate again with sigma_a_manoeuvre (the update then uses that noise)."""
        a = self.cfg["assoc"]
        d2, man = self._d2(trk, o, self.cfg["kf"]["sigma_a"]), False
        dt = (o.t_us - trk.t_last_obs_us) / 1e6
        v_impl = float(np.hypot(o.x - trk.last_z[0], o.y - trk.last_z[1]) / dt) if dt > 0 else 0.0
        s_v = math.sqrt(trk.last_sigma ** 2 + o.sigma ** 2) / dt if dt > 0 else 0.0
        if self._same_frame_conflict(trk, o):
            return INF, float("nan"), float("nan"), False
        gate = a["gate_loose"] if trk.status == "lost" else a["gate_confirmed"]     # loose gate only to re-acquire a lost track
        if d2 > gate and trk.status == "confirmed" and dt <= a["manoeuvre_max_dt_s"]:     # a velocity must exist to manoeuvre from
            d2m = self._d2(trk, o, a["sigma_a_manoeuvre"], cap=False)
            if d2m <= gate:
                d2, man = d2m, True
        if d2 > gate or (dt > 0 and v_impl - a["v_sigma_k"] * s_v > a["v_max"]):
            return INF, d2, v_impl, False
        mine = any(self.links.get(k) == trk.id for k in o.keys)
        other = any(k in self.links and self.links[k] != trk.id and self.tracks[self.links[k]].status in LIVE for k in o.keys)
        return d2 - (a["link_bonus"] if mine else 0.0) + (a["link_penalty"] if other and not mine else 0.0), d2, v_impl, man

    def _same_frame_conflict(self, trk, o) -> bool:
        """Two people seen in the same camera frame are two people: the track may not take an observation that shares a frame with one
        of its observations of disjoint 2D keys, unless the two are a duplicate candidate (pass boxes overlap or centres within
        obs.dup_candidate_m)."""
        for f in set(o.frames) & set(trk.frame_obs):
            for keys, xy, bbox in trk.frame_obs[f]:
                if set(keys) & set(o.keys):
                    continue
                if bev_overlap_frac(bbox, o.bbox) > 0 or math.hypot(xy[0] - o.x, xy[1] - o.y) <= self.cfg["obs"]["dup_candidate_m"]:
                    continue
                return True
        return False

    def _remember_frames(self, trk, o):
        for f in o.frames:
            trk.frame_obs.setdefault(f, []).append((list(o.keys), (o.x, o.y), dict(o.bbox)))

    def _link(self, trk, o, t_us):
        new = [k for k in o.keys if self.links.get(k) != trk.id]
        if new and trk.keys:
            prev = [self.links[k] for k in new if k in self.links]
            self._log(t_us, "merge_2d", trk.id, o.obs_id, f"prev_keys={'|'.join(trk.keys)};new_keys={'|'.join(new)}" + (f";from_track={prev[0]}" if prev else ""))
        for k in o.keys:
            self.links[k] = trk.id
            if k not in trk.keys:
                trk.keys.append(k)

    def _spawn(self, o, t_us):
        c = self.cfg
        trk = Track(id=self.next_id, kf=KF.init(np.array([o.x, o.y]), o.sigma, o.v0, c["obs"]["v0_sigma"], o.t_us), status="tentative", keys=[],
                    obs_ids=[o.obs_id], t_first_us=o.t_us, t_last_obs_us=o.t_us, z_c=o.z_c, box=dict(o.box), last_z=np.array([o.x, o.y]),
                    last_sigma=o.sigma, t_last_eval_us=o.t_us, obs_times=[o.t_us])
        self.next_id += 1
        self.tracks[trk.id] = trk
        self._remember_frames(trk, o)
        self._log(t_us, "spawn", trk.id, o.obs_id, f"keys={'|'.join(o.keys)}")
        self._link(trk, o, t_us)
        self.obs_track[o.obs_id] = trk.id
        return trk

    def _update(self, trk, o, d2, v_impl, man=False):
        trk.kf.predict_to(o.t_us, self.cfg["assoc"]["sigma_a_manoeuvre"] if man else self.cfg["kf"]["sigma_a"])
        nis = trk.kf.update(np.array([o.x, o.y]), o.sigma)
        self.nis.append(dict(obs_id=o.obs_id, quality=o.quality, range_m=o.range_m, nis=nis, status=trk.status, manoeuvre=man))   # NIS with the full S
        trk.n_updates += 1
        trk.obs_ids.append(o.obs_id)
        trk.obs_times.append(o.t_us)
        self._remember_frames(trk, o)
        trk.last_z, trk.last_sigma, trk.z_c, trk.box = np.array([o.x, o.y]), o.sigma, o.z_c, dict(o.box)
        if trk.status == "tentative" and o.t_us - trk.t_first_us >= self.cfg["life"]["confirm_min_dt_s"] * 1e6:
            trk.status = "confirmed"
            self._log(o.t_us, "confirm", trk.id, o.obs_id, f"n_obs={len(trk.obs_ids)}")
        elif trk.status == "lost":
            trk.status = "confirmed"
            self._log(o.t_us, "reassoc", trk.id, o.obs_id, f"gap_s={(o.t_us - trk.t_last_obs_us) / 1e6:.2f};d2={d2:.2f}")
        trk.t_last_obs_us = o.t_us
        trk.misses = trk.lost_misses = 0
        self._link(trk, o, o.t_us)
        self.obs_track[o.obs_id] = trk.id

    def process_batch(self, batch: list) -> None:
        self._expire(batch[0].t_us)
        live = self._live()
        C = np.full((len(live), len(batch)), INF)
        D2 = np.full_like(C, np.nan)
        V = np.full_like(C, np.nan)
        M = np.zeros(C.shape, bool)
        for i, trk in enumerate(live):
            for j, o in enumerate(batch):
                C[i, j], D2[i, j], V[i, j], M[i, j] = self._pair(trk, o)
        # stage 1: pairs inside the regular gate only; stage 2: the manoeuvre hypothesis for the tracks and observations left over,
        # so a manoeuvre pair can never displace a regular one (3D first, the 2D score only orders pairs inside a stage)
        pairs = []
        if len(live) and len(batch):
            C1 = np.where(M, INF, C)
            ri, ci = linear_sum_assignment(C1)
            pairs = [(i, j) for i, j in zip(ri, ci) if C1[i, j] < INF / 2]
            fi, fj = sorted(set(range(len(live))) - {i for i, _ in pairs}), sorted(set(range(len(batch))) - {j for _, j in pairs})
            if fi and fj:
                C2 = np.where(M, C, INF)[np.ix_(fi, fj)]
                ri, ci = linear_sum_assignment(C2)
                pairs += [(fi[i], fj[j]) for i, j in zip(ri, ci) if C2[i, j] < INF / 2]
        taken = {j: i for i, j in pairs}
        for j in sorted(range(len(batch)), key=lambda j: batch[j].t_us):
            o = batch[j]
            if j in taken:
                i = taken[j]
                self._update(live[i], o, D2[i, j], V[i, j], bool(M[i, j]))
                self.obs_info[o.obs_id] = dict(d2=float(D2[i, j]), v_impl=float(V[i, j]), reason="assigned_manoeuvre" if M[i, j] else "assigned")
                continue
            linked = [self.links[k] for k in o.keys if k in self.links and self.tracks[self.links[k]].status in LIVE]
            reason, d2, vi = "spawn", float("nan"), float("nan")
            if linked:
                T = linked[0]
                i = next((i for i, t in enumerate(live) if t.id == T), None)
                d2, vi = (float(D2[i, j]), float(V[i, j])) if i is not None else (float("nan"), float("nan"))
                # split_2d only if the linked track rejected it (gate / speed / same frame); taken by a better pair -> spawn_track_taken
                reason = "split_2d" if i is None or C[i, j] >= INF / 2 else "spawn_track_taken"
            trk = self._spawn(o, o.t_us)
            if reason == "split_2d":
                self._log(o.t_us, "split_2d", trk.id, o.obs_id, f"from_track={linked[0]};keys={'|'.join(o.keys)};d2={d2:.2f};v_impl={vi:.2f}")
            self.obs_info[o.obs_id] = dict(d2=d2, v_impl=vi, reason=reason)

    # ------------------------------------------------------------------ lifecycle
    def tick(self, frame: int, t_us: int) -> None:
        self.frame = frame
        lf = self.cfg["life"]
        self._expire(t_us)
        t_prev = self.t_prev if self.t_prev is not None else t_us - 500_000
        for trk in [t for t in self._live() if t.status in ("confirmed", "lost")]:
            passes = self.vis_fn(trk, t_prev, t_us, frame) if self.vis_fn is not None else None
            if passes is None:
                if trk.status == "confirmed" and t_us - max(trk.t_last_obs_us, trk.t_last_eval_us) > lf["time_lost_s"] * 1e6:
                    trk.status = "lost"
                    self._log(t_us, "lost", trk.id, detail="reason=time;lost_rule=time")
                continue
            for p in passes:
                observed = any(p["t0_us"] - 120_000 <= t <= p["t1_us"] + 120_000 for t in trk.obs_times)
                out = pass_outcome(p, p["opportunity"], observed)
                trk.last_pass, trk.t_last_eval_us = out, t_us
                self._log(t_us, "pass", trk.id, detail=f"outcome={out};t0={p['t0_us']};t1={p['t1_us']};n_scans={p.get('n_scans', 0)}")
                if out == "observed":
                    trk.misses = 0
                elif out in ("absent_strong", "absent_weak"):
                    if trk.status == "confirmed":
                        trk.misses += 1
                        if out == "absent_strong" or trk.misses >= lf["miss_to_lost"]:
                            trk.status, trk.lost_misses = "lost", 0
                            self._log(t_us, "lost", trk.id, detail="reason=" + ("miss_strong" if out == "absent_strong" else "miss_2"))
                    else:
                        trk.lost_misses += 1
                        if trk.lost_misses >= lf["miss_to_delete"]:
                            self._delete(trk, t_us, "miss")
                            break
        self.alive_at[frame] = [(t.id, t.status) for t in self.tracks.values() if t.status in LIVE and t.t_first_us <= t_us]
        self.t_prev = t_us

    def run(self, obs: list, ticks: list) -> None:
        items = [(b[0].t_us, 0, b) for b in make_batches(obs, self.cfg["assoc"]["batch_gap_s"])] + [(t, 1, (f, t)) for f, t in ticks]
        for t, kind, x in sorted(items, key=lambda e: (e[0], e[1])):
            if kind == 0:
                self.process_batch(x)
            else:
                self.tick(*x)
