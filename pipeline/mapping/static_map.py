"""Final static map (2026-11-16, design docs/2026-11-16-static-map-design.md).

Map domain: the points v3 could see (camera FOV and <= 10 m at measurement, mount / operator masks, a v3 camera frame within half a
frame step); every other point is dropped as if it never existed. Removal: every tracked person (3D id) whatever its speed. A point is
removed when, at its own measurement time, it lies inside the cylinder around the person's RTS-smoothed position (radius from the nearest
v3 box, height above the local floor). A candidate is kept when its 0.1 m voxel also holds non-tube points spanning more than 3 s
(static structure next to the person)."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np
import yaml

from pipeline.detect.ground import to_level
from pipeline.track.track_obs import Obs
from pipeline.track.track_post import smooth_track

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = 1_000_000


def load_map_config(path: str | None = None) -> dict:
    if path:
        return yaml.safe_load(open(path, encoding="utf-8"))
    from pipeline.common.config import load_config
    return load_config()["map"]               # configs/pipeline.yaml map (= map.yaml)


# ------------------------------------------------------------------ voxels
_OFF, _B = 1 << 20, 21


def voxel_keys(xyz: np.ndarray, size: float) -> np.ndarray:
    k = np.floor(np.asarray(xyz, float) / size).astype(np.int64) + _OFF
    return (k[:, 0] << (2 * _B)) | (k[:, 1] << _B) | k[:, 2]


class VoxelSum:
    """Streaming per-voxel mean of positions and extra columns, with counts."""

    def __init__(self, size: float, flush: int = 2_000_000):
        self.size, self.flush = size, flush
        self.keys, self.sums, self.cnt = np.zeros(0, np.int64), None, np.zeros(0, np.int64)
        self.buf, self.names = [], None

    def add(self, xyz: np.ndarray, cols: dict | None = None) -> None:
        if len(xyz) == 0:
            return
        cols = cols or {}
        if self.names is None:
            self.names = sorted(cols)
        vals = np.column_stack([np.asarray(xyz, float)] + [np.asarray(cols[n], float)[:, None] for n in self.names])
        self.buf.append((voxel_keys(xyz, self.size), vals))
        if sum(len(k) for k, _ in self.buf) > self.flush:
            self._reduce()

    def _reduce(self) -> None:
        if not self.buf:
            return
        k = np.concatenate([self.keys] + [b[0] for b in self.buf])
        width = self.buf[0][1].shape[1]
        s = np.concatenate(([self.sums] if self.sums is not None else []) + [b[1] for b in self.buf])
        c = np.concatenate([self.cnt] + [np.ones(len(b[0]), np.int64) for b in self.buf])
        assert s.shape[1] == width
        o = np.argsort(k, kind="stable")
        k, s, c = k[o], s[o], c[o]
        st = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        self.keys, self.sums, self.cnt = k[st], np.add.reduceat(s, st, axis=0), np.add.reduceat(c, st)
        self.buf = []

    def result(self):
        """-> (mean xyz (N,3), {name: mean (N,)}, count (N,))."""
        self._reduce()
        if self.sums is None:
            return np.zeros((0, 3)), {n: np.zeros(0) for n in (self.names or [])}, np.zeros(0, np.int64)
        m = self.sums / self.cnt[:, None]
        return m[:, :3], {n: m[:, 3 + i] for i, n in enumerate(self.names)}, self.cnt


class VoxelSpan:
    """Streaming per-voxel first / last measurement time (us) of the points added."""

    def __init__(self, flush: int = 4_000_000):
        self.flush = flush
        self.keys, self.t0, self.t1 = np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
        self.buf = []

    def add(self, keys: np.ndarray, t_us: np.ndarray) -> None:
        if len(keys):
            self.buf.append((np.asarray(keys, np.int64), np.asarray(t_us, np.int64)))
            if sum(len(k) for k, _ in self.buf) > self.flush:
                self._reduce()

    def _reduce(self) -> None:
        if not self.buf:
            return
        k = np.concatenate([self.keys] + [b[0] for b in self.buf])
        a = np.concatenate([self.t0] + [b[1] for b in self.buf])
        b = np.concatenate([self.t1] + [b[1] for b in self.buf])
        o = np.argsort(k, kind="stable")
        k, a, b = k[o], a[o], b[o]
        st = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        self.keys, self.t0, self.t1 = k[st], np.minimum.reduceat(a, st), np.maximum.reduceat(b, st)
        self.buf = []

    def span_s(self, keys: np.ndarray) -> np.ndarray:
        """Time span (s) of the voxels' points; 0 for voxels never added."""
        self._reduce()
        keys = np.asarray(keys, np.int64)
        i = np.clip(np.searchsorted(self.keys, keys), 0, max(len(self.keys) - 1, 0))
        out = np.zeros(len(keys))
        if len(self.keys):
            hit = self.keys[i] == keys
            out[hit] = (self.t1[i[hit]] - self.t0[i[hit]]) / S
        return out


def isolated_from(keys: np.ndarray, struct_keys: np.ndarray) -> np.ndarray:
    """True where neither the voxel nor any of its 26 neighbours is in struct_keys (same voxel size and packing as voxel_keys)."""
    keys = np.asarray(keys, np.int64)
    st = np.unique(np.asarray(struct_keys, np.int64))
    hit = np.zeros(len(keys), bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                hit |= np.isin(keys + (dx << (2 * _B)) + (dy << _B) + dz, st)
    return ~hit


def structure_keys(vs: "VoxelSpan", span_s: float) -> np.ndarray:
    """Voxels whose added points span more than span_s (long-observed structure)."""
    vs._reduce()
    return vs.keys[(vs.t1 - vs.t0) > span_s * S]

def protect_mask(cand_keys: np.ndarray, vs: VoxelSpan, span_s: float) -> np.ndarray:
    """Candidates whose voxel holds non-tube points spanning more than span_s -> True (static structure, kept)."""
    return vs.span_s(cand_keys) > span_s


def scan_in_frames(scan_t_us: np.ndarray, frame_t_us: np.ndarray, half_s: float, gap_max_s: float) -> np.ndarray:
    """Scan times covered by v3 camera frames: between two consecutive frames at most gap_max_s apart (frame steps vary 0.3-0.7 s),
    or within half_s before the first / after the last frame."""
    ft = np.sort(np.asarray(frame_t_us, np.int64))
    t = np.asarray(scan_t_us, np.int64)
    ok = (t >= ft[0] - half_s * S) & (t <= ft[-1] + half_s * S)
    i = np.clip(np.searchsorted(ft, t), 1, max(len(ft) - 1, 1))
    inner = (t > ft[0]) & (t < ft[-1])
    if len(ft) > 1:
        ok &= ~inner | (ft[i] - ft[i - 1] <= gap_max_s * S)
    return ok


# ------------------------------------------------------------------ tubes
@dataclass
class Tube:
    track_id: int
    t: np.ndarray                      # sample times (us), ascending, inside the spans only
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray                      # odometry z of the nearest observation's box centre (only used for the leveling rotation)
    r: np.ndarray
    top: np.ndarray                    # cylinder top, height above the local floor
    spans: list = field(default_factory=list)


def tube_spans(t_obs_us: list, margin_s: float, gap_s: float) -> list:
    """Observation times -> [(t0, t1)]: observations closer than gap_s form one span, widened by margin_s on both sides."""
    t = sorted(int(v) for v in t_obs_us)
    out, a = [], t[0]
    for p, q in zip(t, t[1:] + [None]):
        if q is None or q - p > gap_s * S:
            out.append((int(a - margin_s * S), int(p + margin_s * S)))
            a = q
    return out


def tube_radius(l, w, sigma, mcfg: dict):
    c = mcfg["tube"]
    r = np.hypot(l, w) / 2 + c["r_margin"] + np.minimum(sigma, c["sigma_cap"])
    return np.minimum(r, c["r_max"]) if np.ndim(r) else float(min(r, c["r_max"]))


def obs_from_csv(rows: list) -> dict:
    """observations.csv rows -> {track_id: [Obs]} (rows without a track left out)."""
    out = {}
    for r in rows:
        tid = int(r["track_id"])
        if tid < 0:
            continue
        box = dict(l=float(r["l"]), w=float(r["w"]), h=float(r["h"]), yaw_deg=float(r["yaw_deg"]))
        out.setdefault(tid, []).append(Obs(obs_id=r["obs_id"], t_us=int(r["t_us"]), keys=r["keys"].split("|"),
                                           frames=[int(f) for f in r["frames"].split("|") if f], rows=[], x=float(r["x"]), y=float(r["y"]),
                                           z_c=float(r["z_c"]), sigma=float(r["sigma"]), quality=r["quality"], range_m=float(r["range_m"]), box=box,
                                           bbox=dict(cx=float(r["x"]), cy=float(r["y"]), l=box["l"], w=box["w"], yaw_deg=box["yaw_deg"]),
                                           v0=np.array([float(r["v0x"]), float(r["v0y"])]), n_pts=int(r["n_pts"])))
    return out


def build_tubes(obs_by_track: dict, tcfg: dict, mcfg: dict) -> list:
    c = mcfg["tube"]
    tcfg = dict(tcfg, kf=dict(tcfg["kf"], sigma_a=c["sigma_a"]))     # map smoothing noise (the tracker config is not changed)
    out = []
    for tid, obs in sorted(obs_by_track.items()):
        obs = sorted(obs, key=lambda o: o.t_us)
        spans = tube_spans([o.t_us for o in obs], c["margin_s"], c["gap_s"])
        ts = np.unique(np.concatenate([np.r_[np.arange(a, b, int(c["sample_dt_s"] * S)), b] for a, b in spans]).astype(np.int64))
        sm = smooth_track(obs, [(-(i + 1), int(t)) for i, t in enumerate(ts)], tcfg)
        to = np.array([o.t_us for o in obs])
        near = np.abs(ts[:, None] - to[None, :]).argmin(1)
        sig = np.array([max(r["sigma_x"], r["sigma_y"]) for r in sm])
        L_, W_ = np.array([obs[k].box["l"] for k in near]), np.array([obs[k].box["w"] for k in near])
        out.append(Tube(track_id=int(tid), t=ts, x=np.array([r["x"] for r in sm]), y=np.array([r["y"] for r in sm]),
                        z=np.array([obs[k].z_c for k in near]), r=tube_radius(L_, W_, sig, mcfg),
                        top=c["z_lo"] + np.array([obs[k].box["h"] for k in near]) + c["top_margin"], spans=spans))
    return out


def in_tubes(tubes: list, X_odom: np.ndarray, t_us: np.ndarray, plane: dict, mcfg: dict) -> np.ndarray:
    """Track id of the tube holding each point at its own time (nearest centre if several), -1 outside every tube.
    plane: dict(R_level, ground_z, R_c, p_c) of a camera frame near the points' time (height above that frame's floor)."""
    X_odom = np.asarray(X_odom, float)
    t_us = np.asarray(t_us, np.int64)
    tid = np.full(len(X_odom), -1, int)
    if len(X_odom) == 0:
        return tid
    best = np.full(len(X_odom), np.inf)
    lv = to_level((X_odom - plane["p_c"]) @ plane["R_c"], plane)
    z_lo = mcfg["tube"]["z_lo"]
    for tb in tubes:
        m = np.zeros(len(t_us), bool)
        for a, b in tb.spans:
            m |= (t_us >= a) & (t_us <= b)
        idx = np.flatnonzero(m & (lv[:, 2] > z_lo))
        if len(idx) == 0:
            continue
        tt = t_us[idx].astype(float)
        C = np.stack([np.interp(tt, tb.t, tb.x), np.interp(tt, tb.t, tb.y), np.interp(tt, tb.t, tb.z)], 1)
        cl = to_level((C - plane["p_c"]) @ plane["R_c"], plane)
        d = np.hypot(lv[idx, 0] - cl[:, 0], lv[idx, 1] - cl[:, 1])
        ok = (d <= np.interp(tt, tb.t, tb.r)) & (lv[idx, 2] <= np.interp(tt, tb.t, tb.top)) & (d < best[idx])
        tid[idx[ok]], best[idx[ok]] = tb.track_id, d[ok]
    return tid


def tube_distance(tubes: list, X_odom: np.ndarray, t_us: np.ndarray, plane: dict) -> np.ndarray:
    """Horizontal distance (leveled frame) to the nearest cylinder centre active at each point's own time; inf when none is active.
    Diagnosis only (which kept points lie just outside a tracked person), never used to remove points."""
    X_odom = np.asarray(X_odom, float)
    t_us = np.asarray(t_us, np.int64)
    best = np.full(len(X_odom), np.inf)
    if len(X_odom) == 0:
        return best
    lv = to_level((X_odom - plane["p_c"]) @ plane["R_c"], plane)
    for tb in tubes:
        m = np.zeros(len(t_us), bool)
        for a, b in tb.spans:
            m |= (t_us >= a) & (t_us <= b)
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        tt = t_us[idx].astype(float)
        C = np.stack([np.interp(tt, tb.t, tb.x), np.interp(tt, tb.t, tb.y), np.interp(tt, tb.t, tb.z)], 1)
        cl = to_level((C - plane["p_c"]) @ plane["R_c"], plane)
        best[idx] = np.minimum(best[idx], np.hypot(lv[idx, 0] - cl[:, 0], lv[idx, 1] - cl[:, 1]))
    return best

def read_csv(path: str) -> list:
    return list(csv.DictReader(open(path)))
