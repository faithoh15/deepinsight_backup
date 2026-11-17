"""Offline per-track smoothing (design §5), track summaries and the output consistency checks (design §7)."""
from __future__ import annotations

import math

import numpy as np

from pipeline.track.track_kf import KF, cv_F, cv_Q, rts, run_filter

NAN = float("nan")


def segment_labels(frame_times: list, obs_times_us: list, obs_frames: set, gap_s: float) -> list:
    ts = sorted(obs_times_us)
    out = []
    for f, t in frame_times:
        if f in obs_frames:
            out.append("obs")
            continue
        k = int(np.searchsorted(ts, t))
        inside = 0 < k < len(ts) and ts[k] - ts[k - 1] <= gap_s * 1e6
        out.append("interp" if inside else "pred")
    return out


def speed_stats(v, Sv) -> tuple:
    v, Sv = np.asarray(v, float), np.asarray(Sv, float)
    s = float(np.hypot(*v))
    sg = float(math.sqrt(max((v / s) @ Sv @ (v / s), 0.0))) if s > 1e-6 else float(math.sqrt(max(np.trace(Sv) / 2, 0.0)))
    return s, sg, max(0.0, s - 1.96 * sg), s + 1.96 * sg


def _row(f, t, seg, x, P, x_f, single):
    r = dict(frame=int(f), t_img_us=int(t), segment=seg, x=float(x[0]), y=float(x[1]), sigma_x=float(math.sqrt(max(P[0, 0], 0))),
             sigma_y=float(math.sqrt(max(P[1, 1], 0))))
    if single:
        r.update(vx=NAN, vy=NAN, speed=NAN, speed_lo95=NAN, speed_hi95=NAN, sigma_vx=NAN, sigma_vy=NAN, sigma_speed=NAN, speed_filt=NAN)
    else:
        s, sg, lo, hi = speed_stats(x[2:], P[2:, 2:])
        r.update(vx=float(x[2]), vy=float(x[3]), speed=s, speed_lo95=lo, speed_hi95=hi, sigma_vx=float(math.sqrt(max(P[2, 2], 0))),
                 sigma_vy=float(math.sqrt(max(P[3, 3], 0))), sigma_speed=sg, speed_filt=float(np.hypot(*x_f[2:])))
    return r


def smooth_track(track_obs: list, frame_times: list, cfg: dict) -> list:
    """Kalman filter over the track's observations with the camera-frame times as predict-only steps, then RTS. Frames before the
    first observation are extrapolated backwards from the smoothed first state. One observation only: speed fields NaN."""
    if not track_obs:
        return []
    obs = sorted(track_obs, key=lambda o: o.t_us)
    sa = cfg["kf"]["sigma_a"]
    o0 = obs[0]
    kf0 = KF.init(np.array([o0.x, o0.y]), o0.sigma, o0.v0, cfg["obs"]["v0_sigma"], o0.t_us)
    steps = sorted([(o.t_us, 0, np.array([o.x, o.y]), o.sigma) for o in obs[1:]] + [(t, 1, None, None) for f, t in frame_times if t > o0.t_us],
                   key=lambda s: (s[0], s[1]))
    filt = run_filter(kf0, [(t, z, sz) for t, _, z, sz in steps], sa)
    sm = rts(filt)
    kinds = [None] + [k for _, k, _, _ in steps]
    at = {}
    for (xs, Ps), d, k in zip(sm, filt, kinds):
        if k == 1:
            at[d["t_us"]] = (xs, Ps, d["x_f"])
    labels = segment_labels(frame_times, [o.t_us for o in obs], {f for o in obs for f in o.frames}, cfg["post"]["interp_max_gap_s"])
    single = len(obs) == 1
    out = []
    for (f, t), seg in zip(frame_times, labels):
        if t in at:
            xs, Ps, xf = at[t]
        else:                                                    # before (or at) the first observation: backwards from the smoothed first state
            dt = (t - o0.t_us) / 1e6
            F = cv_F(dt)
            xs, Ps = F @ sm[0][0], F @ sm[0][1] @ F.T + cv_Q(abs(dt), sa)
            xf = F @ filt[0]["x_f"]
        out.append(_row(f, t, seg, xs, Ps, xf, single))
    return out


def summarize_track(track, frame_rows: list, events: list) -> dict:
    sp = [r["speed"] for r in frame_rows if r["segment"] in ("obs", "interp") and np.isfinite(r.get("speed", NAN))]
    frames = [r["frame"] for r in frame_rows if "frame" in r]
    seg = [r["segment"] for r in frame_rows]
    ev = [e for e in events if e["track_id"] == track.id]
    return dict(track_id=track.id, status_final=track.status, t_first_us=track.t_first_us, t_last_us=track.t_last_obs_us,
                frame_first=min(frames) if frames else -1, frame_last=max(frames) if frames else -1, n_obs=len(track.obs_ids), keys="|".join(track.keys),
                n_obs_frames=seg.count("obs"), n_interp_frames=seg.count("interp"), n_pred_frames=seg.count("pred"),
                speed_med=float(np.median(sp)) if sp else NAN, speed_p90=float(np.percentile(sp, 90)) if sp else NAN, speed_max=float(max(sp)) if sp else NAN,
                end_reason=track.end_reason, n_merge_2d=sum(e["kind"] == "merge_2d" for e in ev), n_split_2d=sum(e["kind"] == "split_2d" for e in ev))


def consistency_checks(obs: list, row_obs: dict, n_rows: int, obs_track: dict, tracks: dict, frame_rows: list) -> dict:
    ids = {o.obs_id for o in obs}
    in_tracks = [oid for t in tracks.values() for oid in t.obs_ids]
    frames_of = {t.id: {f for o in obs if o.obs_id in set(t.obs_ids) for f in o.frames} for t in tracks.values()}
    return dict(rows_all_mapped=len(row_obs) == n_rows and set(row_obs.values()) == ids,
                obs_used_once=sum(t.n_updates for t in tracks.values()) + len(tracks) == len(obs_track) == len(obs),
                obs_single_track=len(in_tracks) == len(set(in_tracks)) == len(obs) and all(obs_track[o] == t.id for t in tracks.values() for o in t.obs_ids),
                obs_frames_consistent=all(r["frame"] in frames_of[r["track_id"]] for r in frame_rows if r["segment"] == "obs"),
                rows_have_frame_rows=all((f, obs_track[oid]) in {(r["frame"], r["track_id"]) for r in frame_rows if r.get("has_det") in (1, "1", True)}
                                         for (f, k), oid in row_obs.items()))
