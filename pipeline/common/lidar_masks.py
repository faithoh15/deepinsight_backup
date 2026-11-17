"""Mount (self) mask voxel keys and the learned operator sector mask test (verbatim from proposed_pipeline/build_windows.py: voxel_key,
operator_hit)."""
from __future__ import annotations

import numpy as np


def voxel_key(xyz, voxel):
    c = np.floor(xyz / voxel).astype(np.int64) + 100_000
    return (c[:, 0] * 200_000 + c[:, 1]) * 200_000 + c[:, 2]


def operator_hit(xyz, rng, a, mask):
    """Points of one scan (device frame) that belong to the operator: learned sector mask within operator-range, or
    horizontal distance < operator-core."""
    na, nz = mask.shape
    zlo, zhi = a["operator_z"]
    hit = np.hypot(xyz[:, 0], xyz[:, 1]) < a["operator_core"]
    near = rng < a["operator_range"]
    p = xyz[near]
    az = ((np.degrees(np.arctan2(p[:, 1], p[:, 0])) + 360) % 360 / a["operator_az_step"]).astype(int) % na
    zb = ((p[:, 2] - zlo) / a["operator_z_step"]).astype(int)
    ok = (zb >= 0) & (zb < nz)
    h = np.zeros(len(p), bool)
    h[ok] = mask[az[ok], zb[ok]]
    hit[np.nonzero(near)[0][h]] = True
    return hit
