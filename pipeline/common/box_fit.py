"""Oriented 3D box fitting helpers (verbatim from track_boxes/make_boxes.py: fit_box, corners, pca_yaw, heading_arrow, rot2, EDGES)."""
from __future__ import annotations

import math

import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------------------------------
def rot2(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def fit_box(P, yaw, margin=0.0):
    """Tight box around points P (N,3) with horizontal axes rotated by yaw -> (cx, cy, cz, l, w, h)."""
    Rl = rot2(yaw)
    uv = (P[:, :2] - P[:, :2].mean(0)) @ Rl              # local coordinates (u along heading, v across)
    u0, u1 = uv[:, 0].min() - margin, uv[:, 0].max() + margin
    v0, v1 = uv[:, 1].min() - margin, uv[:, 1].max() + margin
    z0, z1 = P[:, 2].min() - margin, P[:, 2].max() + margin
    c_local = np.array([(u0 + u1) / 2, (v0 + v1) / 2])
    cxy = P[:, :2].mean(0) + Rl @ c_local
    return float(cxy[0]), float(cxy[1]), float((z0 + z1) / 2), float(u1 - u0), float(v1 - v0), float(z1 - z0)


def corners(cx, cy, cz, l, w, h, yaw):
    """8 corners (8,3): bottom 4 (counter-clockwise from +l,+w) then top 4."""
    Rl = rot2(yaw)
    out = []
    for dz in (-h / 2, h / 2):
        for du, dv in ((l / 2, w / 2), (-l / 2, w / 2), (-l / 2, -w / 2), (l / 2, -w / 2)):
            xy = np.array([cx, cy]) + Rl @ np.array([du, dv])
            out.append([xy[0], xy[1], cz + dz])
    return np.array(out)


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def heading_arrow(C, length=0.5):
    """box corners (8,3) -> (start, end) of a heading arrow on the bottom face: centre -> front face + length."""
    cen, front = C[:4].mean(0), (C[0] + C[3]) / 2
    d = front - cen; n = np.linalg.norm(d[:2])
    return cen, front + (d / n * length if n > 1e-9 else 0)


def pca_yaw(P):
    xy = P[:, :2] - P[:, :2].mean(0)
    w, V = np.linalg.eigh(xy.T @ xy)
    v = V[:, 1]
    return math.atan2(v[1], v[0])
