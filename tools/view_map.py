#!/usr/bin/env python
"""3D viewer of the final static map of one recording (results of build_map.py, read only).

  python view.py <name> map      (copy of 3d_detection/scripts/view_map.py; reads final/output/<name>/3_map/)

Keys: 1 = map before removal (colour = height), 2 = final static map (colour = height),
3 = overlay (grey = final map, red = removed persons, yellow = person-cylinder points kept as structure), 4 = residue diagnosis
(grey = final map, magenta = briefly seen kept points near a tracked person, cyan = briefly seen kept points without a tracked person),
C = ceiling cut (heights above 2.0 m hidden, on by default), F = floor (heights <= 0.1 m hidden), S = screenshot to <map>/screens/, Q = quit.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

from pipeline.common.config import load_config, load_datasets  # noqa: E402
from pipeline.common.io import read_ply_fields  # noqa: E402
from pipeline.common.paths import result_paths  # noqa: E402

MODES = {"1": "before", "2": "final", "3": "overlay", "4": "residual"}
H_LO, H_HI = 0.1, 2.0


def _height_rgb(h):
    c = cv2.applyColorMap(((np.asarray(h, float) - H_LO) / (H_HI - H_LO) * 255).clip(0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    return c[:, ::-1] / 255.0


def map_layers(mode: str, maps: dict, ceiling_cut: bool, hide_floor: bool):
    """-> (xyz (N,3), rgb (N,3) in 0-1) of one display mode. maps: name -> PLY fields (x, y, z, h, + track_id / cause)."""
    def pts(name, colour=None, sel=None):
        d = maps[name]
        h = np.asarray(d["h"], float)
        keep = np.ones(len(h), bool) if sel is None else sel
        if ceiling_cut:
            keep &= h <= H_HI
        if hide_floor:
            keep &= h > H_LO
        xyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(float)[keep]
        rgb = _height_rgb(h[keep]) if colour is None else np.tile(np.array(colour, float), (len(xyz), 1))
        return xyz, rgb

    if mode == "before":
        layers = [pts("before")]
    elif mode == "final":
        layers = [pts("static")]
    elif mode == "overlay":
        layers = [pts("static", (0.6, 0.6, 0.6)), pts("removed", (0.9, 0.15, 0.15)), pts("protected", (1.0, 0.85, 0.0))]
    else:
        c = np.asarray(maps["residual"]["cause"])
        layers = [pts("static", (0.45, 0.45, 0.45)), pts("residual", (1.0, 0.0, 1.0), c == 1), pts("residual", (0.0, 0.8, 1.0), c == 2)]
    return np.concatenate([l[0] for l in layers]), np.concatenate([l[1] for l in layers])


def main(argv=None):
    import open3d as o3d
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    a.dataset = a.name
    mdir = result_paths(load_config(), a.name, load_datasets()).mapping
    maps = {k: read_ply_fields(os.path.join(mdir, f"map_{k}.ply")) for k in ("before", "static", "removed", "protected", "residual")}
    state = dict(mode="final", ceiling=True, floor=False)
    v = o3d.visualization.VisualizerWithKeyCallback()
    v.create_window(f"final map: {a.dataset}", width=1400, height=1000)
    v.get_render_option().background_color = np.array([0.06, 0.06, 0.06])
    v.get_render_option().point_size = 2.0

    def load(first=False):
        ctr = v.get_view_control()
        prm = None if first else ctr.convert_to_pinhole_camera_parameters()
        xyz, rgb = map_layers(state["mode"], maps, state["ceiling"], state["floor"])
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        pc.colors = o3d.utility.Vector3dVector(rgb)
        v.clear_geometries()
        v.add_geometry(pc, reset_bounding_box=first)
        if prm is not None:
            ctr.convert_from_pinhole_camera_parameters(prm)
        print(f"mode {state['mode']}: {len(xyz)} voxels, ceiling cut {state['ceiling']}, floor hidden {state['floor']}")
        return False

    def set_mode(m):
        def f(_):
            state["mode"] = m
            return load()
        return f

    def toggle(k):
        def f(_):
            state[k] = not state[k]
            return load()
        return f

    def shot(_):
        os.makedirs(os.path.join(mdir, "screens"), exist_ok=True)
        n = len(os.listdir(os.path.join(mdir, "screens")))
        path = os.path.join(mdir, "screens", f"{n:03d}_{state['mode']}.png")
        v.capture_screen_image(path)
        print("saved", path)
        return False

    for key, m in MODES.items():
        v.register_key_callback(ord(key), set_mode(m))
    for key, fn in ((ord("C"), toggle("ceiling")), (ord("F"), toggle("floor")), (ord("S"), shot), (ord("Q"), lambda _: sys.exit(0))):
        v.register_key_callback(key, fn)
    load(first=True)
    v.run()


if __name__ == "__main__":
    main()
