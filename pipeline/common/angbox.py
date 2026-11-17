"""Angular person boxes of the side-by-side images (proposed_pipeline/fusion_common.py: AngBox, angles_to_box, box_to_angles, tile sizes, track palette).
Verbatim."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


W_IMG, H_IMG = 8000, 3000


TILE_W, TILE_H = 4000, 3000


PAL_TRACK = np.random.default_rng(3).integers(60, 255, (1024, 3))


def angles_to_box(cam, yaw, pitch, dyaw, dpitch):
    uc = (yaw + 90.0) / 180.0 * (TILE_W - 1) + cam * TILE_W
    vc = (pitch + 60.0) / 120.0 * (TILE_H - 1)
    w = dyaw / 180.0 * (TILE_W - 1)
    h = dpitch / 120.0 * (TILE_H - 1)
    return uc - w / 2, vc - h / 2, uc + w / 2, vc + h / 2


# ----------------------------------------------------------------------------------------------------------------------
# boxes
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class AngBox:
    key: str                 # "t<track_id>" or "d<frame>_<det>"
    track_id: int            # -1 for detection-only boxes
    source: str              # 'track' | 'det'
    cam: int
    yaw: float
    pitch: float
    dyaw: float
    dpitch: float
    conf: float              # detector confidence (min of the two rows)
    dist_m: float            # distance of the box centre from the device at t (world mode) or from its height
    height_px: float
    truncated: int
    interp_mode: str         # 'world' | 'dir' | 'single'
    box_conf: float
    f0: int
    f1: int
    t0_us: int
    t1_us: int
    world_xyz: np.ndarray    # box centre point in the odometry frame (interpolated)
    det_frame: int = -1
    det_id: int = -1

    def pixels(self):
        return angles_to_box(self.cam, self.yaw, self.pitch, self.dyaw, self.dpitch)

    @property
    def dt_s(self):
        return (self.t1_us - self.t0_us) / 1e6


def box_to_angles(x1, y1, x2, y2):
    """Side-by-side box -> (cam, yaw, pitch, dyaw, dpitch) in degrees, in the tile's camera frame (exact: the tile is
    equirectangular)."""
    uc, vc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cam = 0 if uc < TILE_W else 1
    yaw = (uc - cam * TILE_W) / (TILE_W - 1) * 180.0 - 90.0
    pitch = vc / (TILE_H - 1) * 120.0 - 60.0
    dyaw = (x2 - x1) / (TILE_W - 1) * 180.0
    dpitch = (y2 - y1) / (TILE_H - 1) * 120.0
    return cam, yaw, pitch, dyaw, dpitch
