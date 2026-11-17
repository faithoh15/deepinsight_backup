"""Stage result (2026-11-17): speed classes (저속 / 고속 / 초고속 / 미정) of the 3D tracks, per frame and per v3 box, and one combined
box table with the 3D id, speed, heading and the box in the corrected map frame -> 4_result/ (pipeline/result/export.py).
Reads 1_detection/persons.csv, 2_tracking/{tracks,track_frames,rows}.csv, 3_map/pose_refine/pose_correction.csv; changes none of them.
Outputs: tracks.csv, track_frames.csv (2_tracking tables + class columns), boxes.csv (one row per v3 box), summary.json.
"""
from __future__ import annotations

import json
import os

from pipeline.result.export import build_result
from stages._base import stage_main


def run(P, cfg, **kw):
    s = build_result(P.detection, P.tracking, os.path.join(P.pose, "pose_correction.csv"), os.path.join(P.odometry, "odometry.csv"),
                     P.result, cfg["result"])
    s = dict(dataset=P.name, **s)
    with open(os.path.join(P.result, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1, ensure_ascii=False)
    return s


if __name__ == "__main__":
    stage_main(run)
