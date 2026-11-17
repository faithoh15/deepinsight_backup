"""Every folder of one recording: NAS source, local cache copy and the result folders final/output/<name>/..."""
from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass

from pipeline.common.config import load_datasets

SUFFIX = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{6})$")


def source_of(name: str, datasets: dict | None = None) -> str:
    """NAS folder of a dataset name = the date suffix of the name; the name must be listed and its listed source must agree."""
    datasets = load_datasets() if datasets is None else datasets
    if name not in datasets:
        close = difflib.get_close_matches(name, list(datasets), n=3, cutoff=0.6)
        raise KeyError(f"unknown dataset name {name!r} (configs/datasets.csv)" + (f"; did you mean: {', '.join(close)}" if close else ""))
    m = SUFFIX.search(name)
    if not m or m.group(1) != datasets[name]:
        raise ValueError(f"{name!r}: date suffix {m.group(1) if m else None} != listed source {datasets[name]!r}")
    return datasets[name]


@dataclass
class Paths:
    name: str
    source: str
    nas: str
    data: str
    root: str
    prep: str
    clock: str
    odometry: str
    masks: str
    sidebyside: str
    yolo: str
    operator_mask: str
    track2d: str
    detection: str
    tracking: str
    mapping: str
    pose: str
    result: str
    logs: str
    run_json: str


def result_paths(cfg: dict, name: str, datasets: dict | None = None) -> Paths:
    src = source_of(name, datasets)
    root = os.path.join(cfg["paths"]["results_root"], name)
    prep = os.path.join(root, "0_prep")
    mapping = os.path.join(root, "3_map")
    return Paths(name=name, source=src, nas=os.path.join(cfg["paths"]["nas_root"], src), data=os.path.join(cfg["paths"]["cache_root"], src),
                 root=root, prep=prep, clock=os.path.join(prep, "clock.yaml"), odometry=os.path.join(prep, "odometry"),
                 masks=os.path.join(prep, "masks"), sidebyside=os.path.join(prep, "sidebyside"), yolo=os.path.join(prep, "yolo"),
                 operator_mask=os.path.join(prep, "operator_mask"), track2d=os.path.join(prep, "track2d"),
                 detection=os.path.join(root, "1_detection"), tracking=os.path.join(root, "2_tracking"), mapping=mapping,
                 pose=os.path.join(mapping, "pose_refine"), result=os.path.join(root, "4_result"), logs=os.path.join(root, "logs"), run_json=os.path.join(root, "run.json"))
