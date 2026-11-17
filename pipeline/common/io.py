"""CSV / PLY / JSON writers (results are CSV, PLY, PNG, JPG, MP4 and a config copy only)."""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from pipeline.common.recording import write_ply


def write_person_ply(path: str, person: dict) -> None:
    """Accumulated person points in the leveled device frame at the image time, with scan id and time relative to the image."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    P = person["xyz_lv"]
    write_ply(path, dict(x=P[:, 0].astype(np.float32), y=P[:, 1].astype(np.float32), z=P[:, 2].astype(np.float32),
                           scan_id=person["scan_id_pts"].astype(np.int32), time_rel_s=((person["t_pts"] - person["t_c"]) / 1e6).astype(np.float32),
                           in_main=person["in_main"].astype(np.uint8), range_m=person["range_pts"].astype(np.float32)))


def read_ply_fields(path: str) -> dict:
    """Reader for the binary little-endian PLY written by legacy.write_ply (named scalar fields)."""
    with open(path, "rb") as f:
        props, n = [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                t, name = line.split()[1:]
                props.append((name, {"float": "<f4", "double": "<f8", "int": "<i4", "short": "<i2", "uchar": "u1"}[t]))
            elif line == "end_header":
                break
        d = np.frombuffer(f.read(), dtype=np.dtype(props), count=n)
    return {name: d[name].copy() for name, _ in props}


def write_csv(path: str, rows: list, fields: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items() if k in fields})


def write_json(path: str, obj) -> None:
    json.dump(obj, open(path, "w"), indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
