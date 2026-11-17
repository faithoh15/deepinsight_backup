"""Common entry of every stage module: `python -m stages.<stage> --name NAME [--config PATH] [--datasets PATH]`."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from pipeline.common.config import load_config, load_datasets
from pipeline.common.paths import Paths, result_paths


def open_run(P: Paths):
    from pipeline.common.recording import Run
    from pipeline.prep.clock import read_clock
    return Run(P.data, read_clock(P.clock))


def stage_main(fn) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--datasets", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    P = result_paths(cfg, a.name, load_datasets(a.datasets) if a.datasets else None)
    os.makedirs(P.root, exist_ok=True)
    t0 = time.time()
    summary = fn(P, cfg) or {}
    print(json.dumps(dict(summary, stage_seconds=round(time.time() - t0, 1)), ensure_ascii=False, default=str), flush=True)
    sys.stdout.flush()
