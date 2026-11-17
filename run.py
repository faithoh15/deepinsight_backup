#!/usr/bin/env python
"""Run the whole pipeline for one or more recordings: NAS copy -> prep -> v3 detection -> 3D tracking -> pose-corrected final map.

  python run.py NAME [NAME ...]        names from configs/datasets.csv (result folder = output/<NAME>/)
  python run.py --all                  every listed recording
  options: --from STAGE --to STAGE (range), --force (rerun stages already done), --list (names and their state)
Stages: fetch clock odometry masks unwrap yolo operator track2d detection tracking pose map. A failed recording does not stop the
others; the failures are listed at the end (exit code 1).
"""
from __future__ import annotations

import argparse
import os
import sys

FINAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FINAL)
from pipeline.common.config import load_config, load_datasets  # noqa: E402
from pipeline.common.paths import result_paths, source_of  # noqa: E402
from pipeline.runner import STAGES, run_recording  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--from", dest="frm", default=None)
    ap.add_argument("--to", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--config", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--datasets", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--stage-module", default="stages", help=argparse.SUPPRESS)       # tests only
    ap.add_argument("--stage-list", default=None, help=argparse.SUPPRESS)             # tests only: "a:main,b:yolo"
    a = ap.parse_args()
    cfg = load_config(a.config)
    ds = load_datasets(a.datasets)
    stages = [tuple(x.split(":")) for x in a.stage_list.split(",")] if a.stage_list else STAGES
    if a.list:
        for n in ds:
            P = result_paths(cfg, n, ds)
            done = sum(os.path.exists(os.path.join(P.logs, f"{s}.done")) for s, _ in stages)
            print(f"{n:<42s} {done:2d}/{len(stages)}")
        return
    names = list(ds) if a.all else a.names
    if not names:
        ap.error("give dataset names or --all")
    for n in names:
        try:
            source_of(n, ds)
        except (KeyError, ValueError) as e:
            print(f"오류: {e}", file=sys.stderr)
            sys.exit(2)
    for s in (a.frm, a.to):
        if s and s not in [x for x, _ in stages]:
            print(f"오류: unknown stage {s!r}; stages: {' '.join(x for x, _ in stages)}", file=sys.stderr)
            sys.exit(2)
    failed, warnings = [], []
    for i, n in enumerate(names):
        print(f"=== [{i + 1}/{len(names)}] {n}", flush=True)
        rj = run_recording(n, cfg, stages, a.frm, a.to, a.force, a.stage_module, a.config, a.datasets,
                           echo=lambda s: print(s, flush=True))
        if rj.get("failed_stage"):
            failed.append((n, rj["failed_stage"]))
        warnings += [(n, s, st["warning"]) for s, st in rj.get("stages", {}).items() if st.get("warning")]
    for n, s, w in warnings:
        print(f"경고: {n} ({s}) {w}")
    if failed:
        print("실패: " + ", ".join(f"{n} ({s})" for n, s in failed))
        sys.exit(1)
    print("모두 완료")


if __name__ == "__main__":
    main()
