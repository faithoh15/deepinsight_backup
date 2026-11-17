#!/usr/bin/env python
"""3D viewers of the results of one recording.

  python view.py NAME det [--frame N] [--track] [--scan K] [--screenshot]   v3 boxes and person points (1_detection; --track: 3D ids of 2_tracking)
  python view.py NAME map                                                   final static map (3_map): keys 1 before / 2 final / 3 overlay / 4 residue
"""
from __future__ import annotations

import os
import sys

FINAL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FINAL)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2 or argv[1] not in ("det", "map"):
        print(__doc__)
        sys.exit(2)
    name, mode, rest = argv[0], argv[1], argv[2:]
    from pipeline.common.paths import source_of
    try:
        source_of(name)
    except (KeyError, ValueError) as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(2)
    if mode == "det":
        from tools.view_3d import main as m
    else:
        from tools.view_map import main as m
    m(["--name", name] + rest)


if __name__ == "__main__":
    main()
