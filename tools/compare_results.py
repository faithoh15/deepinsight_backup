#!/usr/bin/env python
"""Compare a reference result folder with a new one: CSV (bytes, else per-column differences with processing-time columns ignored),
PLY (fields), NPZ (arrays), JPG/PNG (bytes). JSON and logs are not compared (they hold paths and run times).

  python tools/compare_results.py REF NEW [--files a.csv b/c.ply ...] [--ignore-cols seconds ...]   (exit 1 if anything differs)
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import os
import sys

import numpy as np

FINAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FINAL not in sys.path:
    sys.path.insert(0, FINAL)   # noqa: allowed for the CLI entry only (final's own root)

EXT = (".csv", ".ply", ".npz", ".jpg", ".png")


def _csv_diff(pa, pb, ignore):
    ra, rb = list(csv.reader(open(pa, encoding="utf-8"))), list(csv.reader(open(pb, encoding="utf-8")))
    if not ra or not rb or ra[0] != rb[0]:
        return f"header differs: {ra[0] if ra else None} vs {rb[0] if rb else None}"
    if len(ra) != len(rb):
        return f"rows {len(ra) - 1} vs {len(rb) - 1}"
    head = ra[0]
    keep = [i for i, h in enumerate(head) if h not in ignore]
    worst = {}
    for x, y in zip(ra[1:], rb[1:]):
        for i in keep:
            if x[i] != y[i]:
                try:
                    d = abs(float(x[i]) - float(y[i]))
                except ValueError:
                    d = float("inf")
                worst[head[i]] = max(worst.get(head[i], 0.0), d)
    if not worst:
        return None
    return "values differ: " + ", ".join(f"{k} (max {v:.3g})" for k, v in sorted(worst.items(), key=lambda kv: -kv[1])[:6])


def _ply_diff(pa, pb):
    from pipeline.common.io import read_ply_fields
    a, b = read_ply_fields(pa), read_ply_fields(pb)
    if set(a) != set(b):
        return f"fields {sorted(a)} vs {sorted(b)}"
    bad = [k for k in a if not np.array_equal(np.asarray(a[k]), np.asarray(b[k]))]
    return f"fields differ: {bad}" if bad else None


def _npz_diff(pa, pb):
    a, b = np.load(pa), np.load(pb)
    if set(a.files) != set(b.files):
        return f"keys {sorted(a.files)} vs {sorted(b.files)}"
    bad = [k for k in a.files if not np.array_equal(a[k], b[k])]
    return f"arrays differ: {bad}" if bad else None


def compare_dirs(ref: str, new: str, files: list | None = None, ignore_cols=("seconds",)) -> list:
    """-> list of difference descriptions ([] = identical)."""
    if files is None:
        files = sorted(os.path.relpath(os.path.join(r, f), ref) for r, _, fs in os.walk(ref) for f in fs if f.endswith(EXT))
    out = []
    for rel in files:
        pa, pb = os.path.join(ref, rel), os.path.join(new, rel)
        if not os.path.exists(pb):
            out.append(f"{rel}: missing in new")
            continue
        if filecmp.cmp(pa, pb, shallow=False):
            continue
        d = (_csv_diff(pa, pb, set(ignore_cols)) if rel.endswith(".csv") else _ply_diff(pa, pb) if rel.endswith(".ply")
             else _npz_diff(pa, pb) if rel.endswith(".npz") else "bytes differ")
        if d:
            out.append(f"{rel}: {d}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--ignore-cols", nargs="*", default=["seconds"])
    a = ap.parse_args()
    d = compare_dirs(a.ref, a.new, a.files, a.ignore_cols)
    n = len(a.files) if a.files is not None else sum(1 for r, _, fs in os.walk(a.ref) for f in fs if f.endswith(EXT))
    for x in d:
        print("DIFF", x)
    print(f"{n} files compared, {len(d)} different")
    sys.exit(1 if d else 0)


if __name__ == "__main__":
    main()
