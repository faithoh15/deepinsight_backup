"""Copy one recording from the NAS to the local cache. Files are copied one by one (the gvfs AFP mount has hung under parallel reads);
a file is copied when it is missing in the cache or has another size, through '<file>.part' + rename so an interrupted copy never
leaves a truncated file under the final name. Nothing is ever deleted, the source is only read."""
from __future__ import annotations

import os
import shutil


def inventory(root: str) -> dict:
    """relative path -> size of every file below root (skipping '*.part')."""
    out = {}
    for r, _, fs in os.walk(root):
        for f in fs:
            if not f.endswith(".part"):
                p = os.path.join(r, f)
                out[os.path.relpath(p, root)] = os.path.getsize(p)
    return out


def fetch_recording(src: str, dst: str, echo=None) -> dict:
    if not os.path.isdir(src):
        raise FileNotFoundError(f"recording not found on the NAS (is it mounted?): {src}")
    want = inventory(src)
    have = inventory(dst) if os.path.isdir(dst) else {}
    todo = sorted(k for k, n in want.items() if have.get(k) != n)
    for i, rel in enumerate(todo):
        s, d = os.path.join(src, rel), os.path.join(dst, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(s, d + ".part")
        if os.path.getsize(d + ".part") != want[rel]:
            raise IOError(f"size mismatch after copy: {s}")
        os.replace(d + ".part", d)
        if echo and (i % 200 == 0 or i == len(todo) - 1):
            echo(f"copy {i + 1}/{len(todo)} {rel}")
    final = inventory(dst)
    missing = [k for k, n in want.items() if final.get(k) != n]
    if missing:
        raise IOError(f"{len(missing)} files still differ after the copy, e.g. {missing[:3]}")
    return dict(copied=len(todo), files=len(want), bytes=int(sum(want.values())))
