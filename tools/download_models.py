#!/usr/bin/env python
"""Download the official Ultralytics weights used by the pipeline to the paths of configs/pipeline.yaml `models` (default models/) and
check their sha256: the same files produced the published results. A file already present with the right hash is not downloaded again.

  python tools/download_models.py
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

FINAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FINAL not in sys.path:
    sys.path.insert(0, FINAL)

BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
MODELS = {
    "yolo26l.pt": dict(url=BASE + "yolo26l.pt", sha256="9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26"),          # person boxes
    "yolo26l-seg.pt": dict(url=BASE + "yolo26l-seg.pt", sha256="636024306410afa1732692322fba57d22ea2b1c2f07613fcee131a93d7dd380c"),  # operator mask
}


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch_model(url: str, dest: str, sha256: str, timeout: float = 60.0, echo=None) -> str:
    """'present' if dest already has this sha256, else download through dest.part -> 'downloaded'; a wrong hash raises ValueError
    and leaves neither dest nor dest.part."""
    if os.path.exists(dest) and sha256_of(dest) == sha256:
        return "present"
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    part = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(part, "wb") as f:
            total, n = int(r.headers.get("Content-Length") or 0), 0
            for b in iter(lambda: r.read(1 << 20), b""):
                f.write(b)
                n += len(b)
                if echo and total:
                    echo(f"\r  {os.path.basename(dest)} {n / 1e6:.0f}/{total / 1e6:.0f} MB")
        got = sha256_of(part)
        if got != sha256:
            raise ValueError(f"sha256 mismatch for {url}: {got} (expected {sha256})")
        os.replace(part, dest)
    finally:
        if os.path.exists(part):
            os.remove(part)
    return "downloaded"


def main() -> int:
    from pipeline.common.config import load_config
    models = load_config()["models"]
    bad = 0
    for key, dest in models.items():
        name = os.path.basename(dest)
        if name not in MODELS:
            print(f"[건너뜀] {key}: {dest} — 이 도구가 아는 공식 가중치가 아닙니다. 직접 넣어 주세요.")
            continue
        try:
            st = fetch_model(MODELS[name]["url"], dest, MODELS[name]["sha256"], echo=lambda s: print(s, end="", flush=True))
            print(("\n" if st == "downloaded" else "") + f"[확인] {name}: {'받음' if st == 'downloaded' else '이미 있음'} (sha256 일치) -> {dest}")
        except Exception as e:                                          # network error or wrong file
            bad += 1
            print(f"\n[오류] {name}: {e}\n        직접 받으려면: curl -L -o '{dest}' {MODELS[name]['url']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
