#!/usr/bin/env python
"""Check the installation before the first run: configs/local.yaml, both conda environments (packages and the pipeline's own modules),
GPU, model files, result folder and every recording of configs/datasets.csv. Nothing is changed.

  python tools/check_setup.py            (exit code 1 when something must be fixed)
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

FINAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if FINAL not in sys.path:
    sys.path.insert(0, FINAL)

PACKAGES = {"main": ["numpy", "scipy", "cv2", "open3d", "yaml", "pandas"],
            "yolo": ["numpy", "scipy", "cv2", "yaml", "torch", "torchvision", "ultralytics", "lap"]}
YOLO_ONLY = ("pipeline.prep.yolo", "pipeline.prep.operator")             # import torch / ultralytics at module level
RECORDING = ("calib/config/camera.yaml", "calib/config/imu_camera0.yaml", "imu_data.csv", "lidar/", "images_cam0/", "images_cam1/")
ENV_FILE = {"main": "envs/environment-main.yml", "yolo": "envs/environment-yolo.yml"}
TAG = {"ok": "[확인]", "warn": "[경고]", "error": "[오류]"}


def env_modules(env: str) -> list:
    """packages + the stage modules of this environment (+ every pipeline module they may import lazily)."""
    from pipeline.runner import STAGES
    mods = list(PACKAGES[env]) + [f"stages.{s}" for s, e in STAGES if e == env]
    if env == "main":
        for f in sorted(glob.glob(os.path.join(FINAL, "pipeline", "**", "*.py"), recursive=True)):
            m = os.path.relpath(f, FINAL)[:-3].replace(os.sep, ".").replace(".__init__", "")
            if m not in YOLO_ONLY:
                mods.append(m)
    else:
        mods += list(YOLO_ONLY) + ["pipeline.prep.track2d"]
    return mods


def check_env(python: str, modules: list) -> tuple:
    if not (os.path.exists(python) or shutil.which(python)):
        return "error", f"python이 없습니다: {python}"
    code = ("import importlib, json, platform\nbad = []\n"
            f"for m in {modules!r}:\n"
            "    try:\n        importlib.import_module(m)\n"
            "    except Exception as e:\n        bad.append(f'{m} ({type(e).__name__}: {e})')\n"
            "print(json.dumps(dict(python=platform.python_version(), bad=bad)))")
    try:
        p = subprocess.run([python, "-c", code], cwd=FINAL, capture_output=True, text=True, timeout=600)
        r = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:                                              # python broken or crashed while importing
        return "error", f"{python} 실행 실패: {e}"
    if r["bad"]:
        return "error", f"Python {r['python']}, 불러오기 실패: " + "; ".join(r["bad"])[:600]
    return "ok", f"Python {r['python']}, 모듈 {len(modules)}개 정상"


def check_gpu(python: str) -> tuple:
    code = "import torch, json; c = torch.cuda.is_available(); print(json.dumps([c, torch.cuda.get_device_name(0) if c else '']))"
    try:
        ok, name = json.loads(subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=300).stdout.strip().splitlines()[-1])
    except Exception as e:
        return "warn", f"GPU 확인 실패: {e}"
    return ("ok", f"GPU 사용 가능: {name}") if ok else ("warn", "GPU를 쓸 수 없습니다: YOLO 단계가 CPU로 아주 느리게 돕니다 (NVIDIA 드라이버 확인)")


def check_model(path: str, sha256: str) -> tuple:
    from tools.download_models import sha256_of
    if not os.path.exists(path):
        return "error", f"없음: {path} -> python tools/download_models.py"
    if sha256_of(path) != sha256:
        return "warn", f"공개 결과를 만든 파일과 다릅니다(sha256): {path} — 돌아가지만 결과가 다를 수 있습니다"
    return "ok", f"{os.path.basename(path)} (sha256 일치)"


def check_recording(folder: str) -> list:
    """required items missing in one recording folder ('x/' = folder that is missing or empty)."""
    missing = []
    for item in RECORDING:
        p = os.path.join(folder, item.rstrip("/"))
        ok = os.path.isdir(p) and bool(os.listdir(p)) if item.endswith("/") else os.path.isfile(p)
        if not ok:
            missing.append(item)
    return missing


def main() -> int:
    from pipeline.common.config import LOCAL, load_config, load_datasets
    from pipeline.common.paths import source_of
    from tools.download_models import MODELS
    cfg, out = load_config(), []

    def say(level, msg):
        out.append(level)
        print(f"{TAG[level]} {msg}", flush=True)

    print("== 설정")
    say("ok", f"configs/local.yaml 사용") if os.path.exists(LOCAL) else \
        say("warn", "configs/local.yaml이 없어 기본값을 씁니다 (cp configs/local.example.yaml configs/local.yaml 후 경로 수정)")
    for k, v in cfg["paths"].items():
        print(f"       {k} = {v}")
    print("== 실행 환경")
    for env in ("main", "yolo"):
        lvl, msg = check_env(cfg["envs"][env], env_modules(env))
        say(lvl, f"{env}: {msg}" + (f"\n       -> conda env create -f {ENV_FILE[env]} 로 만들거나 configs/local.yaml의 envs.{env}를 고치세요"
                                   if lvl == "error" else ""))
        if env == "yolo" and lvl == "ok":
            say(*check_gpu(cfg["envs"][env]))
    print("== 모델")
    for key, path in cfg["models"].items():
        name = os.path.basename(path)
        say(*check_model(path, MODELS[name]["sha256"])) if name in MODELS else \
            say("ok" if os.path.exists(path) else "error", f"{key}: {path}" + ("" if os.path.exists(path) else " 없음"))
    print("== 결과 폴더")
    res = cfg["paths"]["results_root"]
    try:
        os.makedirs(res, exist_ok=True)
        probe = os.path.join(res, ".write_test")
        open(probe, "w").write("x")
        os.remove(probe)
        free = shutil.disk_usage(res).free / 1e9
        say("ok" if free > 20 else "warn", f"{res} 쓰기 가능, 남은 공간 {free:.0f} GB (녹화 하나에 결과 약 1~3 GB)")
    except OSError as e:
        say("error", f"{res}에 쓸 수 없습니다: {e}")
    ds = load_datasets()
    print(f"== 녹화 (configs/datasets.csv {len(ds)}개, 원본 폴더 {cfg['paths']['nas_root']})")
    for name in ds:
        try:
            src = source_of(name, ds)
        except (KeyError, ValueError) as e:
            say("error", str(e))
            continue
        folder = os.path.join(cfg["paths"]["nas_root"], src)
        if not os.path.isdir(folder):
            say("error", f"{name}: 원본 폴더가 없습니다 {folder}")
            continue
        miss = check_recording(folder)
        say("error", f"{name}: 빠진 것 {', '.join(miss)}") if miss else say("ok", name)
    n_err, n_warn = out.count("error"), out.count("warn")
    print(f"== 결과: 오류 {n_err}, 경고 {n_warn}" + (" — 실행 준비 완료" if not n_err else " — [오류] 항목을 고친 뒤 다시 확인하세요"))
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
