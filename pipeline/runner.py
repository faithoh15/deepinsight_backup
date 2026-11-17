"""Stage order and execution of one recording. Each stage runs as its own process with the python of its environment; a stage is
done when logs/<stage>.done exists (written only after success). Running a stage invalidates the later stages (their .done is removed)."""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import time

from pipeline.common.config import FINAL, load_datasets
from pipeline.common.paths import result_paths

STAGES = [("fetch", "main"), ("clock", "main"), ("odometry", "main"), ("masks", "main"), ("unwrap", "main"), ("yolo", "yolo"),
          ("operator", "yolo"), ("track2d", "yolo"), ("detection", "main"), ("tracking", "main"), ("pose", "main"), ("map", "main"),
          ("result", "main")]


def _last_json(log_path):
    """the stage summary = the JSON object printed as the last line of the stage log (stages/_base.stage_main), or None."""
    try:
        lines = [x for x in open(log_path, encoding="utf-8", errors="ignore").read().splitlines() if x.strip()]
        return json.loads(lines[-1]) if lines and lines[-1].lstrip().startswith("{") else None
    except (OSError, ValueError):
        return None


def _save(path, rj):
    tmp = path + ".tmp"
    json.dump(rj, open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def run_recording(name, cfg, stages=STAGES, frm=None, to=None, force=False, module_prefix="stages", config_path=None,
                  datasets_path=None, final_dir=FINAL, echo=print) -> dict:
    P = result_paths(cfg, name, load_datasets(datasets_path) if datasets_path else None)
    os.makedirs(P.logs, exist_ok=True)
    rj = json.load(open(P.run_json, encoding="utf-8")) if os.path.exists(P.run_json) else dict(name=name, source=P.source, stages={})
    rj.setdefault("stages", {})
    names = [s for s, _ in stages]
    i0 = names.index(frm) if frm else 0
    i1 = names.index(to) if to else len(names) - 1
    for k, (stage, env) in enumerate(stages):
        if not i0 <= k <= i1:
            continue
        done = os.path.join(P.logs, f"{stage}.done")
        tag = f"[{k + 1}/{len(stages)}] {stage}"
        if os.path.exists(done) and not force:
            rj["stages"].setdefault(stage, dict(status="ok"))
            echo(f"  {tag}: 완료됨, 건너뜀")
            continue
        missing = [s for s, _ in stages[:k] if not os.path.exists(os.path.join(P.logs, f"{s}.done"))]
        if missing:                                                 # its inputs would be missing or stale
            err = f"earlier stage(s) not done: {', '.join(missing)} (run them first, e.g. --from {missing[0]})"
            rj["stages"][stage] = dict(status="failed", error=err, start=dt.datetime.now().isoformat(timespec="seconds"))
            rj["failed_stage"] = stage
            _save(P.run_json, rj)
            echo(f"  {tag}: 실패 — {err}")
            return rj
        for later, _ in stages[k:]:
            p = os.path.join(P.logs, f"{later}.done")
            if os.path.exists(p):
                os.remove(p)
            if later != stage and later in rj["stages"]:
                rj["stages"][later]["status"] = "pending"           # built from the old input: must run again
        py = cfg["envs"][env]
        cmd = [py, "-m", f"{module_prefix}.{stage}", "--name", name]
        if config_path:
            cmd += ["--config", config_path]
        if datasets_path:
            cmd += ["--datasets", datasets_path]
        st = dict(status="running", start=dt.datetime.now().isoformat(timespec="seconds"), python=py, cmd=" ".join(cmd))
        rj["stages"][stage] = st
        _save(P.run_json, rj)
        echo(f"  {tag}: 실행 ({env})")
        t0 = time.time()
        try:
            with open(os.path.join(P.logs, f"{stage}.log"), "w", encoding="utf-8") as log:
                rc = subprocess.run(cmd, cwd=final_dir, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1")).returncode
        except KeyboardInterrupt:
            st.update(status="interrupted", end=dt.datetime.now().isoformat(timespec="seconds"), seconds=round(time.time() - t0, 1))
            _save(P.run_json, rj)
            raise
        st.update(end=dt.datetime.now().isoformat(timespec="seconds"), seconds=round(time.time() - t0, 1), returncode=rc,
                  status="ok" if rc == 0 else "failed")
        _save(P.run_json, rj)
        if rc != 0:
            echo(f"  {tag}: 실패 (logs/{stage}.log)")
            rj["failed_stage"] = stage
            _save(P.run_json, rj)
            return rj
        summary = _last_json(os.path.join(P.logs, f"{stage}.log"))
        if summary is not None:
            st["summary"] = summary
            bad = {k: v for k, v in summary.get("frame_status", {}).items() if str(k).startswith("error")}
            if bad:                                                 # frames whose processing raised: their persons are missing
                st["warning"] = f"{sum(bad.values())} frames failed: " + "; ".join(f"{k} x{v}" for k, v in bad.items())[:300]
            _save(P.run_json, rj)
        open(done, "w").write(st["end"] + "\n")
        echo(f"  {tag}: 완료 ({st['seconds']:.0f} s)" + (f" — 경고: {st['warning']}" if st.get("warning") else ""))
    rj.pop("failed_stage", None)
    _save(P.run_json, rj)
    return rj
