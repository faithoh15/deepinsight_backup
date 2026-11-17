"""Task 3: runner (order, done markers, skip, force, failure, interruption, invalidation, CLI) with fake stages."""
import json
import os
import subprocess
import sys
import tempfile

import yaml

FINAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FINAL)
from pipeline.common.config import load_config  # noqa: E402
from pipeline.runner import run_recording  # noqa: E402

SCRATCH = os.path.join(tempfile.gettempdir(), "final_tests")
FAKE = [("a", "main"), ("b", "main"), ("c", "main")]
NAMES = ("rec1_2026-01-01-000001", "rec2_2026-01-01-000002")


def setup():
    os.makedirs(SCRATCH, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=SCRATCH)
    cfg = load_config()
    cfg["paths"] = dict(nas_root=os.path.join(tmp, "nas"), cache_root=os.path.join(tmp, "cache"), results_root=os.path.join(tmp, "out"))
    cfg["envs"] = dict(main=sys.executable, yolo=sys.executable)
    cp, dp = os.path.join(tmp, "pipeline.yaml"), os.path.join(tmp, "datasets.csv")
    yaml.safe_dump(cfg, open(cp, "w"), allow_unicode=True)
    open(dp, "w").write("name,source\n" + "".join(f"{n},{n[-17:]}\n" for n in NAMES))
    os.makedirs(os.path.join(tmp, "nas", NAMES[1][-17:]))          # only the second recording exists on the fake NAS
    return cfg, cp, dp, tmp


def trace(cfg, name):
    p = os.path.join(cfg["paths"]["results_root"], name, "trace.txt")
    return open(p).read().split() if os.path.exists(p) else []


def kw(cp, dp):
    return dict(module_prefix="tests.fake_stages", config_path=cp, datasets_path=dp)


def test_stages_run_in_order_and_mark_done():
    cfg, cp, dp, _ = setup()
    r = run_recording(NAMES[0], cfg, FAKE, **kw(cp, dp))
    assert trace(cfg, NAMES[0]) == ["a", "b", "c"] and [r["stages"][s]["status"] for s in "abc"] == ["ok"] * 3
    logs = os.path.join(cfg["paths"]["results_root"], NAMES[0], "logs")
    assert all(os.path.exists(os.path.join(logs, f"{s}.done")) and os.path.exists(os.path.join(logs, f"{s}.log")) for s in "abc")
    rj = json.load(open(os.path.join(cfg["paths"]["results_root"], NAMES[0], "run.json")))
    assert rj["stages"]["b"]["seconds"] >= 0 and rj["stages"]["b"]["python"] == sys.executable


def test_done_stage_is_skipped_unless_force():
    cfg, cp, dp, _ = setup()
    run_recording(NAMES[0], cfg, FAKE, **kw(cp, dp))
    run_recording(NAMES[0], cfg, FAKE, **kw(cp, dp))
    assert trace(cfg, NAMES[0]) == ["a", "b", "c"]
    run_recording(NAMES[0], cfg, FAKE, frm="b", to="b", force=True, **kw(cp, dp))
    assert trace(cfg, NAMES[0]) == ["a", "b", "c", "b"]


def test_rerun_of_a_stage_invalidates_later_stages():
    cfg, cp, dp, _ = setup()
    run_recording(NAMES[0], cfg, FAKE, **kw(cp, dp))
    run_recording(NAMES[0], cfg, FAKE, frm="b", to="b", force=True, **kw(cp, dp))
    assert not os.path.exists(os.path.join(cfg["paths"]["results_root"], NAMES[0], "logs", "c.done"))
    rj = json.load(open(os.path.join(cfg["paths"]["results_root"], NAMES[0], "run.json")))
    assert rj["stages"]["c"]["status"] == "pending" and rj["stages"]["b"]["status"] == "ok"      # run.json agrees with the .done files
    run_recording(NAMES[0], cfg, FAKE, **kw(cp, dp))
    assert trace(cfg, NAMES[0]) == ["a", "b", "c", "b", "c"]


def test_failed_stage_stops_that_recording_and_leaves_no_done():
    cfg, cp, dp, _ = setup()
    r = run_recording(NAMES[0], cfg, [("a", "main"), ("boom", "main"), ("c", "main")], **kw(cp, dp))
    assert r["stages"]["boom"]["status"] == "failed" and "c" not in r["stages"] and trace(cfg, NAMES[0]) == ["a"]
    logs = os.path.join(cfg["paths"]["results_root"], NAMES[0], "logs")
    assert not os.path.exists(os.path.join(logs, "boom.done")) and "boom" in open(os.path.join(logs, "boom.log")).read()


def test_interrupted_stage_is_rerun():
    cfg, cp, dp, _ = setup()
    run_recording(NAMES[0], cfg, FAKE, **kw(cp, dp))
    os.remove(os.path.join(cfg["paths"]["results_root"], NAMES[0], "logs", "c.done"))     # as if c was killed before finishing
    run_recording(NAMES[0], cfg, FAKE, **kw(cp, dp))
    assert trace(cfg, NAMES[0]) == ["a", "b", "c", "c"]


def test_cli_continues_with_next_recording_after_a_failure():
    cfg, cp, dp, _ = setup()
    p = subprocess.run([sys.executable, "run.py", *NAMES, "--config", cp, "--datasets", dp, "--stage-module", "tests.fake_stages",
                        "--stage-list", "needs_src:main,a:main"], cwd=FINAL, capture_output=True, text=True)
    assert p.returncode == 1, p.stdout + p.stderr
    assert trace(cfg, NAMES[0]) == [] and trace(cfg, NAMES[1]) == ["a"]
    assert "실패" in p.stdout and NAMES[0] in p.stdout


def test_cli_rejects_an_unknown_name_before_running():
    cfg, cp, dp, _ = setup()
    p = subprocess.run([sys.executable, "run.py", "rec1_2026-01-01-00000", "--config", cp, "--datasets", dp, "--stage-module",
                        "tests.fake_stages", "--stage-list", "a:main"], cwd=FINAL, capture_output=True, text=True)
    assert p.returncode == 2 and NAMES[0] in (p.stdout + p.stderr)


def test_stage_refuses_to_run_when_an_earlier_stage_is_not_done():
    cfg, cp, dp, _ = setup()
    r = run_recording(NAMES[0], cfg, FAKE, frm="b", **kw(cp, dp))
    assert r["stages"]["b"]["status"] == "failed" and "a" in r["stages"]["b"]["error"] and trace(cfg, NAMES[0]) == []
    assert r["failed_stage"] == "b"


def test_error_frames_reported_as_warning():
    cfg, cp, dp, _ = setup()
    p = subprocess.run([sys.executable, "run.py", NAMES[0], "--config", cp, "--datasets", dp, "--stage-module", "tests.fake_stages",
                        "--stage-list", "a:main,errframes:main"], cwd=FINAL, capture_output=True, text=True)
    rj = json.load(open(os.path.join(cfg["paths"]["results_root"], NAMES[0], "run.json")))
    assert rj["stages"]["errframes"]["status"] == "ok" and "2" in rj["stages"]["errframes"]["warning"]
    assert p.returncode == 0 and "경고" in p.stdout and NAMES[0] in p.stdout


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _runner import run
    run(globals())
