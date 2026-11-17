"""configs/local.yaml overlay (this computer's paths) and path resolution of paths / envs / models in load_config."""
import os
import sys
import tempfile

import yaml

FINAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FINAL)
from pipeline.common.config import conda_python, load_config  # noqa: E402

TMP = tempfile.mkdtemp(prefix="final_test_")


def _write(name, obj):
    p = os.path.join(TMP, name)
    yaml.safe_dump(obj, open(p, "w", encoding="utf-8"), allow_unicode=True)
    return p


def test_relative_paths_resolve_against_repo_root():
    cfg = load_config(local=os.path.join(TMP, "none.yaml"))           # no local file
    assert cfg["paths"]["results_root"] == os.path.join(FINAL, "output")
    assert cfg["models"]["yolo"] == os.path.join(FINAL, "models", "yolo26l.pt")
    assert cfg["models"]["operator_seg"] == os.path.join(FINAL, "models", "yolo26l-seg.pt")
    assert all(os.path.isabs(v) for v in cfg["paths"].values())
    assert cfg["detection"]["candidates"]["scans_before"] == 20       # stage settings untouched


def test_local_overrides_only_given_keys():
    local = _write("local.yaml", {"paths": {"nas_root": "/data/rec"}, "envs": {"yolo": "/opt/y/bin/python"}})
    base = load_config(local=os.path.join(TMP, "none.yaml"))
    cfg = load_config(local=local)
    assert cfg["paths"]["nas_root"] == "/data/rec"
    assert cfg["paths"]["results_root"] == base["paths"]["results_root"]
    assert cfg["envs"]["yolo"] == "/opt/y/bin/python" and cfg["envs"]["main"] == base["envs"]["main"]
    assert cfg["tracking"] == base["tracking"]


def test_home_and_variables_expand():
    os.environ["FINAL_TEST_DATA"] = "/mnt/x"
    local = _write("local2.yaml", {"paths": {"nas_root": "~/rec", "cache_root": "$FINAL_TEST_DATA/cache"}})
    cfg = load_config(local=local)
    assert cfg["paths"]["nas_root"] == os.path.join(os.path.expanduser("~"), "rec")
    assert cfg["paths"]["cache_root"] == "/mnt/x/cache"


def test_conda_env_names():
    base = os.path.join(TMP, "miniforge3")
    os.makedirs(os.path.join(base, "bin"))
    assert conda_python("deepinsight", {"CONDA_EXE": os.path.join(base, "bin", "conda")}, home=TMP) == \
        os.path.join(base, "envs", "deepinsight", "bin", "python")
    assert conda_python("yolo_env", {}, home=TMP) == os.path.join(base, "envs", "yolo_env", "bin", "python")   # found under home
    empty = tempfile.mkdtemp(prefix="final_home_")
    assert conda_python("x", {}, home=empty) == os.path.join(empty, "miniconda3", "envs", "x", "bin", "python")  # guess, named in errors
    other = os.path.join(TMP, "anaconda3", "envs", "only_here", "bin")                  # env lives in a second conda installation
    os.makedirs(other)
    open(os.path.join(other, "python"), "w").write("")
    assert conda_python("only_here", {"CONDA_EXE": os.path.join(base, "bin", "conda")}, home=TMP) == os.path.join(other, "python")
    user_envs = os.path.join(TMP, ".conda", "envs", "user_env", "bin")                  # conda's per-user env folder
    os.makedirs(user_envs)
    open(os.path.join(user_envs, "python"), "w").write("")
    assert conda_python("user_env", {}, home=TMP) == os.path.join(user_envs, "python")
    local = _write("local3.yaml", {"envs": {"main": "conda:abc", "yolo": "/abs/python"}})
    cfg = load_config(local=local)
    assert cfg["envs"]["main"].endswith(os.path.join("envs", "abc", "bin", "python")) and cfg["envs"]["yolo"] == "/abs/python"


def test_explicit_config_path_skips_local_overlay():
    p = _write("cfg.yaml", {"paths": {"nas_root": "a", "cache_root": "/c", "results_root": "/r"}, "envs": {"main": "/p", "yolo": "/q"},
                            "models": {"yolo": "/m1", "operator_seg": "/m2"}})
    local = _write("local4.yaml", {"paths": {"results_root": "/should_not_apply"}})
    cfg = load_config(p, local=local)
    assert cfg["paths"]["results_root"] == "/r" and cfg["paths"]["nas_root"] == os.path.join(FINAL, "a")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _runner import run
    run(globals())
