"""Settings of the final pipeline: configs/pipeline.yaml (all stages) and configs/datasets.csv (name -> source folder).

This computer's paths go in configs/local.yaml (not in git; copy configs/local.example.yaml): it is merged over pipeline.yaml when the
default config is loaded. Values of paths / envs / models: '~' and $VARIABLES are expanded, relative paths are relative to the repository
folder, and an env written as 'conda:NAME' is the python of that conda environment."""
from __future__ import annotations

import csv
import os

import yaml

FINAL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     # repository folder
LOCAL = os.path.join(FINAL, "configs", "local.yaml")
CONDA_DIRS = ("miniconda3", "anaconda3", "miniforge3", "mambaforge")


def conda_python(name: str, environ=None, home: str | None = None) -> str:
    """python of conda environment NAME, searched in <base>/envs/NAME for the conda of $CONDA_EXE (set by `conda init`) and
    ~/miniconda3, ~/anaconda3, ~/miniforge3, ~/mambaforge, then ~/.conda/envs/NAME; the first that exists wins. If none exists the path
    under the first conda installation found (else ~/miniconda3) is returned, so the missing environment is named in the error."""
    environ = os.environ if environ is None else environ
    home = os.path.expanduser("~") if home is None else home
    exe = environ.get("CONDA_EXE")
    bases = ([os.path.dirname(os.path.dirname(exe))] if exe else []) + [os.path.join(home, d) for d in CONDA_DIRS]
    cands = [os.path.join(b, "envs", name, "bin", "python") for b in bases] + [os.path.join(home, ".conda", "envs", name, "bin", "python")]
    found = next((c for c in cands if os.path.exists(c)), None)
    if found:
        return found
    base = next((b for b in bases if os.path.isdir(b)), os.path.join(home, CONDA_DIRS[0]))
    return os.path.join(base, "envs", name, "bin", "python")


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _path(v: str) -> str:
    v = os.path.expandvars(os.path.expanduser(str(v)))
    return v if os.path.isabs(v) else os.path.normpath(os.path.join(FINAL, v))


def _resolve(cfg: dict) -> dict:
    for k, v in (cfg.get("paths") or {}).items():
        cfg["paths"][k] = _path(v)
    for k, v in (cfg.get("models") or {}).items():
        cfg["models"][k] = _path(v)
    for k, v in (cfg.get("envs") or {}).items():
        v = str(v)
        cfg["envs"][k] = conda_python(v[6:]) if v.startswith("conda:") else (_path(v) if os.sep in v or v.startswith("~") else v)
    return cfg


def load_config(path: str | None = None, local: str | None = None) -> dict:
    """path = a complete config file (then configs/local.yaml is not applied); default = configs/pipeline.yaml + configs/local.yaml."""
    cfg = yaml.safe_load(open(path or os.path.join(FINAL, "configs", "pipeline.yaml"), encoding="utf-8"))
    if path is None:
        local = LOCAL if local is None else local
        if os.path.exists(local):
            cfg = _merge(cfg, yaml.safe_load(open(local, encoding="utf-8")) or {})
    return _resolve(cfg)


def load_datasets(path: str | None = None) -> dict:
    """name -> source (the date folder of the recording), in file order."""
    with open(path or os.path.join(FINAL, "configs", "datasets.csv"), encoding="utf-8") as f:
        return {r["name"].strip(): r["source"].strip() for r in csv.DictReader(f)}
