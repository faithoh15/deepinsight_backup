"""The code is portable: no machine-specific absolute path in code or configs/pipeline.yaml, nothing imported from outside the
repository except installed packages; the result comparison tool reports differences."""
import ast
import glob
import importlib
import os
import sys
import tempfile

FINAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FINAL)
SCRATCH = os.path.join(tempfile.gettempdir(), "final_tests")
FORBIDDEN = ("/home/", "/run/user/", "/mnt/", "/media/")          # machine-specific absolute paths (provenance docstrings allowed)


def code_files():
    fs = glob.glob(os.path.join(FINAL, "pipeline", "**", "*.py"), recursive=True) + glob.glob(os.path.join(FINAL, "stages", "*.py"))
    fs += glob.glob(os.path.join(FINAL, "tools", "*.py")) + [p for p in (os.path.join(FINAL, "run.py"), os.path.join(FINAL, "view.py")) if os.path.exists(p)]
    return fs


def test_no_import_outside_final():
    for f in code_files():
        src = open(f, encoding="utf-8").read()
        tree = ast.parse(src)
        docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
                and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}   # provenance notes allowed
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "insert" and "sys.path" in ast.unparse(node.func):
                assert ast.unparse(node.args[1]) == "FINAL", f"{f}: sys.path.insert of {ast.unparse(node.args[1])}"   # only final's root
            if isinstance(node, ast.ImportFrom) and node.module and (node.module.split(".")[0] in ("src", "legacy", "common", "fusion_common")):
                assert False, f"{f}: import {node.module}"
            if isinstance(node, ast.Constant) and id(node) not in docs and isinstance(node.value, str) and any(p in node.value for p in FORBIDDEN):
                assert False, f"{f}: path {node.value[:80]}"
    before = set(sys.modules)
    for f in glob.glob(os.path.join(FINAL, "pipeline", "**", "*.py"), recursive=True):
        mod = os.path.relpath(f, FINAL)[:-3].replace(os.sep, ".").replace(".__init__", "")
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as e:                 # YOLO-only modules need torch / ultralytics (yolo_env); run this file there too
            assert mod in ("pipeline.prep.yolo", "pipeline.prep.operator") and e.name in ("torch", "ultralytics", "torchvision"), (mod, e)
    installed = tuple({os.path.realpath(x) for x in (sys.prefix, sys.base_prefix, sys.exec_prefix)})
    for name in set(sys.modules) - before:
        p = os.path.realpath(getattr(sys.modules[name], "__file__", None) or "")
        assert not p or p.startswith(installed) or p.startswith(os.path.realpath(FINAL)), (name, p)


def test_config_has_no_absolute_paths():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(FINAL, "configs", "pipeline.yaml"), encoding="utf-8"))
    for sec in ("paths", "envs", "models"):
        for k, v in cfg[sec].items():
            assert not os.path.isabs(str(v)) and not str(v).startswith("~"), (sec, k, v)


def test_compare_reports_column_differences():
    from tools.compare_results import compare_dirs
    a, b = tempfile.mkdtemp(dir=SCRATCH), tempfile.mkdtemp(dir=SCRATCH)
    open(os.path.join(a, "t.csv"), "w").write("frame,x,seconds\n1,0.5,3.1\n2,0.7,2.0\n")
    open(os.path.join(b, "t.csv"), "w").write("frame,x,seconds\n1,0.5,3.1\n2,0.7,2.0\n")
    assert compare_dirs(a, b) == []
    open(os.path.join(b, "t.csv"), "w").write("frame,x,seconds\n1,0.5,9.9\n2,0.7,2.5\n")
    assert compare_dirs(a, b) == []                                   # processing time column ignored
    open(os.path.join(b, "t.csv"), "w").write("frame,x,seconds\n1,0.5,9.9\n2,0.8,2.5\n")
    d = compare_dirs(a, b)
    assert len(d) == 1 and "t.csv" in d[0] and "x" in d[0]
    os.remove(os.path.join(b, "t.csv"))
    assert any("missing" in x for x in compare_dirs(a, b))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _runner import run
    run(globals())
