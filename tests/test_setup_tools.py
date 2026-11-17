"""tools/download_models.py (download + sha256 check) and tools/check_setup.py (environment, model and recording checks)."""
import hashlib
import os
import sys
import tempfile

FINAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FINAL)
from tools.check_setup import check_env, check_model, check_recording  # noqa: E402
from tools.download_models import MODELS, fetch_model  # noqa: E402

TMP = tempfile.mkdtemp(prefix="final_setup_")


def _src(data=b"weights-bytes"):
    p = os.path.join(tempfile.mkdtemp(dir=TMP), "w.pt")
    open(p, "wb").write(data)
    return "file://" + p, hashlib.sha256(data).hexdigest()


def test_known_models_are_the_official_weights():
    assert set(MODELS) == {"yolo26l.pt", "yolo26l-seg.pt"}
    assert MODELS["yolo26l.pt"]["sha256"].startswith("9fe3c544f2b1") and MODELS["yolo26l-seg.pt"]["sha256"].startswith("636024306410")
    assert all(m["url"].startswith("https://github.com/ultralytics/assets/releases/download/") for m in MODELS.values())


def test_fetch_model_downloads_and_checks_hash():
    url, sha = _src()
    dest = os.path.join(TMP, "m", "a.pt")
    assert fetch_model(url, dest, sha) == "downloaded"
    assert open(dest, "rb").read() == b"weights-bytes"
    assert fetch_model(url, dest, sha) == "present"                   # second call: hash matches, no download


def test_fetch_model_rejects_a_wrong_hash_and_leaves_nothing():
    url, _ = _src()
    dest = os.path.join(TMP, "m", "b.pt")
    try:
        fetch_model(url, dest, "0" * 64)
        assert False, "no error"
    except ValueError as e:
        assert "sha256" in str(e)
    assert not os.path.exists(dest) and not os.path.exists(dest + ".part")


def test_check_model():
    url, sha = _src(b"abc")
    dest = os.path.join(TMP, "m", "c.pt")
    assert check_model(dest, sha)[0] == "error"                       # missing
    fetch_model(url, dest, sha)
    assert check_model(dest, sha)[0] == "ok"
    assert check_model(dest, "1" * 64)[0] == "warn"                   # other weights: runs, results may differ


def test_check_env():
    assert check_env(sys.executable, ["os", "json"])[0] == "ok"
    lvl, msg = check_env(sys.executable, ["os", "no_such_module_xyz"])
    assert lvl == "error" and "no_such_module_xyz" in msg
    lvl, msg = check_env("/nonexistent/bin/python", ["os"])
    assert lvl == "error" and "/nonexistent/bin/python" in msg


def test_check_recording():
    root = tempfile.mkdtemp(dir=TMP)
    missing = check_recording(root)
    assert set(missing) == {"calib/config/camera.yaml", "calib/config/imu_camera0.yaml", "imu_data.csv", "lidar/", "images_cam0/", "images_cam1/"}
    os.makedirs(os.path.join(root, "calib", "config"))
    for f in ("calib/config/camera.yaml", "calib/config/imu_camera0.yaml", "imu_data.csv"):
        open(os.path.join(root, f), "w").write("x")
    for d, f in (("lidar", "1.ply"), ("images_cam0", "1.jpg"), ("images_cam1", "1.jpg")):
        os.makedirs(os.path.join(root, d))
        open(os.path.join(root, d, f), "w").write("x")
    assert check_recording(root) == []
    os.remove(os.path.join(root, "lidar", "1.ply"))
    assert check_recording(root) == ["lidar/"]                        # empty folder counts as missing


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _runner import run
    run(globals())
