"""Task 3: NAS -> local cache copy (complete copy reused, missing / truncated files filled, source never modified, missing source named)."""
import os
import sys
import tempfile

FINAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FINAL)
from pipeline.prep.fetch import fetch_recording  # noqa: E402

SCRATCH = os.path.join(tempfile.gettempdir(), "final_tests")


def fake_source():
    os.makedirs(SCRATCH, exist_ok=True)
    root = tempfile.mkdtemp(dir=SCRATCH)
    src = os.path.join(root, "nas", "2026-01-01-000001")
    for d, n in (("lidar", 3), ("images_cam0", 2), ("images_cam1", 2), ("calib/sub", 1)):
        os.makedirs(os.path.join(src, d))
        for k in range(n):
            open(os.path.join(src, d, f"{k}.bin"), "wb").write(os.urandom(100 + k))
    open(os.path.join(src, "imu_data.csv"), "w").write("Stamp,GyroX\n1,0\n")
    return src, os.path.join(root, "cache", "2026-01-01-000001")


def snapshot(d):
    return {os.path.relpath(os.path.join(r, f), d): (os.path.getsize(os.path.join(r, f)), os.path.getmtime(os.path.join(r, f)))
            for r, _, fs in os.walk(d) for f in fs}


def test_fetch_copies_once_then_reuses():
    src, dst = fake_source()
    a = fetch_recording(src, dst)
    assert a["copied"] == 9 and a["files"] == 9
    assert {k: v[0] for k, v in snapshot(dst).items()} == {k: v[0] for k, v in snapshot(src).items()}
    b = fetch_recording(src, dst)
    assert b["copied"] == 0


def test_fetch_fills_a_partial_copy():
    src, dst = fake_source()
    fetch_recording(src, dst)
    os.remove(os.path.join(dst, "lidar", "1.bin"))
    open(os.path.join(dst, "lidar", "2.bin"), "wb").write(b"x")                      # truncated
    r = fetch_recording(src, dst)
    assert r["copied"] == 2
    assert open(os.path.join(dst, "lidar", "2.bin"), "rb").read() == open(os.path.join(src, "lidar", "2.bin"), "rb").read()
    assert not any(f.endswith(".part") for r_, _, fs in os.walk(dst) for f in fs)


def test_missing_source_names_the_path():
    src, dst = fake_source()
    try:
        fetch_recording(src + "_nope", dst)
        assert False, "no error"
    except FileNotFoundError as e:
        assert src + "_nope" in str(e)


def test_same_source_and_cache_copies_nothing():
    src, _ = fake_source()                                              # default setup: cache_root == nas_root (local recordings)
    before = snapshot(src)
    r = fetch_recording(src, src)
    assert r["copied"] == 0 and r["files"] == 9
    assert snapshot(src) == before


def test_source_is_never_modified():
    src, dst = fake_source()
    before = snapshot(src)
    fetch_recording(src, dst)
    os.remove(os.path.join(dst, "lidar", "0.bin"))
    fetch_recording(src, dst)
    assert snapshot(src) == before


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _runner import run
    run(globals())
