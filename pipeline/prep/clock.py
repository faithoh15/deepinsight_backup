"""Image clock offset of a recording. The image stamps are late against LiDAR/IMU by about 9.6-9.7 s; the rule used for every
recording so far (proposed_pipeline/tweaks/*.yaml): offset = -(first cam0 image stamp - first IMU stamp), rounded to ms."""
from __future__ import annotations

import glob
import os

import pandas as pd
import yaml


def image_offset_from_stamps(first_cam0_us: int, first_imu_us: int) -> int:
    return -int(round((int(first_cam0_us) - int(first_imu_us)) / 1000.0)) * 1000


def compute_clock(data_dir: str) -> dict:
    stamps = [int(os.path.basename(f).split(".")[0]) for f in glob.glob(os.path.join(data_dir, "images_cam0", "*.jpg"))]
    if not stamps:
        raise FileNotFoundError(f"no cam0 images in {data_dir}/images_cam0")
    imu0 = int(pd.read_csv(os.path.join(data_dir, "imu_data.csv"), usecols=["Stamp"], nrows=1)["Stamp"].iloc[0])
    cam0 = min(stamps)
    return dict(image_offset_us=image_offset_from_stamps(cam0, imu0), first_cam0_us=cam0, first_imu_us=imu0,
                first_cam0_minus_imu_s=round((cam0 - imu0) / 1e6, 6))


def write_clock(path: str, info: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# image clock offset: -(first cam0 stamp - first IMU stamp), rounded to ms (same rule as proposed_pipeline/tweaks)\n")
        yaml.safe_dump({k: (int(v) if isinstance(v, int) else v) for k, v in info.items()}, f, sort_keys=False)


def read_clock(path: str) -> int:
    return int(yaml.safe_load(open(path, encoding="utf-8"))["image_offset_us"])
