"""Stage masks: mount / operator masks + device pose at every camera frame -> 0_prep/masks/ (pipeline.prep.masks)."""
import os

from pipeline.prep.masks import run_masks
from stages._base import open_run, stage_main


def run(P, cfg, **kw):
    return run_masks(open_run(P), os.path.join(P.odometry, "odometry.csv"), P.masks, cfg["prep"]["masks"])


if __name__ == "__main__":
    stage_main(run)
