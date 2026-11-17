"""Stage odometry: gyro rotation + LiDAR translation -> 0_prep/odometry/odometry.csv (pipeline.prep.odometry)."""
from pipeline.prep.odometry import run_odometry
from stages._base import open_run, stage_main


def run(P, cfg, scans=None, **kw):
    return run_odometry(open_run(P), P.odometry, cfg["prep"]["odometry"], scans=scans, figure=cfg["debug_figures"])


if __name__ == "__main__":
    stage_main(run)
