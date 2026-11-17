"""Stage track2d (yolo_env): polar 2D tracking of the YOLO boxes -> 0_prep/track2d/tracks.csv (pipeline.prep.track2d)."""
import os

from stages._base import stage_main


def run(P, cfg, frames=None, **kw):
    from pipeline.prep.track2d import run_track2d
    return run_track2d(P.yolo, P.sidebyside, P.data, os.path.join(P.masks, "frame_poses.csv"), P.track2d, cfg["prep"]["track2d"],
                       frames=frames, figure=cfg["debug_figures"])


if __name__ == "__main__":
    stage_main(run)
