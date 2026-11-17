"""Stage yolo (yolo_env): YOLO person boxes on the side-by-side images -> 0_prep/yolo/ (pipeline.prep.yolo)."""
from pipeline.prep.clock import read_clock
from stages._base import stage_main


def run(P, cfg, frames=None, **kw):
    from pipeline.prep.yolo import run_yolo
    return run_yolo(P.sidebyside, P.yolo, cfg["models"]["yolo"], cfg["prep"]["yolo"], read_clock(P.clock), frames=frames,
                    figure=cfg["debug_figures"])


if __name__ == "__main__":
    stage_main(run)
