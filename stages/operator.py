"""Stage operator (yolo_env): operator pixel masks + operator columns of the YOLO boxes -> 0_prep/operator_mask/, 0_prep/yolo/."""
from stages._base import stage_main


def run(P, cfg, **kw):
    from pipeline.prep.operator import run_operator
    return run_operator(P.sidebyside, P.yolo, P.operator_mask, cfg["models"]["operator_seg"], cfg["prep"]["operator"], figure=cfg["debug_figures"])


if __name__ == "__main__":
    stage_main(run)
