"""Stage clock: image clock offset of the recording -> 0_prep/clock.yaml (pipeline.prep.clock)."""
from pipeline.prep.clock import compute_clock, write_clock
from stages._base import stage_main


def run(P, cfg, **kw):
    info = compute_clock(P.data)
    write_clock(P.clock, info)
    return info


if __name__ == "__main__":
    stage_main(run)
