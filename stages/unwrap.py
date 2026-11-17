"""Stage unwrap: side-by-side equirectangular images -> 0_prep/sidebyside/ (pipeline.prep.unwrap)."""
from pipeline.prep.unwrap import run_unwrap
from stages._base import open_run, stage_main


def run(P, cfg, frames=None, **kw):
    return run_unwrap(open_run(P), P.sidebyside, cfg["prep"]["unwrap"], frames=frames)


if __name__ == "__main__":
    stage_main(run)
