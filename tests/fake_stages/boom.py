"""Fake stage that fails."""
from stages._base import stage_main


def run(P, cfg, **kw):
    raise RuntimeError("boom")


if __name__ == "__main__":
    stage_main(run)
