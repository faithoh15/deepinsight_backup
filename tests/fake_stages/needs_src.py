"""Fake fetch: fails when the recording's NAS folder does not exist."""
import os

from stages._base import stage_main


def run(P, cfg, **kw):
    if not os.path.isdir(P.nas):
        raise FileNotFoundError(P.nas)
    return {}


if __name__ == "__main__":
    stage_main(run)
