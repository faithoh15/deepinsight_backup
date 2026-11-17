"""Fake stage for the runner tests: appends its name to <root>/trace.txt."""
import os

from stages._base import stage_main


def run(P, cfg, **kw):
    os.makedirs(P.root, exist_ok=True)
    with open(os.path.join(P.root, "trace.txt"), "a") as f:
        f.write("c\n")
    return dict(stage="c")


if __name__ == "__main__":
    stage_main(run)
