"""Fake detection-like stage whose summary reports two frames that failed."""
from stages._base import stage_main


def run(P, cfg, **kw):
    return dict(frame_status={"ok": 3, "error: ValueError: x": 2})


if __name__ == "__main__":
    stage_main(run)
