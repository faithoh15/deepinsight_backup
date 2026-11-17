"""Stage fetch: NAS recording -> local cache (pipeline.prep.fetch)."""
from pipeline.prep.fetch import fetch_recording
from stages._base import stage_main


def run(P, cfg, **kw):
    return fetch_recording(P.nas, P.data, echo=lambda s: print(s, flush=True))


if __name__ == "__main__":
    stage_main(run)
