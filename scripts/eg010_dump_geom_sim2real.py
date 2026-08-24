"""APS-aligned geometry workpack for the 0.1.0 sim2real rerank driver.

Same job as ``eg010_dump_geom.py`` — the 0.1.0 env has no eventcv, so coordinates
and coverage are exported from here — but for the Brisbane sim2real protocol:
sunset1 -> daytime+morning+sunrise with ``aps_align=True``, the exact geometry the
megaevent v8 sim2real cells were scored on. The coverage counts are asserted
against that run's provenance (sim2real/megaevent_v8_real.json rows), so this pack
cannot silently describe different rows than the table it will sit beside.

    pixi run python3 scripts/eg010_dump_geom_sim2real.py

Writes ``v8_bench/eg010_geom_brisbane_sim2real.npz`` with ``<traverse>_xy``
(float64 [n, 2], raw frame order) and ``<traverse>_cov`` (bool [n], GPS-covered
AND APS-aligned) for the four traverses.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from brisbane_pooled import traverse_geometry  # noqa: E402
from brisbane_resolution import _Args  # noqa: E402

OUT = "/media/adam/vprdatasets/megaevent/v8_bench"
NPZ_ROOT = "/media/adam/vprdatasets/megaevent/brisbane_npz"
EVENTLAB = "/media/adam/vprdatasets/eventgem"
SEQS = ["sunset1", "daytime", "morning", "sunrise"]
# The megaevent v8/v5 sim2real runs' aligned row counts (json provenance).
EXPECTED = {"sunset1": 14228, "daytime": 14127, "morning": 13286, "sunrise": 13526}


def main():
    args = _Args(EVENTLAB, "brisbane_event", 50, False, False)
    args.filter_dt_us = 50_000
    geom = traverse_geometry(args, SEQS, NPZ_ROOT, aps_align=True)
    payload = {}
    for s in SEQS:
        xy, covered = geom[s][0], geom[s][1]
        n = int(covered.sum())
        if n != EXPECTED[s]:
            raise SystemExit(f"{s}: {n} aligned rows != {EXPECTED[s]} from the "
                             f"megaevent sim2real runs — geometry drifted")
        payload[f"{s}_xy"] = np.asarray(xy, dtype=np.float64)
        payload[f"{s}_cov"] = np.asarray(covered, dtype=bool)
    path = os.path.join(OUT, "eg010_geom_brisbane_sim2real.npz")
    np.savez_compressed(path, **payload)
    print(f"-> {path} " + " ".join(
        f"{s}:{int(payload[f'{s}_cov'].sum())}/{len(payload[f'{s}_cov'])}"
        for s in SEQS))


if __name__ == "__main__":
    main()
