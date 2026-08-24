"""Geometry workpack for the 0.1.0 pooled+rerank driver.

The released keypoint rerank must run inside the 0.1.0 pixi env (its validated
cv2 4.13 USAC_FAST path — megaevent carries cv2 5.0), but that env has no
eventcv, so the pooled coordinates and coverage masks are exported here through
the exact same ``eg010_pooled.geometry`` call the pooled global scoring used.

    pixi run python3 scripts/eg010_dump_geom.py

Writes ``v8_bench/eg010_geom_<dataset>.npz`` with ``<traverse>_xy`` (float64
[n, 2], raw frame order) and ``<traverse>_cov`` (bool [n]) per traverse.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import eg010_pooled as ep  # noqa: E402

OUT = "/media/adam/vprdatasets/megaevent/v8_bench"


def main():
    for ds, (eventlab_dir, query, database) in ep.CONFIGS.items():
        seqs = [query, *database]
        geom = ep.geometry(ds, eventlab_dir, seqs)
        payload = {}
        for s in seqs:
            payload[f"{s}_xy"] = np.asarray(geom[s][0], dtype=np.float64)
            payload[f"{s}_cov"] = np.asarray(geom[s][1], dtype=bool)
        path = os.path.join(OUT, f"eg010_geom_{ds}.npz")
        np.savez_compressed(path, **payload)
        print(f"-> {path} " + " ".join(
            f"{s}:{int(payload[f'{s}_cov'].sum())}/{len(payload[f'{s}_cov'])}"
            for s in seqs))


if __name__ == "__main__":
    main()
