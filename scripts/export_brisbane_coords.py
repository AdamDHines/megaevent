"""Export per-slice calibrated GPS coords for the gept in-loop pooled Brisbane eval.

Writes ``<out>`` (default: ``~/repo/gept/data/brisbane_pooled_coords.npz``) with, per
traverse, ``<cond>_xy`` float32 ``[n_slices, 2]`` (metres, one shared local projection)
and ``<cond>_valid`` bool ``[n_slices]`` (False outside the GPS span). Indexed by
**eventcv's dt=50 ms slice index of the raw HDF5** — the same slicing gept's
``EventH5Dataset`` uses, so the trainer consumes these with ``start=0`` and asserts the
lengths match its readers (any eventcv slicing drift becomes a hard error, not a silent
GT shift). Clock calibration comes from the same ``clock_offsets`` the benchmark uses.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import brisbane_pooled as bp  # noqa: E402
from brisbane_resolution import _Args  # noqa: E402

SEQUENCES = ("sunset1", "sunset2", "daytime", "morning", "night", "sunrise")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eventlab-dir", default="/media/adam/vprdatasets/eventgem")
    ap.add_argument("--out", default=os.path.expanduser(
        "~/repo/gept/data/brisbane_pooled_coords.npz"))
    ap.add_argument("--dt-ms", type=int, default=50)
    cli = ap.parse_args()

    args = _Args(cli.eventlab_dir, "brisbane_event", cli.dt_ms, False, False)
    args.filter_dt_us = 50_000
    geom = bp.traverse_geometry(args, list(SEQUENCES), None)

    out = {"dt_ms": np.int32(cli.dt_ms)}
    for s in SEQUENCES:
        xy, valid = geom[s][0], geom[s][1]
        out[f"{s}_xy"] = np.asarray(xy, dtype=np.float32)
        out[f"{s}_valid"] = np.asarray(valid, dtype=bool)
        print(f"  {s}: {len(xy)} slices, {int(valid.sum())} valid")
    os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    np.savez_compressed(cli.out, **out)
    print(f"-> {cli.out} ({os.path.getsize(cli.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
