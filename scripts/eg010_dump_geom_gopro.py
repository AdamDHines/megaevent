"""Union-masked geometry workpack for the 0.1.0 gopro pairwise rerank driver.

Same job as ``eg010_dump_geom_sim2real.py`` — the 0.1.0 env has no eventcv, so
coordinates and coverage are exported from here — but for the gopro pairwise cell:
sunset1 (reference) -> morning (query), rows = GPS-covered AND NOT out_of_tolerance
in EITHER the ``aps`` or the ``gopro`` frame tree. The ENU origin comes from the
full five-traverse list, byte-identical to ``pairwise_sunset_ref.py``'s geometry,
so the ledger gate in ``scripts/gopro_pairwise.py`` sees the exact same frame.

    pixi run python3 scripts/eg010_dump_geom_gopro.py

Writes ``v8_bench/eg010_geom_brisbane_gopro.npz`` with ``<traverse>_xy``
(float64 [n, 2], raw frame order) and ``<traverse>_cov`` (bool [n]) for sunset1
and morning.

No hardcoded row counts: the masked count is re-derived here from the raw
``select.json`` files and cross-checked against ``frame_validity_mask``, and the
gopro tree's recorded time-warp fingerprint must equal the current
``logs/gopro2/warp_<seq>.json`` — a pack can never describe a stale tree.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from brisbane_pooled import traverse_geometry  # noqa: E402
from brisbane_resolution import _Args  # noqa: E402
from src import traversenpz as tnpz  # noqa: E402

OUT = "/media/adam/vprdatasets/megaevent/v8_bench"
NPZ_ROOT = "/media/adam/vprdatasets/megaevent/brisbane_npz"
EVENTLAB = "/media/adam/vprdatasets/eventgem"
DATASET = "brisbane_event"
# Full list for the origin (matches pairwise_sunset_ref.CONFIGS geom_traverses)...
GEOM_SEQS = ["daytime", "morning", "night", "sunrise", "sunset1"]
# ...but only the pair's traverses are masked and exported.
EXPORT_SEQS = ["sunset1", "morning"]
MASK_SOURCES = ("aps", "gopro")


def select_report(source, seq):
    path = os.path.join(NPZ_ROOT, DATASET, source, seq, "select.json")
    with open(path) as handle:
        return path, json.load(handle)


def main():
    args = _Args(EVENTLAB, DATASET, 50, False, False)
    args.filter_dt_us = 50_000
    geom = traverse_geometry(args, GEOM_SEQS, NPZ_ROOT)
    payload = {}
    for s in EXPORT_SEQS:
        xy, covered = geom[s][0], geom[s][1]
        n = len(covered)
        keep = tnpz.frame_validity_mask(NPZ_ROOT, DATASET, s, n, MASK_SOURCES)
        cov = covered & keep

        # Re-derive the same mask from the raw JSONs; a disagreement means the mask
        # function and the extractors' contract have drifted apart.
        bad = set()
        for source in MASK_SOURCES:
            path, report = select_report(source, s)
            if int(report["n_slices"]) != n:
                raise SystemExit(f"{path}: n_slices {report['n_slices']} != grid {n}")
            bad.update(int(i) for i in report["out_of_tolerance"])
        raw = covered & ~np.isin(np.arange(n), sorted(bad))
        if not np.array_equal(cov, raw):
            raise SystemExit(f"{s}: frame_validity_mask disagrees with the raw "
                             f"select.json union ({int(cov.sum())} vs {int(raw.sum())})")

        _, gopro = select_report("gopro", s)
        warp_path = os.path.join(os.path.dirname(HERE), "logs", "gopro2",
                                 f"warp_{s}.json")
        with open(warp_path) as handle:
            warp = json.load(handle)
        tw = gopro.get("timewarp")
        if not tw or tw.get("fingerprint") != warp["fingerprint"]:
            raise SystemExit(
                f"{s}: gopro tree framed by warp {tw and tw.get('fingerprint')} but "
                f"{warp_path} is {warp['fingerprint']} — the tree is stale, re-run "
                f"scripts/extract_gopro.py --seq {s} --timewarp {warp_path}")

        payload[f"{s}_xy"] = np.asarray(xy, dtype=np.float64)
        payload[f"{s}_cov"] = np.asarray(cov, dtype=bool)

    path = os.path.join(OUT, "eg010_geom_brisbane_gopro.npz")
    np.savez_compressed(path, **payload)
    print(f"-> {path} " + " ".join(
        f"{s}:{int(payload[f'{s}_cov'].sum())}/{len(payload[f'{s}_cov'])}"
        for s in EXPORT_SEQS))


if __name__ == "__main__":
    main()
