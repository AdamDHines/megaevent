"""Every Brisbane traverse as the query, all others pooled as the reference.

    pixi run python3 scripts/brisbane_all_queries.py

The reported protocol (`scripts/brisbane_pooled.py`) run five times over: one traverse queries,
the other four are concatenated into a single gallery, nearest descriptor anywhere in it wins,
25 m Euclidean radius, native descriptor, no PCA. Scored straight off the cached banks, which
reproduces `table_native.py`'s Brisbane cells to 3 dp.

**sunset2 is excluded everywhere**, exactly as the reported row does, so every query faces the
same four-traverse gallery size. It is also the one traverse no method has a cached bank for, so
including it would mean re-rendering six traverses for ten models.

Event-GeM here is the **global** stage. Its `+rerank` needs a keypoint store per pairing and is
run separately by `scripts/eventgem_pooled.py`.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from brisbane_resolution import _Args  # noqa: E402
import brisbane_pooled as bp  # noqa: E402
from src.imagevpr import build_gt  # noqa: E402

TRAV = ["daytime", "morning", "night", "sunrise", "sunset1"]
DAY = "/media/adam/vprdatasets/megaevent/brisbane_daytime"
MODELS = [
    ("megaevent B-MLoc  real s500",  f"{DAY}/realbest", "v5b_mloc_s500",   "r322ba50"),
    ("megaevent B-SALAD real s1000", f"{DAY}/realbest", "v5b_salad_s1000", "r322ba50"),
    ("megaevent S-MLoc  real s1000", f"{DAY}/realbest", "v6s_mloc_s1000",  "r322ba50"),
    ("megaevent S-SALAD real s1000", f"{DAY}/realbest", "v6s_salad_s1000", "r322ba50"),
    ("megaevent B-MLoc  ship s3500", f"{DAY}/v5",       "pm_s3500",        "r322ba50"),
    ("megaevent B-SALAD ship s8000", f"{DAY}/ship",     "b_salad_s8000",   "r322ba50"),
    ("megaevent S-MLoc  ship s3500", f"{DAY}/ship",     "s_mloc_s3500",    "r322ba50"),
    ("megaevent S-SALAD ship s2000", f"{DAY}/ship",     "s_salad_s2000",   "r322ba50"),
    ("MegaLoc",                      f"{DAY}/megaloc",  "megaloc",         "r322ba50"),
    ("Event-GeM (global)",           f"{DAY}/eventgem", "eventgem",        "r240ba50"),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = _Args("/media/adam/vprdatasets/eventgem", "brisbane_event", 50, False, False)
    args.filter_dt_us = 50000
    geom = bp.traverse_geometry(args, TRAV, None)

    files = {}
    for name, d, label, tag in MODELS:
        files[name] = {t: {label: os.path.join(d, f"{tag}_{t}_{label}.npy")} for t in TRAV}
        for t in TRAV:
            if not os.path.exists(files[name][t][label]):
                raise SystemExit(f"{name}: missing bank {files[name][t][label]}")

    r1, r10, sizes = {}, {}, {}
    for q in TRAV:
        dbs = [t for t in TRAV if t != q]
        qxy, cov, _, _ = geom[q]
        # Geometry is model-independent, so the GT is built once per query and shared.
        _, xy, _ = bp.pool_database(files[MODELS[0][0]], geom, dbs, MODELS[0][2])
        gt = build_gt(xy, qxy[cov], 25.0)
        sizes[q] = (xy.shape[0], int(cov.sum()))
        for name, _, label, _ in MODELS:
            desc, _, _ = bp.pool_database(files[name], geom, dbs, label)
            qb = np.load(files[name][q][label], mmap_mode="r")
            qd = torch.from_numpy(np.asarray(qb[cov]))
            ranked = bp.topk_ranked(desc, qd, device, chunk=256)
            rec, _, _ = bp.recall_from_ranked(ranked, gt)
            r1[(q, name)], r10[(q, name)] = rec[1], rec[10]
            del desc, qd, ranked
        del gt
        print(f"  scored query={q}  ({sizes[q][0]:,} db x {sizes[q][1]:,} q)", flush=True)

    for title, table in (("R@1", r1), ("R@10", r10)):
        print(f"\n\nBrisbane {title} — each traverse as QUERY, other four pooled as reference\n")
        print(f"  {'model':31s}" + "".join(f"{q:>10s}" for q in TRAV) + f"{'mean':>10s}")
        for name, _, _, _ in MODELS:
            vals = [table[(q, name)] for q in TRAV]
            print(f"  {name:31s}" + "".join(f"{v:>10.3f}" for v in vals)
                  + f"{sum(vals) / len(vals):>10.3f}")
        print(f"  {'':31s}" + "".join(f"{sizes[q][1]:>10,}" for q in TRAV) + f"{'queries':>10s}")


if __name__ == "__main__":
    main()
