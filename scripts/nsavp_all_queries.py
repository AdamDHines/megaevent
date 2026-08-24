"""Every NSAVP route-0 traverse as the query, the other five pooled as the reference.

    pixi run python3 scripts/nsavp_all_queries.py

The Brisbane all-queries sweep (`scripts/brisbane_all_queries.py`) applied to NSAVP: one
traverse queries, the other five concatenate into a single gallery, nearest descriptor anywhere
wins, 25 m Euclidean, native descriptor, no PCA. Scored off the cached banks, so the R0_FA0 row
reproduces `table_native.py`'s NSAVP cell.

**Route 0 only.** R1 is a different route with zero overlap — every R0_FA0 query frame has an
R0_* pose within 25 m and none has an R1_* pose within 25 m (median 825-848 m), so pooling R1 in
would add ~77k pure distractors and measure something else.

The six traverses are the full condition x direction grid: forward/reverse (F/R) crossed with
afternoon/night/sunset (A/N/S). Three of the five galleries a forward query faces are reverse
traverses, which is the heading-selectivity part of this benchmark.

Event-GeM is its **global** stage; `+rerank` needs a keypoint store per pairing.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import brisbane_pooled as bp  # noqa: E402
import nsavp_pooled as npl  # noqa: E402
from src.imagevpr import build_gt  # noqa: E402

TRAV = ["R0_FA0", "R0_FN0", "R0_FS0", "R0_RA0", "R0_RN0", "R0_RS0"]
ROOT = "/media/adam/vprdatasets/megaevent"
EL = "/media/adam/vprdatasets/eventlab"
MODELS = [
    ("megaevent B-MLoc  real s500",  f"{ROOT}/nsavp_pooled",   "v5b_mloc_s500",   "r322ba50"),
    ("megaevent B-SALAD real s1000", f"{ROOT}/nsavp_pooled",   "v5b_salad_s1000", "r322ba50"),
    ("megaevent S-MLoc  real s1000", f"{ROOT}/nsavp_pooled",   "v6s_mloc_s1000",  "r322ba50"),
    ("megaevent S-SALAD real s1000", f"{ROOT}/nsavp_pooled",   "v6s_salad_s1000", "r322ba50"),
    ("megaevent B-MLoc  ship s3500", f"{ROOT}/nsavp_pooled",   "pm_s3500",        "r322ba50"),
    ("megaevent B-SALAD ship s8000", f"{ROOT}/nsavp_pooled",   "b_salad_s8000",   "r322ba50"),
    ("megaevent S-MLoc  ship s3500", f"{ROOT}/nsavp_pooled",   "s_mloc_s3500",    "r322ba50"),
    ("megaevent S-SALAD ship s2000", f"{ROOT}/nsavp_pooled",   "s_salad_s2000",   "r322ba50"),
    ("MegaLoc",                      f"{ROOT}/megaloc_pooled",  "megaloc",        "r322ba50"),
    ("Event-GeM (global)",           f"{ROOT}/eventgem_pooled", "eventgem",       "r240ba50"),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geom = npl.traverse_geometry(os.path.join(EL, "nsavp"), TRAV, 50)

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
        _, xy, _ = bp.pool_database(files[MODELS[0][0]], geom, dbs, MODELS[0][2])
        gt = build_gt(xy, qxy[cov], 25.0)
        sizes[q] = (xy.shape[0], int(cov.sum()))
        for name, _, label, _ in MODELS:
            desc, _, _ = bp.pool_database(files[name], geom, dbs, label)
            qb = np.load(files[name][q][label], mmap_mode="r")
            qd = torch.from_numpy(np.asarray(qb[cov]))
            ranked = bp.topk_ranked(desc, qd, device, chunk=256, db_chunk=npl.DB_CHUNK)
            rec, _, _ = bp.recall_from_ranked(ranked, gt)
            r1[(q, name)], r10[(q, name)] = rec[1], rec[10]
            del desc, qd, ranked
        del gt
        print(f"  scored query={q}  ({sizes[q][0]:,} db x {sizes[q][1]:,} q)", flush=True)

    for title, table in (("R@1", r1), ("R@10", r10)):
        print(f"\n\nNSAVP {title} — each R0 traverse as QUERY, other five pooled as reference\n")
        print(f"  {'model':31s}" + "".join(f"{q:>9s}" for q in TRAV) + f"{'mean':>9s}")
        for name, _, _, _ in MODELS:
            vals = [table[(q, name)] for q in TRAV]
            print(f"  {name:31s}" + "".join(f"{v:>9.3f}" for v in vals)
                  + f"{sum(vals) / len(vals):>9.3f}")
        print(f"  {'':31s}" + "".join(f"{sizes[q][1]:>9,}" for q in TRAV) + f"{'queries':>9s}")


if __name__ == "__main__":
    main()
