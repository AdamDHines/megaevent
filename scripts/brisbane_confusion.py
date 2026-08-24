"""Every model's full query x database R@1 on Brisbane, from cached banks.

    pixi run python3 scripts/brisbane_confusion.py

The reported Brisbane number pools four traverses into one gallery and takes the nearest
descriptor anywhere in it. That is the protocol and it is what the ``POOLED`` column here
reproduces (it agrees with ``table_native.py`` to 3 dp). The single-traverse columns are a
**diagnostic only** — the same query scored against one traverse at a time — to show where a
pooled match could have come from. No reported number uses them.

Ground truth depends only on the geometry, so the loop is (query, database) on the outside and
models on the inside: one 25 m GT per configuration, shared by every model.

Event-GeM is its **global** stage only. Its ``+rerank`` needs a keypoint store per query/database
pairing, which is ~9 min per cell; it exists only for the two pooled configurations that were run
in full (sunset1 and daytime).
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
# (pretty, bank dir, label, tag) — grouped as they read in the paper.
MODELS = [
    ("ours B-MLoc  real s500",  f"{DAY}/realbest", "v5b_mloc_s500",   "r322ba50"),
    ("ours B-SALAD real s1000", f"{DAY}/realbest", "v5b_salad_s1000", "r322ba50"),
    ("ours S-MLoc  real s1000", f"{DAY}/realbest", "v6s_mloc_s1000",  "r322ba50"),
    ("ours S-SALAD real s1000", f"{DAY}/realbest", "v6s_salad_s1000", "r322ba50"),
    ("ours B-MLoc  ship s3500", f"{DAY}/v5",       "pm_s3500",        "r322ba50"),
    ("ours B-SALAD ship s8000", f"{DAY}/ship",     "b_salad_s8000",   "r322ba50"),
    ("ours S-MLoc  ship s3500", f"{DAY}/ship",     "s_mloc_s3500",    "r322ba50"),
    ("ours S-SALAD ship s2000", f"{DAY}/ship",     "s_salad_s2000",   "r322ba50"),
    ("MegaLoc",                 f"{DAY}/megaloc",  "megaloc",         "r322ba50"),
    ("SALAD",                   f"{DAY}/salad",    "salad",           "r322ba50"),
    ("CricaVPR",                f"{DAY}/cricavpr", "cricavpr",        "r224ba50"),
    ("MixVPR",                  f"{DAY}/mixvpr",   "mixvpr",          "r320ba50"),
    ("Event-GeM (global)",      f"{DAY}/eventgem", "eventgem",        "r240ba50"),
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
            p = files[name][t][label]
            if not os.path.exists(p):
                raise SystemExit(f"{name}: missing bank {p}")

    # config -> {model: R@1}.  Outer loop is the configuration so the GT is built once.
    configs = []
    for q in TRAV:
        configs += [(q, [db]) for db in TRAV if db != q]
        configs.append((q, [t for t in TRAV if t != q]))

    scores = {}
    for q, dbs in configs:
        key = (q, "POOLED" if len(dbs) > 1 else dbs[0])
        qxy, cov, _, _ = geom[q]
        qxy_c = qxy[cov]
        # xy is model-independent; take it from the first model's banks.
        first = MODELS[0]
        _, xy, _ = bp.pool_database(files[first[0]], geom, dbs, first[2])
        gt = build_gt(xy, qxy_c, 25.0)
        for name, _, label, _ in MODELS:
            desc, _, _ = bp.pool_database(files[name], geom, dbs, label)
            qb = np.load(files[name][q][label], mmap_mode="r")
            qd = torch.from_numpy(np.asarray(qb[cov]))
            ranked = bp.topk_ranked(desc, qd, device, chunk=256)
            rec, _, _ = bp.recall_from_ranked(ranked, gt)
            scores[(key, name)] = rec[1]
            del desc, qd, ranked
        del gt
        print(f"  done {key}", flush=True)

    cols = TRAV + ["POOLED"]
    print("\n\nBrisbane R@1 — rows = QUERY traverse, columns = DATABASE (single traverse), "
          "POOLED = all four at once (the reported protocol)\n")
    for name, _, _, _ in MODELS:
        print(f"{name}")
        print(f"  {'query':10s}" + "".join(f"{c:>10s}" for c in cols) + f"{'best1':>9s}")
        for q in TRAV:
            singles = {db: scores[((q, db), name)] for db in TRAV if db != q}
            row = "".join(f"{scores[((q, c), name)]:>10.3f}" if (q, c) in
                          [(q, x) for x in singles] + [(q, "POOLED")] else f"{'--':>10s}"
                          for c in cols)
            print(f"  {q:10s}{row}{max(singles.values()):>9.3f}")
        print()

    print("\nPOOLED R@1 by query — every model, the reported protocol\n")
    print(f"  {'model':26s}" + "".join(f"{q:>10s}" for q in TRAV))
    for name, _, _, _ in MODELS:
        print(f"  {name:26s}" + "".join(f"{scores[((q,'POOLED'), name)]:>10.3f}" for q in TRAV))

    print("\nSymmetry — R@1(a as query, b as db) vs R@1(b as query, a as db), ours S-MLoc real\n")
    ref = "ours S-MLoc  real s1000"
    print(f"  {'pair':24s}{'a->b':>8s}{'b->a':>8s}{'diff':>8s}")
    for i, a in enumerate(TRAV):
        for b in TRAV[i + 1:]:
            ab, ba = scores[((a, b), ref)], scores[((b, a), ref)]
            print(f"  {a + ' / ' + b:24s}{ab:>8.3f}{ba:>8.3f}{ab - ba:>+8.3f}")


if __name__ == "__main__":
    main()
