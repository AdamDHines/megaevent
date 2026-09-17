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

import argparse
import json
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

# The v8-era roster, on `accumulate` — the representation v8 trains and evaluates on, and
# the only one on which our cells and the controls' cells are the same frames. MegaLoc is
# not the strongest control here (mixvpr and qaa both beat it on the pooled cell), which is
# the reason this roster exists: a per-traverse breakdown against ONE control cannot show
# whether an advantage is cross-condition stability or a lucky pairing. Every bank listed
# is already on disk; each control keeps its own published resolution in its tag.
ROOT = "/media/adam/vprdatasets/megaevent"
MODELS_V8_ACCUM = [
    ("ours v8-B",   f"{ROOT}/v8_bench/brisbane", "v8",       "r322ba50"),
    ("MixVPR",      f"{ROOT}/mixvpr_pooled",     "mixvpr",   "r320ba50_accum"),
    ("QAA",         f"{ROOT}/qaa_pooled",        "qaa",      "r322ba50_accum"),
    ("BoQ",         f"{ROOT}/boq_pooled",        "boq",      "r322ba50_accum"),
    ("MegaLoc",     f"{ROOT}/megaloc_pooled",    "megaloc",  "r322ba50_accum"),
    ("SALAD",       f"{ROOT}/salad_pooled",      "salad",    "r322ba50_accum"),
]
ROSTERS = {"v5": MODELS, "v8accum": MODELS_V8_ACCUM}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db-chunk", type=int, default=None,
                    help="split the gallery into chunks of this many rows when ranking. "
                         "topk_ranked merges per-chunk top-k into the same global top-k "
                         "(brisbane_pooled.py:81), so this only caps resident VRAM — needed "
                         "when the card is shared and the pooled 54,620 x 8448 bank (1.8 GB) "
                         "will not fit beside another job.")
    ap.add_argument("--score-device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="'cpu' trades minutes of ranking for immunity to a busy GPU.")
    ap.add_argument("--queries", default=None,
                    help="comma-separated query traverses to score (default: all five). "
                         "The reported protocol only queries sunset1, so '--queries sunset1' "
                         "is 5 configurations instead of 25 and reads a fifth of the banks; "
                         "the other rows are a symmetry diagnostic.")
    ap.add_argument("--roster", default="v5", choices=sorted(ROSTERS),
                    help="'v5' is the original countmask roster this script shipped with "
                         "(the default, so every published table it produced still "
                         "reproduces); 'v8accum' is the current roster on accumulate.")
    ap.add_argument("--out-json", default=None,
                    help="also serialize the full R@1 grid (query x database x method, "
                         "POOLED included) — the stdout tables are lossy and scrollback "
                         "is not a ledger. scripts/figure_confusion.py consumes this.")
    cli = ap.parse_args()
    global MODELS
    MODELS = ROSTERS[cli.roster]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cli.score_device != "auto":
        device = torch.device(cli.score_device)
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
    queries = TRAV if not cli.queries else [q.strip() for q in cli.queries.split(",")]
    unknown = [q for q in queries if q not in TRAV]
    if unknown:
        raise SystemExit(f"unknown query traverse(s) {unknown}; expected from {TRAV}")
    configs = []
    for q in queries:
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
            ranked = bp.topk_ranked(desc, qd, device, chunk=256, db_chunk=cli.db_chunk)
            rec, _, _ = bp.recall_from_ranked(ranked, gt)
            scores[(key, name)] = rec[1]
            del desc, qd, ranked
        del gt
        print(f"  done {key}", flush=True)

    if cli.out_json:
        doc = {"dataset": "brisbane_event", "roster": cli.roster,
               "threshold_m": 25.0, "traverses": TRAV, "queries": queries,
               "protocol": "single-traverse cells are the pairwise-protocol geometry "
                           "(one query traverse against one database traverse, 25 m, "
                           "native); POOLED is the retired pooled membership, kept as a "
                           "diagnostic reference only",
               "results": {q: {db: {name: float(scores[((q, db), name)])
                                    for name, _, _, _ in MODELS}
                               for db in [t for t in TRAV if t != q] + ["POOLED"]}
                           for q in queries}}
        os.makedirs(os.path.dirname(cli.out_json) or ".", exist_ok=True)
        tmp = cli.out_json + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(doc, handle, indent=1)
        os.replace(tmp, cli.out_json)
        print(f"\n-> {cli.out_json}")

    cols = TRAV + ["POOLED"]
    print("\n\nBrisbane R@1 — rows = QUERY traverse, columns = DATABASE (single traverse), "
          "POOLED = all four at once (the reported protocol)\n")
    for name, _, _, _ in MODELS:
        print(f"{name}")
        print(f"  {'query':10s}" + "".join(f"{c:>10s}" for c in cols) + f"{'best1':>9s}")
        for q in queries:
            singles = {db: scores[((q, db), name)] for db in TRAV if db != q}
            row = "".join(f"{scores[((q, c), name)]:>10.3f}" if (q, c) in
                          [(q, x) for x in singles] + [(q, "POOLED")] else f"{'--':>10s}"
                          for c in cols)
            print(f"  {q:10s}{row}{max(singles.values()):>9.3f}")
        print()

    print("\nPOOLED R@1 by query — every model, the reported protocol\n")
    print(f"  {'model':26s}" + "".join(f"{q:>10s}" for q in queries))
    for name, _, _, _ in MODELS:
        print(f"  {name:26s}"
              + "".join(f"{scores[((q,'POOLED'), name)]:>10.3f}" for q in queries))

    if len(queries) < len(TRAV):
        print("\n(symmetry block skipped: it needs every traverse as a query)")
        return
    ref = MODELS[0][0]
    print(f"\nSymmetry — R@1(a as query, b as db) vs R@1(b as query, a as db), {ref}\n")
    print(f"  {'pair':24s}{'a->b':>8s}{'b->a':>8s}{'diff':>8s}")
    for i, a in enumerate(TRAV):
        for b in TRAV[i + 1:]:
            ab, ba = scores[((a, b), ref)], scores[((b, a), ref)]
            print(f"  {a + ' / ' + b:24s}{ab:>8.3f}{ba:>8.3f}{ab - ba:>+8.3f}")


if __name__ == "__main__":
    main()
