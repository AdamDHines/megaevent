"""NSAVP heading selectivity: same- vs cross-heading recall, and top-1 provenance.

    pixi run python3 scripts/nsavp_provenance.py

NSAVP's route-0 traverses cover one route in both directions, so it is the dataset that can
ask: *does a method recognise a place, or a place seen from one heading?* Three
measurements per method, both query directions:

* **same-heading cell** — the published pairwise cell (``R0_FS0 -> R0_FA0``, snow
  reference, afternoon query, both forward). Gated against the cached ledger.
* **cross-heading cell** — the same query against the *opposite*-heading snow reference
  (``R0_RS0 -> R0_FA0``). Identical route, identical condition gap, the only change is
  180 deg of viewpoint. The drop from same to cross is the heading-selectivity cost: what
  the method loses when the map was recorded the other way, which a deployed robot cannot
  choose.
* **balanced-gallery provenance** — both snow traverses pooled ({FS0, RS0}), so every
  place is present at both headings and the method picks. The fraction of top-1s drawn
  from the same-heading traverse is its revealed heading preference
  (``brisbane_pooled.pool_database`` already tags every gallery row with its source).

Everything runs off the cached pairwise banks; geometry, GT and ranking are
``scripts/pairwise_sunset_ref.py``'s own (own-grid rows keep their own grids and chunking).
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

import brisbane_pooled as bp  # noqa: E402
import pairwise_sunset_ref as psr  # noqa: E402
from src.imagevpr import build_gt  # noqa: E402

DATASET = "nsavp"
# (query, same-heading reference, cross-heading reference). Snow is the reference pair on
# both sides so the appearance gap is held fixed while the heading flips.
DIRECTIONS = [("R0_FA0", "R0_FS0", "R0_RS0"),
              ("R0_RA0", "R0_RS0", "R0_FS0")]
TRAVERSES = sorted({t for row in DIRECTIONS for t in row})
# No NSAVP banks for the acstats fairness arm, and it is not a roster method.
EXCLUDE = {"MegaLoc acstats"}


def galleries(name, kind, files, arms, geom):
    """{traverse: (desc, xy)} for one method, plus a pooled-pair builder."""
    if kind == "bank":
        def single(t):
            desc, xy, _ = bp.pool_database(files[name], geom, [t], name)
            return desc, xy
    else:
        def single(t):
            return arms[name][t]

    def pooled(ts):
        parts = [single(t) for t in ts]
        desc = torch.cat([p[0] for p in parts])
        xy = np.concatenate([p[1] for p in parts])
        source = np.concatenate([np.full(len(p[1]), i, dtype=np.int16)
                                 for i, p in enumerate(parts)])
        return desc, xy, source
    return single, pooled


def query_bank(name, kind, files, arms, geom, query):
    if kind == "bank":
        q_xy, q_cov, _, _ = geom[query]
        bank = np.load(files[name][query][name], mmap_mode="r")
        return torch.from_numpy(np.asarray(bank[q_cov])), q_xy[q_cov]
    return arms[name][query]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threshold-m", type=float, default=psr.THRESHOLD_M)
    ap.add_argument("--methods", nargs="+", default=None, metavar="NAME")
    ap.add_argument("--out-json", default="output/heading/nsavp_provenance.json")
    cli = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = psr.CONFIGS[DATASET]
    roster = [(n, k) for n, k, _, _ in psr.models_for(DATASET) if n not in EXCLUDE]
    if cli.methods:
        roster = [(n, k) for n, k in roster if n in cli.methods]

    ledger_path = f"{psr.V8}/pairwise_sunset_ref_{DATASET}.json"
    with open(ledger_path) as handle:
        ledger = json.load(handle)

    geom = psr.geometry(DATASET)
    files = {}
    for name, _, dirs, _ in psr.models_for(DATASET, "bank"):
        if name in EXCLUDE or (name, "bank") not in roster:
            continue
        bank_dir, template = dirs[DATASET]
        files[name] = {t: {name: os.path.join(bank_dir, template.format(t=t))}
                       for t in TRAVERSES}
        for t in TRAVERSES:
            if not os.path.exists(files[name][t][name]):
                raise SystemExit(f"{name}: missing bank {files[name][t][name]}")
    arms = psr.own_grid_arms(DATASET, geom, TRAVERSES)

    results = {}
    for name, kind in roster:
        chunk, db_chunk = psr.OWN_GRID_CHUNK.get(name, (cfg["chunk"], cfg["db_chunk"]))
        single, pooled = galleries(name, kind, files, arms, geom)
        entry = {}
        for query, same, cross in DIRECTIONS:
            q_desc, q_xy = query_bank(name, kind, files, arms, geom, query)
            cell = {}
            for label, ref in (("same", same), ("cross", cross)):
                db_desc, db_xy = single(ref)
                gt = build_gt(db_xy, q_xy, cli.threshold_m)
                ranked = bp.topk_ranked(db_desc, q_desc, device, k=max(psr.KS),
                                        chunk=chunk, db_chunk=db_chunk)
                rec, _, scorable = bp.recall_from_ranked(ranked, gt, ks=psr.KS)
                cell[label] = {"reference": ref, "n_database": int(db_desc.shape[0]),
                               "scorable": int(scorable.sum()),
                               "recall": {str(k): round(float(rec[k]), 4)
                                          for k in psr.KS}}
                if label == "same":
                    want = (ledger["results"].get(f"{ref}->{query}", {})
                            .get("methods", {}).get(name))
                    if want is not None:
                        w = want["recall"]["native"]
                        worst = max(abs(rec[k] - w[str(k)]) for k in psr.KS)
                        cell[label]["ledger_check_max_abs_diff"] = float(worst)
                        if worst >= 5e-4:
                            raise SystemExit(f"{name} {ref}->{query}: recomputed recall "
                                             f"diverges from the ledger by {worst:.2e}")
                del db_desc, gt, ranked

            db_desc, db_xy, source = pooled([same, cross])
            gt = build_gt(db_xy, q_xy, cli.threshold_m)
            ranked = bp.topk_ranked(db_desc, q_desc, device, k=1,
                                    chunk=chunk, db_chunk=db_chunk)
            scorable = gt.sum(0) > 0
            hit1 = gt[ranked[0], np.arange(gt.shape[1])]
            src1 = source[ranked[0]]
            cell["balanced"] = {
                "references": [same, cross],
                "n_database": int(db_desc.shape[0]),
                "r1": round(float(hit1[scorable].mean()), 4),
                "top1_from_same_heading": round(float((src1[scorable] == 0).mean()), 4),
                "correct_top1_from_same_heading": (
                    round(float((src1[scorable & hit1] == 0).mean()), 4)
                    if (scorable & hit1).any() else None),
            }
            del db_desc, gt, ranked
            entry[query] = cell
            s1, c1 = cell["same"]["recall"]["1"], cell["cross"]["recall"]["1"]
            print(f"  {name:<18s} {query}: same {s1:.3f}  cross {c1:.3f}  "
                  f"drop {s1 - c1:+.3f}  | balanced top-1 from same-heading "
                  f"{cell['balanced']['top1_from_same_heading']:.3f} "
                  f"(correct-only {cell['balanced']['correct_top1_from_same_heading']})",
                  flush=True)
        results[name] = entry

    out = {"dataset": DATASET, "threshold_m": cli.threshold_m,
           "ledger": ledger_path,
           "directions": [{"query": q, "same": s, "cross": c} for q, s, c in DIRECTIONS],
           "note": "same = the published pairwise cell (gated against the ledger); cross "
                   "= identical route and condition gap, opposite heading; balanced = "
                   "both snow traverses pooled so every place exists at both headings — "
                   "top1_from_same_heading is the method's revealed heading preference.",
           "results": results}
    os.makedirs(os.path.dirname(cli.out_json), exist_ok=True)
    tmp = cli.out_json + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(out, handle, indent=1)
    os.replace(tmp, cli.out_json)
    print(f"\n-> {cli.out_json}")


if __name__ == "__main__":
    main()
