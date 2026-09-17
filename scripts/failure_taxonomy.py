"""Failure taxonomy on the pairwise single-reference cells: outranked vs nowhere.

    pixi run python3 scripts/failure_taxonomy.py --dataset brisbane_event
    pixi run python3 scripts/failure_taxonomy.py --dataset nsavp

R@1 says how often a method fails; it says nothing about *how*. For every miss there are
two very different stories:

* **outranked** — the correct place is represented and highly ranked, but a distractor
  edges it out by a small cosine margin. Recoverable: a larger k, temporal pooling or any
  second stage gets it back.
* **nowhere** — the correct place is buried deep in the ranking. The descriptor simply does
  not represent this place under this condition change; nothing downstream can recover it.

This script classifies every miss of every roster method on the published pairwise cells
(the ``scripts/pairwise_sunset_ref.py`` protocol: one reference traverse, one query
traverse, 25 m, native) by the correct answer's **global rank** — the count of gallery rows
strictly above the best-correct cosine, the two-pass trick from
``scripts/springfield_diag.py``. Strict ``>`` makes the rank tie-safe, so it is immune to
the ``db_chunk`` tie merge that moves R@1 by ~1e-3 for constant-norm descriptor banks.

The gate: the top-k ranking recomputed here must reproduce the cached ledger
(``v8_bench/pairwise_sunset_ref_<dataset>.json``) to 5e-4 at every k, using each cell's own
chunking. A taxonomy that does not sit on exactly the published ranking is not reported.

Roster is restricted to bank-grid methods (they share one geometry, hence one GT per cell).
The default picks ours plus the strongest RGB controls on these cells — MegaLoc is the
named flagship but MixVPR and QAA set the bar (see docs/deepdive_notes.md margin table).
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

DEFAULT_METHODS = ["megaevent v8-B", "MegaLoc", "MixVPR", "QAA"]
# Failure classes, by the correct answer's global rank. outranked@5 would be recovered by
# R@5; outranked@100 by any plausible shortlist/second stage; past 100 the place is for
# practical purposes not represented.
RANK_EDGES = (5, 100)
RANK_HIST_EDGES = (1, 2, 5, 10, 20, 50, 100, 1000)


def best_correct(db_desc, q_desc, gt, device, db_chunk=20000, q_chunk=4096):
    """``(bc_c, bc_rank)`` — best-correct cosine per query and its 1-based global rank.

    Two streamed passes (springfield_diag.py's dump machinery, generalised): pass 0 takes
    the masked max over GT rows, pass 1 counts sims strictly above it. Exact regardless of
    chunking — max and count are order-free — so no tie trap.
    """
    n_db, n_q = db_desc.shape[0], q_desc.shape[0]
    bc_c = np.full(n_q, -2.0, np.float32)
    bc_rank = np.zeros(n_q, np.int64)
    with torch.no_grad():
        for pass_no in (0, 1):
            for d in range(0, n_db, db_chunk):
                part = db_desc[d:d + db_chunk].to(device, non_blocking=True)
                for s in range(0, n_q, q_chunk):
                    block = q_desc[s:s + q_chunk].to(device, non_blocking=True)
                    sim = part @ block.T                         # [dbc, qc]
                    cols = slice(s, s + block.size(0))
                    if pass_no == 0:
                        g = torch.from_numpy(np.ascontiguousarray(
                            gt[d:d + db_chunk, cols])).to(device)
                        mv = sim.masked_fill(~g, -2.0).max(dim=0).values.cpu().numpy()
                        np.maximum(bc_c[cols], mv, out=bc_c[cols])
                        del g
                    else:
                        thr = torch.from_numpy(bc_c[cols]).to(device)
                        bc_rank[cols] += (sim > thr[None, :]).sum(dim=0).cpu().numpy()
                    del sim, block
                del part
            torch.cuda.empty_cache()
    bc_rank += 1
    return bc_c, bc_rank


def classify(hit1, bc_rank, scorable):
    """{class: fraction of scorable queries}, classes summing to 1."""
    n = int(scorable.sum())
    miss = scorable & ~hit1
    out = {"correct": float((scorable & hit1).sum() / n)}
    lo = 0
    for edge in RANK_EDGES:
        out[f"outranked@{edge}"] = float(
            (miss & (bc_rank > lo) & (bc_rank <= edge)).sum() / n)
        lo = edge
    out["nowhere"] = float((miss & (bc_rank > lo)).sum() / n)
    assert abs(sum(out.values()) - 1.0) < 1e-6
    return out


def taxonomy_cell(dataset, geom, files, ref, query, methods, device, threshold, ledger,
                  arrays):
    """{method: stats} for one (reference, query) cell, gated against the ledger.

    ``arrays`` collects the per-query evidence (hit1, bc_rank, margin) per method for the
    npz sidecar — the cell shares one geometry and one GT across the bank-grid roster, so
    query indices are comparable between methods and miss overlaps are meaningful.
    """
    cfg = psr.CONFIGS[dataset]
    q_xy_full, q_cov, _, _ = geom[query]
    q_xy = q_xy_full[q_cov]
    cell_key = f"{ref}->{query}"
    arrays[f"{cell_key}|q_xy"] = q_xy.astype(np.float32)
    out = {}
    for name in methods:
        db_desc, db_xy, _ = bp.pool_database(files[name], geom, [ref], name)
        q_bank = np.load(files[name][query][name], mmap_mode="r")
        q_desc = torch.from_numpy(np.asarray(q_bank[q_cov]))
        gt = build_gt(db_xy, q_xy, threshold)
        scorable = gt.sum(0) > 0

        # The published ranking, with the cell's own chunking; the gate lives on it.
        ranked, scores = bp.topk_ranked(db_desc, q_desc, device, k=max(psr.KS),
                                        chunk=cfg["chunk"], db_chunk=cfg["db_chunk"],
                                        return_scores=True)
        rec, _, _ = bp.recall_from_ranked(ranked, gt, ks=psr.KS)
        want = ledger["results"][f"{ref}->{query}"]["methods"][name]["recall"]["native"]
        worst = max(abs(rec[k] - want[str(k)]) for k in psr.KS)
        if worst >= 5e-4:
            raise SystemExit(
                f"{ref}->{query} {name}: recomputed recall diverges from the ledger by "
                f"{worst:.2e} — the taxonomy would not sit on the published ranking")

        bc_c, bc_rank = best_correct(db_desc, q_desc, gt, device)
        hit1 = gt[ranked[0], np.arange(gt.shape[1])]
        miss = scorable & ~hit1
        margin = scores[0] - bc_c                     # top-1 cosine minus best-correct
        wrong_d = np.linalg.norm(db_xy[ranked[0]] - q_xy, axis=1)

        stats = {
            "recall_check_max_abs_diff": float(worst),
            "n_database": int(db_desc.shape[0]),
            "n_scorable": int(scorable.sum()),
            "n_miss": int(miss.sum()),
            "classes": classify(hit1, bc_rank, scorable),
            "miss_bc_rank_median": int(np.median(bc_rank[miss])) if miss.any() else None,
            "miss_bc_rank_hist": {
                f"<={e}": int((bc_rank[miss] <= e).sum()) for e in RANK_HIST_EDGES},
            "miss_margin_median": (round(float(np.median(margin[miss])), 4)
                                   if miss.any() else None),
            "miss_margin_p90": (round(float(np.quantile(margin[miss], 0.9)), 4)
                                if miss.any() else None),
            "miss_wrong_dist_med_m": (round(float(np.median(wrong_d[miss])), 1)
                                      if miss.any() else None),
        }
        out[name] = stats
        arrays[f"{cell_key}|{name}|hit1"] = hit1
        arrays[f"{cell_key}|{name}|scorable"] = scorable
        arrays[f"{cell_key}|{name}|bc_rank"] = bc_rank.astype(np.int32)
        arrays[f"{cell_key}|{name}|margin"] = margin.astype(np.float32)
        cls = stats["classes"]
        print(f"    {name:<16s} correct {cls['correct']:.3f}  "
              + "  ".join(f"{k} {v:.3f}" for k, v in cls.items() if k != "correct")
              + f"  | miss rank med {stats['miss_bc_rank_median']}"
                f"  margin med {stats['miss_margin_median']}", flush=True)
        del db_desc, q_desc, q_bank, gt, ranked, scores
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="brisbane_event", choices=sorted(psr.CONFIGS))
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, metavar="NAME",
                    help="bank-grid roster rows, names as pairwise_sunset_ref prints them")
    ap.add_argument("--threshold-m", type=float, default=psr.THRESHOLD_M)
    ap.add_argument("--out-json", default=None,
                    help="default output/failure_taxonomy/<dataset>.json")
    cli = ap.parse_args()

    bank_names = [m[0] for m in psr.models_for(cli.dataset, "bank")]
    unknown = [m for m in cli.methods if m not in bank_names]
    if unknown:
        raise SystemExit(f"not bank-grid rows on {cli.dataset}: {unknown} "
                         f"(own-grid methods have their own geometry; this taxonomy "
                         f"shares one GT per cell). Available: {bank_names}")

    ledger_path = f"{psr.V8}/pairwise_sunset_ref_{cli.dataset}.json"
    with open(ledger_path) as handle:
        ledger = json.load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = psr.CONFIGS[cli.dataset]
    pairs = [tuple(p) for p in cfg["pairs"]]
    print(f"{cli.dataset}: " + ", ".join(f"{r}->{q}" for r, q in pairs)
          + f"   {cli.threshold_m:g} m, native — gated against {ledger_path}")

    geom = psr.geometry(cli.dataset)
    needed = sorted({t for pair in pairs for t in pair})
    # psr.bank_files insists every roster row has banks; only the requested rows need them
    # here (e.g. "MegaLoc acstats" has no NSAVP banks and is not in this taxonomy).
    files = {}
    for name, _, dirs, _ in psr.models_for(cli.dataset, "bank"):
        if name not in cli.methods:
            continue
        bank_dir, template = dirs[cli.dataset]
        files[name] = {t: {name: os.path.join(bank_dir, template.format(t=t))}
                       for t in needed}
        for t in needed:
            if not os.path.exists(files[name][t][name]):
                raise SystemExit(f"{name}: missing bank {files[name][t][name]}")

    results = {}
    arrays = {}
    for ref, query in pairs:
        print(f"  {ref} -> {query}")
        key = f"{ref}->{query}"
        results[key] = taxonomy_cell(
            cli.dataset, geom, files, ref, query, cli.methods, device,
            cli.threshold_m, ledger, arrays)
        # Is the residual shared? Jaccard between miss sets, and where the queries OUR
        # model cannot represent (rank > last edge) land under each control.
        misses = {m: (~arrays[f"{key}|{m}|hit1"]) & arrays[f"{key}|{m}|scorable"]
                  for m in cli.methods}
        results[key]["_miss_jaccard"] = {
            f"{a} & {b}": round(float((misses[a] & misses[b]).sum()
                                      / max((misses[a] | misses[b]).sum(), 1)), 3)
            for i, a in enumerate(cli.methods) for b in cli.methods[i + 1:]}
        ours = cli.methods[0]
        deep = misses[ours] & (arrays[f"{key}|{ours}|bc_rank"] > RANK_EDGES[-1])
        if deep.any():
            results[key]["_ours_nowhere_under_controls"] = {
                m: {"also_miss": round(float(misses[m][deep].mean()), 3),
                    "also_nowhere": round(float(
                        (misses[m] & (arrays[f"{key}|{m}|bc_rank"] > RANK_EDGES[-1])
                         )[deep].mean()), 3)}
                for m in cli.methods[1:]}

    out = {"dataset": cli.dataset,
           "protocol": ledger["protocol"],
           "threshold_m": cli.threshold_m,
           "ledger": ledger_path,
           "rank_edges": list(RANK_EDGES),
           "note": "classes partition scorable queries. outranked@5: miss with the "
                   "best-correct row's global rank <=5 (recoverable at R@5); "
                   "outranked@100: rank 6..100 (recoverable by a shortlist stage); "
                   "nowhere: rank >100 (the place is not represented under this "
                   "condition change). Ranks are strict-> tie-safe.",
           "methods": cli.methods,
           "results": results}
    path = cli.out_json or os.path.join("output", "failure_taxonomy", f"{cli.dataset}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(out, handle, indent=1)
    os.replace(tmp, path)
    npz_path = os.path.splitext(path)[0] + "_perquery.npz"
    np.savez_compressed(npz_path, **arrays)
    print(f"\n-> {path}\n-> {npz_path}")


if __name__ == "__main__":
    main()
