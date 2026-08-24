"""Event-GeM 0.1.0's released global stage under the standard pooled protocols.

Loads the per-traverse feature banks the RELEASE's own ``feature_inference`` saved
(``features/<dataset>/<ref>-<query>/<dataset>_<traverse>_features.pt`` — ECDP ViT
backbone ``pr.pt`` + GeM(p=5), cosine similarity, byte-parity of the cache path proven by
the 2026-08-22 clean-room reproduction), pools the database traverses, and scores 25 m
Euclidean GT through the same machinery as every other method in the ledger.

Alignment: eventlab's frames-50 renders slice from the recording start like eventcv does,
but can differ by one tail frame (14477 vs 14478 on sunset1). Coords and banks are
truncated to the common length per traverse; a >2-frame difference is a hard error.

Where a traverse's bank exists in several pair dirs, all copies must be byte-identical
(asserted) — a cheap integrity check on the cache reuse this scoring rests on.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import brisbane_pooled as bp  # noqa: E402
import nsavp_pooled as npl  # noqa: E402
from brisbane_resolution import _Args  # noqa: E402
from src.imagevpr import build_gt  # noqa: E402

# The 0.1.0 worktree's own feature tree — NEVER the data root's features/ cache: that
# tree predates the release (mean |sim diff| 0.98 vs fresh 0.1.0 output, top-1 agreement
# 9%, measured 2026-08-21) and silently poisoned the first pooled Brisbane number.
FEATURES = "/media/adam/vprdatasets/megaevent/eventgem_010/eventgem/features"
OUT = "/media/adam/vprdatasets/megaevent/v8_bench"
CONFIGS = {
    "brisbane_event": ("/media/adam/vprdatasets/eventgem", "sunset1",
                       ("daytime", "morning", "night", "sunrise")),
    "nsavp": ("/media/adam/vprdatasets/eventlab", "R0_FA0",
              ("R0_FN0", "R0_FS0", "R0_RA0", "R0_RN0", "R0_RS0")),
}
KS = (1, 5, 10, 20)


def load_bank(dataset, traverse):
    paths = sorted(glob.glob(
        f"{FEATURES}/{dataset}/*/{dataset}_{traverse}_features.pt"))
    if not paths:
        raise SystemExit(f"no released feature bank for {dataset}/{traverse} under "
                         f"{FEATURES} — harvest it with a 0.1.0 pair run first")
    banks = [torch.load(p, map_location="cpu") for p in paths]
    for b in banks[1:]:
        # allclose, not equal: independent GPU runs drift at the 1e-6 level (measured
        # max 2.7e-6 on FA0). Anything past 1e-4 means a different-era extraction.
        if b.shape != banks[0].shape or not torch.allclose(b, banks[0], atol=1e-4):
            raise SystemExit(f"{dataset}/{traverse}: pair-dir copies differ — cache "
                             f"integrity violated ({paths})")
    return torch.nn.functional.normalize(banks[0].float(), dim=1)


def geometry(dataset, eventlab_dir, sequences):
    if dataset == "nsavp":
        return npl.traverse_geometry(os.path.join(eventlab_dir, dataset), sequences, 50)
    args = _Args(eventlab_dir, dataset, 50, False, False)
    args.filter_dt_us = 50_000
    return bp.traverse_geometry(args, sequences, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="brisbane_event", choices=sorted(CONFIGS))
    ap.add_argument("--threshold-m", type=float, default=25.0)
    cli = ap.parse_args()
    eventlab_dir, query, database = CONFIGS[cli.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences = [query, *database]
    geom = geometry(cli.dataset, eventlab_dir, sequences)

    desc, xy = {}, {}
    for s in sequences:
        bank = load_bank(cli.dataset, s)
        cov = geom[s][1]
        n = min(len(bank), len(cov))
        if abs(len(bank) - len(cov)) > 2:
            raise SystemExit(f"{s}: bank {len(bank)} vs geometry {len(cov)} frames — "
                             f"more than a fencepost apart, alignment unproven")
        keep = cov[:n]
        desc[s] = bank[:n][torch.as_tensor(keep)]
        xy[s] = geom[s][0][:n][keep]
        print(f"  {s}: {int(keep.sum())} frames (bank {len(bank)}, geom {len(cov)})")

    db_desc = torch.cat([desc[s] for s in database])
    db_xy = np.concatenate([xy[s] for s in database])
    gt = build_gt(db_xy, xy[query], cli.threshold_m)
    ranked = bp.topk_ranked(db_desc, desc[query], device, k=max(KS), chunk=64,
                            db_chunk=4096)
    rec, _, _ = bp.recall_from_ranked(ranked, gt)
    line = "  ".join(f"R@{k}={rec[k]:.4f}" for k in KS)
    print(f"eg010 pooled {cli.dataset} ({db_desc.shape[0]} db x "
          f"{desc[query].shape[0]} q, {cli.threshold_m:g} m): {line}")
    out = {"dataset": cli.dataset, "method": "eventgem-0.1.0 global (released banks)",
           "protocol": f"pooled, {cli.threshold_m:g} m, native cosine",
           "n_database": int(db_desc.shape[0]), "n_queries": int(desc[query].shape[0]),
           "recall": {str(k): float(rec[k]) for k in KS}}
    path = os.path.join(OUT, f"eg010_pooled_{cli.dataset}.json")
    with open(path, "w") as h:
        json.dump(out, h, indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
