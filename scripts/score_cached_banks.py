"""Score an image set's cached descriptor banks without materialising the similarity matrix.

    pixi run python3 scripts/score_cached_banks.py -d pitts250k \
        --bank megaloc_countmask_r322 --validate

``src/imagevpr.py`` scores by building the whole ``[n_db, n_q]`` matrix, saving it, and
handing it to ``recallAtK``, whose ``argsort(0)`` is an int64 array twice its size. On
Pitts250k at 8448-d that is 2.8 GB + 5.6 GB on top of a 2.8 GB bank, and it OOM-killed a
31 GB box. Nothing is wrong with that path — it is what produced every other image-set
number here — but Pitts250k is the one gallery it does not fit on.

So this reuses ``brisbane_pooled.topk_ranked``, the streamed ranker written for exactly this
reason on NSAVP's 101k gallery: the queries stream past a resident database and the peak
allocation is one ``[n_db, chunk]`` block. The ground truth, the dataset adapter and the
recall definition are unchanged, and ``--validate`` proves it by scoring the same banks both
ways and refusing to continue if they disagree.

Only the native descriptor is scored. Whitening is a separate space and is not what the banks
were cached for.
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

from src import inference as inf  # noqa: E402
from src.imagesets import load_image_set  # noqa: E402
from brisbane_pooled import KS, recall_from_ranked, topk_ranked  # noqa: E402

IMAGE_SPLITS = {
    "msls": ("database", "query"),
    "nycevent": ("database", "queries"),
    "pitts": ("ref", "query"),
    "pitts250k": ("database", "queries"),
    "tokyo247": ("database", "queries"),
}


def load_bank(out_dir, bank, split):
    """The cached ``[N, D]`` bank and the paths it was built from, in row order."""
    array = np.load(os.path.join(out_dir, f"{bank}_{split}.npy"), mmap_mode="r")
    with open(os.path.join(out_dir, f"{bank}_{split}_paths.txt")) as handle:
        paths = handle.read().split()
    if len(paths) != len(array):
        raise SystemExit(f"{bank}_{split}: {len(array)} rows but {len(paths)} paths")
    return array, paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", "-d", required=True, choices=sorted(IMAGE_SPLITS))
    ap.add_argument("--bank", required=True,
                    help="the method tag the banks were cached under, e.g. "
                         "'megaloc_countmask_r322'")
    ap.add_argument("--data-dir", default="/media/adam/vprdatasets/megaevent")
    ap.add_argument("--feature-dir", default="/media/adam/vprdatasets/megaevent/evaluations")
    ap.add_argument("--positive-dist-threshold", type=float, default=25.0)
    ap.add_argument("--score-chunk", type=int, default=256)
    ap.add_argument("--db-chunk", type=int, default=40000,
                    help="database rows resident on the GPU at once")
    ap.add_argument("--validate", action="store_true",
                    help="also score with src.inference.recall_at_k and require agreement. "
                         "Needs room for the full matrix, so use it on a small gallery to "
                         "certify the streamed path, not on the one that OOMs.")
    ap.add_argument("--out-json", default=None)
    cli = ap.parse_args()
    cli.ref, cli.query = IMAGE_SPLITS[cli.dataset]
    cli.limit = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(cli.feature_dir, cli.dataset)
    dataset = load_image_set(cli)
    gt = dataset.gt
    scorable = int((gt.sum(0) > 0).sum())
    print(f"{cli.dataset}: {len(dataset.db_paths)} database x {len(dataset.q_paths)} queries, "
          f"{scorable} scorable @ {cli.positive_dist_threshold:g} m, "
          f"positives/query mean {gt.sum(0).mean():.1f}")

    db, db_paths = load_bank(out_dir, cli.bank, cli.ref)
    queries, q_paths = load_bank(out_dir, cli.bank, cli.query)
    # Row order is the only thing tying a descriptor to a coordinate. A bank built from a
    # different listing of the same folder would score silently and be wrong. The cache
    # records basenames, which are unique within a split and carry the UTM coordinate for
    # the geographic sets, so that is what is compared — order included.
    def stems(paths):
        return [os.path.basename(path) for path in paths]

    if db_paths != stems(dataset.db_paths) or q_paths != stems(dataset.q_paths):
        raise SystemExit(f"{cli.bank}: cached banks were built from a different file list "
                         f"than the dataset adapter returns; the ground truth would not line "
                         f"up with the descriptors")

    db_t = torch.from_numpy(np.asarray(db))
    q_t = torch.from_numpy(np.asarray(queries))

    ranked = topk_ranked(db_t, q_t, device, chunk=cli.score_chunk, db_chunk=cli.db_chunk)
    recall, curve, mask = recall_from_ranked(ranked, gt)
    print(f"  [native, streamed] " + "  ".join(f"R@{k}={recall[k]:.4f}" for k in KS))

    if cli.validate:
        reference = inf.recall_at_k(inf.sim_matrix(db_t, q_t, device), gt)
        print(f"  [native, recall_at_k] " + "  ".join(f"R@{k}={reference[k]:.4f}" for k in KS))
        worst = max(abs(recall[k] - reference[k]) for k in KS)
        if worst > 1e-9:
            raise SystemExit(f"streamed and dense scoring disagree by {worst:.2e}")
        print(f"  agreement exact ({worst:.1e})")

    results = {"dataset": cli.dataset, "method": cli.bank.split("_")[0],
               "native_metric": "cosine", "database": cli.ref, "queries": cli.query,
               "n_database": len(dataset.db_paths), "n_queries": len(dataset.q_paths),
               "threshold_m": cli.positive_dist_threshold,
               "excluded_database": dataset.excluded_db,
               "excluded_queries": dataset.excluded_q,
               "scorable_queries": scorable,
               "positives_per_query_mean": float(gt.sum(0).mean()),
               "recall": {"native": {str(k): recall[k] for k in KS}},
               "recall_curve": {"native": {str(n): v for n, v in curve.items()}},
               "meta": {"bank": cli.bank, "scored_by": "scripts/score_cached_banks.py",
                        "ranker": "streamed topk_ranked", "validated": bool(cli.validate)}}
    out_json = cli.out_json or os.path.join(out_dir, f"results_{cli.bank}.json")
    with open(out_json, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\n-> {out_json}")


if __name__ == "__main__":
    main()
