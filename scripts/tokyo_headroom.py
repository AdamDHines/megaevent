"""Tokyo 24/7 headroom analyses that need no re-extraction — they run off the cached banks.

Run from the repo root::

    pixi run python3 scripts/tokyo_headroom.py --which all

Three questions, all answerable from ``features/tokyo247/`` plus the query ``.npz`` headers:

``pca``
    Is ``src/inference.py``'s hardcoded ``PCA_DIM, PCA_POWER = 2048, 0.5`` at its optimum?
    Sweeps dim x power through this repo's own ``pca_fit`` / ``pca_apply`` / ``recall_at_k``,
    so the harness is the shipped one. The ``(2048, 0.5)`` cell is the reproduction check —
    it must return ``results_megaevent.json``'s value exactly before any other cell means
    anything. Also reports the fit-subsample size effect (``scoring.PCA_FIT_SAMPLES``).

``structure``
    Tokyo 24/7 is 3 times of day x 3 camera directions per location. The filenames keep the
    original ``247query_v2`` stem in field 7, so position-within-run recovers that 3x3 grid
    even though nothing records which axis is which. All 9 images at a location share the
    same ground-truth positives, so per-position recall differences are pure query
    difficulty. Also tries I2E event count as a brightness proxy for the time-of-day axis —
    it does not work, and the between/within variance ratio is reported so that stays a
    measured dead end rather than an untried idea.

``orientation``
    The database is 100% landscape and the queries are not, so the square resize was a
    natural suspect for lost recall. Scored per orientation group; it is not one.

Descriptor banks and similarity matrices are whatever ``--feature-dir`` holds, so run the
normal Tokyo evaluation first (``main.py --dataset tokyo247 --method megaevent``).
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import inference as inf  # noqa: E402
from src import scoring  # noqa: E402
from src.imagevpr import build_gt  # noqa: E402
from src.npzdata import list_npz, utm_from_paths  # noqa: E402

DEFAULT_FEATURES = "features/tokyo247"
DEFAULT_NPZ = "/media/adam/vprdatasets/megaevent/tokyo247/numpy"
MODEL = "s_salad_ft4_countmask"
DIMS = (1024, 2048, 4096, 8448)
POWERS = (0.25, 0.5, 0.75, 1.0)


def load(args):
    """(database bank, query bank, GT [n_db, n_q], query paths)."""
    db = torch.from_numpy(np.load(f"{args.feature_dir}/{MODEL}_database.npy"))
    queries = torch.from_numpy(np.load(f"{args.feature_dir}/{MODEL}_queries.npy"))
    db_paths = list_npz(f"{args.npz_root}/database")
    q_paths = list_npz(f"{args.npz_root}/queries")
    gt = build_gt(utm_from_paths(db_paths), utm_from_paths(q_paths), args.threshold_m)
    print(f"db {tuple(db.shape)}  queries {tuple(queries.shape)}  GT {gt.shape}  "
          f"{gt.sum(0).mean():.2f} positives/query  "
          f"{int((gt.sum(0) > 0).sum())}/{gt.shape[1]} scorable\n")
    return db, queries, gt, q_paths


def subsample(db, n):
    """The seeded fit subsample ``scoring.pca_fit_subsampled`` uses."""
    if db.size(0) <= n:
        return db
    generator = torch.Generator().manual_seed(scoring.PCA_FIT_SEED)
    return db[torch.randperm(db.size(0), generator=generator)[:n]]


def score(db, queries, gt, device, pca=None):
    if pca is not None:
        db, queries = inf.pca_apply(db, pca, device), inf.pca_apply(queries, pca, device)
    return inf.recall_at_k(inf.sim_matrix(db, queries, device), gt, (1, 5, 10, 20))


def run_pca(db, queries, gt, device):
    print("== PCA dim x power, on the shipped 20k-row fit subsample ==")
    native = score(db, queries, gt, device)
    print(f"native (no PCA): " + "  ".join(f"R@{k} {native[k]:.4f}" for k in (1, 5, 10, 20)))
    torch.cuda.empty_cache()

    fit = subsample(db, scoring.PCA_FIT_SAMPLES)
    results = {}
    print("\n  R@1   dim \\ power |" + "".join(f"{p:>9}" for p in POWERS))
    print("  " + "-" * (19 + 9 * len(POWERS)))
    for dim in DIMS:
        row = f"  {dim:>17} |"
        for power in POWERS:
            try:
                pca = inf.pca_fit(fit, device, dim=dim, power=power)
                results[(dim, power)] = score(db, queries, gt, device, pca)
                row += f"{results[(dim, power)][1]:9.4f}"
                del pca
            except torch.OutOfMemoryError:
                row += f"{'oom':>9}"
            torch.cuda.empty_cache()
        print(row)

    shipped = results.get((inf.PCA_DIM, inf.PCA_POWER))
    if shipped is None:
        print(f"\n  shipped cell ({inf.PCA_DIM}, {inf.PCA_POWER}) did not score — "
              "nothing below is calibrated")
    else:
        print(f"\n  reproduction check: ({inf.PCA_DIM}, {inf.PCA_POWER}) -> R@1 {shipped[1]!r}")
        print("     must equal results_megaevent.json's recall.pca['1'] exactly.")
        print("\n  cells beating the shipped R@1, shown at every cutoff:")
        for (dim, power), r in sorted(results.items(), key=lambda kv: -kv[1][1]):
            if r[1] <= shipped[1]:
                continue
            dominates = all(r[k] >= shipped[k] for k in (1, 5, 10, 20))
            print(f"    dim {dim:>5} power {power:<5} : "
                  + "  ".join(f"{r[k]:.4f}" for k in (1, 5, 10, 20))
                  + ("   <- dominates on every cutoff" if dominates else "   (loses at some k)"))

    best = max(results, key=lambda k: results[k][1])
    print(f"\n  fit-subsample size at dim {best[0]} power {best[1]} "
          f"(is PCA_FIT_SAMPLES={scoring.PCA_FIT_SAMPLES} costing anything?):")
    for n in (scoring.PCA_FIT_SAMPLES, 2 * scoring.PCA_FIT_SAMPLES, db.size(0)):
        try:
            pca = inf.pca_fit(subsample(db, n), device, dim=best[0], power=best[1])
            r = score(db, queries, gt, device, pca)
            print(f"    {n:>6} rows: " + "  ".join(f"R@{k} {r[k]:.4f}" for k in (1, 5, 10)))
            del pca
        except torch.OutOfMemoryError:
            print(f"    {n:>6} rows: OOM (needs a bigger card or a chunked SVD)")
        torch.cuda.empty_cache()
    print(f"\n  NB {gt.shape[1]} queries -> one query = {100 / gt.shape[1]:.3f} pt of R@1. "
          "Read every delta above\n     against that before believing it.")


def positions(q_paths):
    """(position-within-run 0..8, location id) for each query."""
    utm = utm_from_paths(q_paths)
    _, location = np.unique(utm, axis=0, return_inverse=True)
    stem = np.array([int(os.path.basename(p).split("@")[7]) for p in q_paths])
    pos = np.empty(len(q_paths), int)
    for group in np.unique(location):
        idx = np.where(location == group)[0]
        pos[idx] = np.argsort(np.argsort(stem[idx]))
    return pos, location


def top1_hits(sim_path, gt):
    sim = np.load(sim_path)
    return gt[sim.argmax(axis=0), np.arange(sim.shape[1])]


def run_structure(gt, q_paths, args):
    print("== query structure: 3 times of day x 3 camera directions per location ==")
    pos, location = positions(q_paths)
    counts = np.bincount(location)
    contiguous = 0
    stem = np.array([int(os.path.basename(p).split("@")[7]) for p in q_paths])
    for group in np.unique(location):
        runs = sorted(stem[location == group])
        contiguous += runs == list(range(runs[0], runs[0] + len(runs)))
    print(f"  {len(q_paths)} queries -> {counts.size} locations, "
          f"{sorted(set(counts.tolist()))} images each, "
          f"{contiguous}/{counts.size} with a contiguous stem run")

    events = np.array([np.load(p)["x"].shape[0] for p in q_paths], float)
    print("\n  I2E event count as a brightness proxy for the time-of-day axis:")
    for label, group in (("pos//3 (time-major?)", pos // 3), ("pos%3  (dir-major?) ", pos % 3)):
        buckets = [events[group == k] for k in range(3)]
        ratio = np.var([b.mean() for b in buckets]) / np.mean([b.var() for b in buckets])
        print(f"    {label}: means {[f'{b.mean() / 1e3:.0f}k' for b in buckets]}  "
              f"between/within var = {ratio:.3f}")
    print("    -> no signal either way. I2E differences *log* luma and countmask divides by a\n"
          "       per-frame percentile alpha, so absolute brightness is normalised away. The axis\n"
          "       assignment needs the raw 247query_v2 timestamps.")

    print("\n  R@1 by position (same GT within a location, so this is pure query difficulty):")
    for name, path in (("native", f"{args.feature_dir}/sim_database_queries.npy"),
                       ("pca   ", f"{args.feature_dir}/sim_database_queries_pca.npy")):
        if not os.path.exists(path):
            print(f"    {name}: {path} missing — run the Tokyo evaluation first")
            continue
        hit = top1_hits(path, gt)
        print(f"    {name} " + " ".join(f"p{p}={hit[pos == p].mean():.3f}" for p in range(9)))
        print(f"           pos//3 -> "
              + "  ".join(f"{hit[pos // 3 == k].mean():.3f}" for k in range(3))
              + "   |  pos%3 -> "
              + "  ".join(f"{hit[pos % 3 == k].mean():.3f}" for k in range(3)))


def run_orientation(gt, q_paths, args):
    print("== portrait vs landscape queries (the database is 100% landscape) ==")
    orientation = np.empty(len(q_paths), dtype=object)
    for i, path in enumerate(q_paths):
        with np.load(path) as npz:
            height, width = (int(v) for v in npz["resolution"])
        orientation[i] = "landscape" if width >= height else "portrait"
    for name, path in (("native", f"{args.feature_dir}/sim_database_queries.npy"),
                       ("pca   ", f"{args.feature_dir}/sim_database_queries_pca.npy")):
        if not os.path.exists(path):
            print(f"  {name}: {path} missing — run the Tokyo evaluation first")
            continue
        hit = top1_hits(path, gt)
        line = f"  {name}"
        for kind in ("landscape", "portrait"):
            mask = orientation == kind
            line += f"   {kind} n={int(mask.sum()):3d} R@1={hit[mask].mean():.4f}"
        print(line + f"   overall {hit.mean():.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--which", default="all", choices=["all", "pca", "structure", "orientation"])
    ap.add_argument("--feature-dir", default=DEFAULT_FEATURES)
    ap.add_argument("--npz-root", default=DEFAULT_NPZ)
    ap.add_argument("--threshold-m", type=float, default=25.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    db, queries, gt, q_paths = load(args)
    for name, fn in (("pca", lambda: run_pca(db, queries, gt, device)),
                     ("structure", lambda: run_structure(gt, q_paths, args)),
                     ("orientation", lambda: run_orientation(gt, q_paths, args))):
        if args.which in ("all", name):
            fn()
            print()


if __name__ == "__main__":
    main()
