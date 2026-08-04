"""Traverse VPR from materialised ``.npz`` slices, with the event source as a variable.

    <npz-root>/<dataset>/<source>/<seq>/frame_%06d.npz
      --eventcv--> the representation each method wants
      --> descriptors --> R@k over the Event-LAB ground-truth band

The third evaluation path in this repo, and the reason it exists is an ablation rather than
a dataset. :mod:`src.inference` scores a traverse straight from its HDF5 event stream, and
:mod:`src.imagevpr` scores a folder of independent images against a geographic radius. This
module scores a *traverse* whose frames come from a folder — which lets the same traverse be
evaluated twice, from two different event sources, under one protocol:

===========  ==========================================  ==============================
``--source`` frames                                      built by
===========  ==========================================  ==============================
``real``     the recording's own 50 ms event slices       ``scripts/dump_event_npz.py``
``i2e``      an I2E micro-saccade over the DAVIS APS      ``scripts/extract_aps.py`` then
             frame captured at the same instant           I2E's ``i2e_infer.py``
===========  ==========================================  ==============================

That answers the standing objection to this repo's Tokyo 24/7 result — that megaevent was
fine-tuned on I2E events and so a benchmark of I2E events flatters it relative to baselines
trained on real ones. On Brisbane-Event-VPR the DAVIS346 recorded both modalities through
one sensor on one clock, so the two arms differ *only* in how the events were produced. If
the baselines score the same on both, the synthetic domain is not what holds them back.

Three things make the arms comparable rather than merely similar:

* **Index alignment.** ``scripts/extract_aps.py`` picks, for each event slice, the APS frame
  nearest that slice's centre, so frame *i* means the same instant in both arms and both
  have exactly ``n_slices`` frames.
* **One ground truth.** Both arms load the same Event-LAB band through the same
  :func:`src.inference.load_gt` resample. Nothing about the ground truth differs.
* **A shared validity mask.** A handful of slices (~0.3%) have no APS frame within half a
  slice, because the DAVIS dropped intensity frames. Those indices are dropped from *both*
  arms, so the real arm is never scored on frames the synthetic arm had to fake.

Scoring, caching, whitening and the recall curve all come from :mod:`src.scoring`, shared
with :mod:`src.imagevpr` — so a method's number here is produced by the same code as its
number on Tokyo 24/7.
"""

import glob
import json
import os

import numpy as np
import torch
from loguru import logger

from src.inference import (
    KS, ground_truth_path, load_gt, make_figure, sim_matrix,
)
from src.methods import get_method
from src.scoring import (
    cached_array, evaluate, figure_recall_curve, figure_retrievals, score_both, seed_summary,
)

SOURCES = ("real", "i2e", "real_masked", "i2e_masked")
SOURCE_BLURB = {
    "real": "real events",
    "i2e": "I2E events from the DAVIS frames",
    "real_masked": "real events, vignette masked",
    "i2e_masked": "I2E events from the DAVIS frames, vignette masked",
}


# ---------------------------------------------------------------------------
# 1. Layout
# ---------------------------------------------------------------------------
def source_dir(args, seq):
    """``<traverse-npz-root>/<dataset>/<source>/<seq>`` — one arm of one traverse."""
    path = os.path.join(args.traverse_npz_root, args.dataset, args.source, seq)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"no '{args.source}' frames for {seq} at {path}. Build them with "
            f"scripts/dump_event_npz.py (real) or scripts/extract_aps.py + I2E's "
            f"i2e_infer.py (i2e).")
    return path


def list_frames(directory):
    """Every ``frame_*.npz``, sorted — and the sort *is* the slice order.

    Six-digit zero padding makes the lexicographic sort numeric, so row *i* of a descriptor
    bank is slice *i*, which is what lets the Event-LAB band index it directly.
    """
    paths = sorted(glob.glob(os.path.join(directory, "frame_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no frame_*.npz in {directory}")
    return paths


def validity_mask(args, seq, n):
    """``[n]`` bool — False where this slice has no APS frame within half a slice.

    Read from the ``select.json`` that ``scripts/extract_aps.py`` writes, and applied
    whichever arm is being scored. That is deliberate: the mask describes the *alignment*,
    not the event source, and applying it to only the synthetic arm would compare a real
    frame against a duplicated one and call the difference a domain gap.
    """
    path = os.path.join(args.traverse_npz_root, args.dataset, "aps", seq, "select.json")
    keep = np.ones(n, dtype=bool)
    if not os.path.exists(path):
        logger.warning(f"{path}: no APS selection report — scoring all {n} frames of "
                       f"{seq} unmasked. The two arms are only comparable if this file is "
                       f"present for both.")
        return keep
    with open(path) as f:
        report = json.load(f)
    if int(report.get("n_slices", n)) != n:
        raise ValueError(
            f"{path} was written for {report.get('n_slices')} slices but {seq} has {n} "
            f"frames on disk — the arms would be scored on different grids")
    bad = np.asarray(report.get("out_of_tolerance", []), dtype=int)
    keep[bad[bad < n]] = False
    return keep


def _check_arms_agree(args, seq, paths):
    """Every other arm must hold the same frames under the same names, or the ablation is void.

    Cheap, and it catches the one failure that would otherwise produce plausible numbers:
    an interrupted or resumed conversion leaving one arm a few frames short, after which
    :func:`src.inference.load_gt` resamples the band onto a *different* grid and every
    recall is quietly wrong.
    """
    ours = [os.path.basename(p) for p in paths]
    for other in SOURCES:
        if other == args.source:
            continue
        path = os.path.join(args.traverse_npz_root, args.dataset, other, seq)
        if not os.path.isdir(path):
            continue
        theirs = sorted(os.path.basename(p)
                        for p in glob.glob(os.path.join(path, "frame_*.npz")))
        if theirs != ours:
            raise ValueError(
                f"{seq}: the '{args.source}' arm has {len(ours)} frames but the '{other}' "
                f"arm has {len(theirs)} — every arm must hold identical frame lists for "
                f"the comparison to mean anything. Rebuild the short one.")


# ---------------------------------------------------------------------------
# 2. Entry point
# ---------------------------------------------------------------------------
def _stride(paths, limit):
    """A uniform subsample of a traverse, for smoke tests.

    A *prefix* would be wrong in a way that looks right: the ground-truth band runs down the
    diagonal, so the first N queries match the first N references and later queries would
    have no reference at all. Striding keeps the band diagonal, and :func:`load_gt` resamples
    onto the strided shape. The recall is still not comparable to a full run — a coarser grid
    is effectively a wider frame tolerance — which is why the caches are named separately.
    """
    if not limit or limit >= len(paths):
        return paths, np.arange(len(paths))
    k = max(1, int(np.ceil(len(paths) / limit)))
    idx = np.arange(0, len(paths), k)
    return [paths[i] for i in idx], idx


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    method = get_method(args.method, args, device)

    db_paths = list_frames(source_dir(args, args.ref))
    q_paths = list_frames(source_dir(args, args.query))
    _check_arms_agree(args, args.ref, db_paths)
    _check_arms_agree(args, args.query, q_paths)

    keep_db = validity_mask(args, args.ref, len(db_paths))
    keep_q = validity_mask(args, args.query, len(q_paths))
    logger.info(f"{args.dataset}/{args.source}: {len(db_paths)} {args.ref} x "
                f"{len(q_paths)} {args.query}; APS alignment drops "
                f"{(~keep_db).sum()} reference and {(~keep_q).sum()} query frames")

    if getattr(args, "limit", None):
        logger.warning(f"--limit {args.limit}: uniform stride over both traverses "
                       f"(smoke test only — the recall is not comparable to a full run)")
        db_paths, db_idx = _stride(db_paths, args.limit)
        q_paths, q_idx = _stride(q_paths, args.limit)
        keep_db, keep_q = keep_db[db_idx], keep_q[q_idx]

    # Ground truth on the full grid, then masked in step with the descriptors. Building it
    # at the masked size instead would resample the band onto a grid with holes in it.
    gt_full, gt_shape = load_gt(ground_truth_path(args), len(db_paths), len(q_paths))
    gt = gt_full[np.ix_(keep_db, keep_q)]
    scorable = int((gt.sum(0) > 0).sum())
    logger.info(f"ground truth {gt_shape[0]}x{gt_shape[1]} -> {gt.shape[0]}x{gt.shape[1]} "
                f"[ref, query], density {gt.mean():.4f}, {scorable}/{gt.shape[1]} "
                f"queries scorable")
    if scorable == 0:
        raise RuntimeError("no query has a ground-truth reference — check the band and "
                           "the frame counts")

    out_dir = os.path.join(args.feature_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{args.source}" + (f"_limit{args.limit}" if getattr(args, "limit", None) else "")

    results = {"dataset": args.dataset, "source": args.source, "method": method.name,
               "native_metric": method.native_metric, "database": args.ref,
               "queries": args.query, "dt_ms": args.dt_ms,
               "n_database": int(keep_db.sum()), "n_queries": int(keep_q.sum()),
               "n_database_raw": len(db_paths), "n_queries_raw": len(q_paths),
               "dropped_database": int((~keep_db).sum()),
               "dropped_queries": int((~keep_q).sum()),
               "gt_shape": list(gt_shape), "gt_density": float(gt.mean()),
               "scorable_queries": scorable,
               "recall": {}, "recall_curve": {}, "meta": method.meta}
    curves = {}

    # The cache key has to carry the arm. cache_paths keys on <method.tag>_<split>, and
    # check_manifest compares basenames — which are identical between the two arms by
    # construction, so without this the second arm would silently reuse the first's bank.
    db_split, q_split = f"{args.ref}_{args.source}", f"{args.query}_{args.source}"
    # Paths for the re-ranker and the montage, masked in step with the descriptors.
    db_kept = [p for p, k in zip(db_paths, keep_db) if k]
    q_kept = [p for p, k in zip(q_paths, keep_q) if k]

    if getattr(method, "multi_seed", False):
        # sparse_event picks its readout pixels at random from a saliency map, and
        # Event-LAB's own results show that choice moving R@1 by as much as 0.20 between
        # runs. A single seed would not be a measurement, so every seed is scored and the
        # spread is reported alongside the mean.
        db_frames = cached_array(args, method, db_split, db_paths, "frames",
                                 method.frames, "count frames", mmap=True)
        q_frames = cached_array(args, method, q_split, q_paths, "frames",
                                method.frames, "count frames", mmap=True)
        per_seed, keep = [], None
        for seed in args.seeds:
            # Pixels are chosen from the full reference saliency map, before masking: the
            # dropped frames are valid event data, they simply have no APS counterpart.
            method.fit_pixels(db_frames, seed)
            db_desc = method.encode(db_frames)[torch.from_numpy(keep_db)]
            q_desc = method.encode(q_frames)[torch.from_numpy(keep_q)]
            seed_res = {"recall": {}, "recall_curve": {}}
            _, pca_out = score_both(method, db_desc, q_desc, gt, device, seed_res, {},
                                    label=f"seed{seed}_")
            per_seed.append({t.replace(f"seed{seed}_", ""): v
                             for t, v in seed_res["recall"].items()})
            results["recall_curve"][f"seed{seed}"] = {
                t.replace(f"seed{seed}_", ""): c
                for t, c in seed_res["recall_curve"].items()}
            if keep is None:            # figures describe the first seed
                keep = pca_out
                for tag in ("native", "pca"):
                    curves[tag] = {int(n): v for n, v in
                                   seed_res["recall_curve"][f"seed{seed}_{tag}"].items()}
        results["recall"] = seed_summary(per_seed)
        results["seeds"] = list(args.seeds)
        for tag in ("native", "pca"):
            m = results["recall"][tag]
            logger.info(f"[{method.name} {tag}] " + "  ".join(
                f"R@{k}={m[str(k)]['mean']:.3f}+-{m[str(k)]['std']:.3f}" for k in KS)
                + f"   (over {len(per_seed)} seeds)")
        rec, curve, ranked, mask, sim = keep
        native_sim = None
    else:
        db_desc = torch.from_numpy(np.asarray(cached_array(
            args, method, db_split, db_paths, "",
            lambda p, s: method.descriptors(p, s), "descriptors")))[torch.from_numpy(keep_db)]
        q_desc = torch.from_numpy(np.asarray(cached_array(
            args, method, q_split, q_paths, "",
            lambda p, s: method.descriptors(p, s), "descriptors")))[torch.from_numpy(keep_q)]
        native, pca_out = score_both(method, db_desc, q_desc, gt, device, results, curves,
                                     paths=(db_kept, q_kept))
        native_sim = native[4]
        rec, curve, ranked, mask, sim = pca_out

    with open(os.path.join(out_dir, f"results_{method.name}{suffix}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Figures. figure_error_map has no traverse analogue — it plots per-image UTM, which a
    # traverse has none of — so make_figure stands in: the similarity matrix with the band
    # contoured, and top-1 retrievals scored against it.
    if native_sim is not None:
        np.save(os.path.join(out_dir, f"sim_{method.tag}_{args.ref}_{args.query}{suffix}.npy"),
                native_sim)
    # The matrix plotted is the strongest space score_both scored — the last whitening, and
    # its re-ranked twin for a method that re-ranks — matching src.imagevpr's convention of
    # describing the space whose failures are actually worth looking at.
    reranked = hasattr(method, "rerank") and native_sim is not None
    space = "PCA-whitened cosine" + (", homography re-ranked" if reranked else "")
    top1 = sim.argmax(axis=0)
    tp = gt[top1, np.arange(gt.shape[1])]
    make_figure(sim, gt, top1, tp, rec, args.ref, args.query,
                f"{method.name} on {SOURCE_BLURB[args.source]} — {space}",
                os.path.join(out_dir, f"fig_{method.name}_{args.ref}_{args.query}{suffix}.png"),
                gt_shape, metric_label=space)
    figure_recall_curve(
        curves,
        f"{args.dataset} / {method.name} / {args.source}: {gt.shape[0]} reference x "
        f"{scorable} scorable queries @ {args.dt_ms} ms",
        os.path.join(out_dir, f"recall_curve_{method.name}{suffix}.png"))
    figure_retrievals(db_kept, q_kept, ranked, gt, mask,
                      os.path.join(out_dir, f"retrievals_{method.name}{suffix}.png"))
    del sim

    logger.info(f"artifacts -> {out_dir}")
    return results
