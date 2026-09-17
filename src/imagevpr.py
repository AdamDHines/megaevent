"""Image-set VPR: independent event streams scored against per-image ground truth.

    <split>/*.npz  --eventcv--> countmask [3,H,W] uint8
      --normalise + resize--> ViT-S/14 + SALAD --> L2-normalised descriptor
    cosine similarity against the database bank --> R@k over dataset positives

The counterpart of :mod:`src.inference`, which handles *continuous traverses*: one long
recording sliced into ``dt_ms`` windows, scored against an Event-LAB ground-truth band.
Here each file is one independent image with no temporal ordering. Tokyo/NYC derive
positives from a metric radius, Pitts supplies an explicit positive list, and MSLS uses
its official per-city 25 m protocol. Everything downstream of loading (model,
descriptors, PCA, metric) is imported from :mod:`src.inference` rather than reimplemented.

Only the *geographic* half lives here: building the ground truth from UTM coordinates, the
error map that plots it, and the smoke-test subset that picks database images by distance.
The rest — descriptor caching, whitening, the recall curve, the native/whitened/re-ranked
sweep, the montage — is in :mod:`src.scoring`, shared with :mod:`src.traversevpr` so both
evaluations report every method in the same spaces under the same metric.

The ``.npz`` files are I2E's *raw event streams* (``x, y, t, p, resolution``), not
pre-rendered frames — the representation is chosen here, at load time, so the same data
can feed a model trained on any of them. Two things about them are easy to get wrong:

* ``resolution`` is stored ``[H, W]``, while eventcv's ``sensor_size`` is ``(W, H)``;
* the streams are ~30 ms long, well under the ~1 s that eventcv's time-unit
  auto-detection assumes, so ``time_unit`` must be passed explicitly.

Ground truth is carried in the filenames, not a sidecar file: I2E's Tokyo 24/7 converter
writes ``@easting@northing@zone@band@lat@lon@...@.npz``, so the UTM coordinate is field 1
and 2 of an ``@``-split basename.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")                       # headless: we only ever write PNGs

import numpy as np
import torch
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.inference import C_FP, C_MUTED, C_TEXT, C_TP, KS
from src.imagesets import build_radius_gt, load_image_set
from src.methods import get_method
from src.npzdata import load_countmask  # noqa: F401
# The dataset-agnostic half of this evaluation, shared with src.traversevpr. Imported
# under the private names it used to define so that importers predating the split
# (src.eventgemlocal, scripts/verify_*.py) keep working unchanged.
from src.scoring import (                                                  # noqa: F401
    MAX_N, MONTAGE_QUERIES, MONTAGE_TOPK, PCA_FIT_SAMPLES, PCA_FIT_SEED,
    cache_paths as _cache_paths,
    cached_array,
    check_manifest as _check_manifest,
    cross_check as _cross_check,
    display_frame as _display,
    evaluate,
    figure_recall_curve,
    figure_retrievals,
    pca_fit_subsampled,
    recall_curve,
    score_both as _score_both,
    seed_summary as _seed_summary,
    style as _style,
)

# ---------------------------------------------------------------------------
# 1. Ground truth (geographic: the part that is specific to an image set)
# ---------------------------------------------------------------------------
def build_gt(db_utm, q_utm, threshold_m):
    """Bool ``[n_db, n_q]``: rows = database, columns = query, True within ``threshold_m``.

    The ``[reference, query]`` orientation :func:`src.inference.recall_at_k` expects, so
    the metric is reused verbatim. Matches ``VPR-methods-evaluation``'s
    ``TestDataset``: positives are every database image inside a metric radius of the
    query, found with a KD-tree rather than a dense distance matrix.
    """
    return build_radius_gt(db_utm, q_utm, threshold_m)


# ---------------------------------------------------------------------------
# 2. Figures that need the geography
# ---------------------------------------------------------------------------
def figure_error_map(db_utm, q_utm, ranked, gt, scorable, threshold_m, out_png):
    """Where the top-1 failures are, geographically.

    Aggregated per *unique* query coordinate rather than plotted per query: Tokyo 24/7
    shoots several images from each spot, so a per-query scatter draws every marker of a
    location on top of the last one and a spot that failed once looks identical to a spot
    that failed every time. Colour is the location's top-1 success rate, area its query
    count, so both survive the overplotting.
    """
    q_idx = np.flatnonzero(scorable)                    # see figure_retrievals
    correct = gt[ranked[0], q_idx]

    coords, inverse = np.unique(q_utm[q_idx], axis=0, return_inverse=True)
    total = np.bincount(inverse, minlength=len(coords))
    hits = np.bincount(inverse, weights=correct.astype(float), minlength=len(coords))
    rate = hits / total

    fig, ax = plt.subplots(figsize=(6.8, 6.0), constrained_layout=True)
    ax.scatter(db_utm[:, 0], db_utm[:, 1], s=3, c="#dcdbd6", marker=".", linewidths=0,
               label=f"database ({len(db_utm)})", zorder=1)
    cmap = LinearSegmentedColormap.from_list("rate", [C_FP, "#f2d9a8", C_TP])
    dots = ax.scatter(coords[:, 0], coords[:, 1], c=rate, cmap=cmap, vmin=0, vmax=1,
                      s=24 + 12 * total, edgecolors="#ffffff", linewidths=0.6, zorder=2)
    cb = fig.colorbar(dots, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("top-1 success rate at this location", color=C_MUTED, fontsize=9)
    cb.ax.tick_params(colors=C_MUTED, labelsize=8)

    ax.set_aspect("equal")
    # Plain metres: a 3.9e6 northing rendered as an offset puts a stray "1e6" under the
    # title, and the absolute coordinate is the useful thing to read off anyway.
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.set_xlabel("UTM easting (m)", color=C_MUTED, fontsize=9)
    ax.set_ylabel("UTM northing (m)", color=C_MUTED, fontsize=9)
    ax.set_title(f"top-1 retrieval by query location ({threshold_m:g} m radius)",
                 fontsize=10, color=C_TEXT, loc="left", pad=16)
    ax.text(0, 1.015, f"{len(coords)} distinct locations, {len(q_idx)} scorable queries; "
                      f"marker area = queries at that spot",
            transform=ax.transAxes, fontsize=8.5, color=C_MUTED, va="bottom")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    _style(ax)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Entry point
# ---------------------------------------------------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    method = get_method(args.method, args, device)

    dataset = load_image_set(args)
    db_paths, q_paths, gt = dataset.db_paths, dataset.q_paths, dataset.gt
    logger.info(f"{args.dataset}: {len(db_paths)} {args.ref} x {len(q_paths)} {args.query}")

    per_query = gt.sum(0)
    scorable = int((per_query > 0).sum())
    logger.info(f"ground truth @ {args.positive_dist_threshold:g} m: {scorable}/{len(q_paths)} "
                f"queries scorable, positives per query mean {per_query.mean():.1f} "
                f"min {per_query.min()} max {per_query.max()}")
    if scorable == 0:
        raise RuntimeError(
            f"no query has a database image within {args.positive_dist_threshold:g} m — "
            f"recall is undefined. Check that both splits carry UTM coordinates from the "
            f"same zone.")

    out_dir = os.path.join(args.feature_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_limit{args.limit}" if getattr(args, "limit", None) else ""

    results = {"dataset": args.dataset, "method": method.name,
               "native_metric": method.native_metric, "database": args.ref,
               "queries": args.query, "n_database": len(db_paths),
               "n_queries": len(q_paths), "threshold_m": args.positive_dist_threshold,
               "excluded_database": dataset.excluded_db,
               "excluded_queries": dataset.excluded_q,
               "scorable_queries": scorable,
               "positives_per_query_mean": float(per_query.mean()),
               "recall": {}, "recall_curve": {}, **{"meta": method.meta}}
    curves = {}

    if getattr(method, "multi_seed", False):
        # sparse_event picks its readout pixels at random from a saliency map, and
        # Event-LAB's own results show that choice moving R@1 by as much as 0.20 between
        # runs. A single seed would not be a measurement, so every seed is scored and the
        # spread is reported alongside the mean.
        db_frames = cached_array(args, method, args.ref, db_paths, "frames",
                                 method.frames, "count frames", mmap=True)
        q_frames = cached_array(args, method, args.query, q_paths, "frames",
                                method.frames, "count frames", mmap=True)
        per_seed, keep = [], None
        for seed in args.seeds:
            method.fit_pixels(db_frames, seed)
            db_desc, q_desc = method.encode(db_frames), method.encode(q_frames)
            seed_res = {"recall": {}, "recall_curve": {}}
            native, pca_out = _score_both(method, db_desc, q_desc, gt, device,
                                          seed_res, {}, label=f"seed{seed}_",
                                          skip_pca=getattr(args, "no_pca", False))
            per_seed.append({t.replace(f"seed{seed}_", ""): v
                             for t, v in seed_res["recall"].items()})
            results["recall_curve"][f"seed{seed}"] = {
                t.replace(f"seed{seed}_", ""): c
                for t, c in seed_res["recall_curve"].items()}
            if keep is None:            # figures describe the first seed
                keep = pca_out
                for space in ("native", "pca"):
                    # --no-pca leaves only the native curve to describe.
                    curve = seed_res["recall_curve"].get(f"seed{seed}_{space}")
                    if curve is not None:
                        curves[space] = {int(n): v for n, v in curve.items()}
        results["recall"] = _seed_summary(per_seed)
        results["seeds"] = list(args.seeds)
        for tag in [t for t in ("native", "pca") if t in results["recall"]]:
            m = results["recall"][tag]
            logger.info(f"[{method.name} {tag}] " + "  ".join(
                f"R@{k}={m[str(k)]['mean']:.3f}+-{m[str(k)]['std']:.3f}" for k in KS)
                + f"   (over {len(per_seed)} seeds)")
        rec, curve, ranked, mask, sim = keep
    else:
        db_desc = torch.from_numpy(np.asarray(cached_array(
            args, method, args.ref, db_paths, "",
            lambda p, s: method.descriptors(p, s), "descriptors")))
        q_desc = torch.from_numpy(np.asarray(cached_array(
            args, method, args.query, q_paths, "",
            lambda p, s: method.descriptors(p, s), "descriptors")))
        if getattr(args, "banks_only", False):
            # Both banks and their manifests are on disk now, which is all
            # scripts/score_cached_banks.py needs. Stop before the dense matrix.
            logger.info(f"--banks-only: cached {tuple(db_desc.shape)} {args.ref} and "
                        f"{tuple(q_desc.shape)} {args.query} descriptors under "
                        f"'{method.tag}' -> {out_dir}; scoring skipped")
            return
        native, pca_out = _score_both(
            method, db_desc, q_desc, gt, device, results, curves,
            paths=(db_paths, q_paths), map_ks=KS if args.dataset == "msls" else (),
            skip_pca=getattr(args, "no_pca", False))
        np.save(os.path.join(out_dir, f"sim_{method.tag}_{args.ref}_{args.query}{suffix}.npy"),
                native[4])
        rec, curve, ranked, mask, sim = pca_out
    del sim

    artifact_tag = method.tag
    if args.dataset == "msls":
        ranking_space = next(reversed(curves))
        results["ranking_space"] = ranking_space
        ranking_path = os.path.join(out_dir,
                                    f"predictions_{artifact_tag}_{ranking_space}{suffix}.txt")
        q_indexes = np.flatnonzero(mask)
        with open(ranking_path, "w") as handle:
            for column, query in enumerate(q_indexes):
                keys = [dataset.db_keys[index] for index in ranked[:, column]]
                handle.write(" ".join([dataset.q_keys[query], *keys]) + "\n")
    with open(os.path.join(out_dir, f"results_{artifact_tag}{suffix}.json"), "w") as f:
        json.dump(results, f, indent=2)

    figure_recall_curve(
        curves,
        f"{args.dataset} / {method.name}: {len(db_paths)} database x {scorable} scorable "
        f"queries, {args.positive_dist_threshold:g} m",
        os.path.join(out_dir, f"recall_curve_{artifact_tag}{suffix}.png"))
    # The montage and the map describe the PCA-whitened space, the stronger of the two,
    # so the failures they show are the ones actually worth looking at.
    figure_retrievals(db_paths, q_paths, ranked, gt, mask,
                      os.path.join(out_dir, f"retrievals_{artifact_tag}{suffix}.png"))
    if dataset.db_utm is not None:
        figure_error_map(dataset.db_utm, dataset.q_utm, ranked, gt, mask,
                         args.positive_dist_threshold,
                         os.path.join(out_dir, f"error_map_{artifact_tag}{suffix}.png"))

    logger.info(f"artifacts -> {out_dir}")
    return results
