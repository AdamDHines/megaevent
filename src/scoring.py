"""Descriptor caching, scoring and the figures that go with them — shared by both evaluations.

:mod:`src.imagevpr` scores folders of independent images against a geographic radius;
:mod:`src.traversevpr` scores continuous traverses against an Event-LAB ground-truth band.
Everything between "here is a list of ``.npz`` paths" and "here is R@k" is the same for
both, and lives here: the per-split array cache, PCA whitening, the recall curve, the
native/whitened/re-ranked scoring sweep, and the two figures that need no geography.

The split is by *what a function knows about the ground truth*. Nothing in this module
knows where ``gt`` came from — it is a bool ``[n_db, n_q]`` with rows = reference and
columns = query, and that is all. Anything that reads a UTM coordinate or an Event-LAB band
stays in the evaluation module that owns it.

One consequence worth stating, because it is the reason the whole comparison holds together:
because both evaluations call the same :func:`score_both`, every method is reported in the
same descriptor spaces under the same metric on both datasets and in both arms of the
real-vs-I2E ablation. A difference in a number is a difference in the data.

:mod:`src.imagevpr` re-exports every name here under its former private alias, so importers
that predate the split (``src.eventgemlocal``, ``scripts/verify_*.py``) keep working.
"""

import os
import time

import matplotlib
matplotlib.use("Agg")                       # headless: we only ever write PNGs

import numpy as np
import torch
from loguru import logger
from matplotlib import pyplot as plt

from src.inference import (
    C_FP, C_MUTED, C_TEXT, C_TP, KS, PCA_POWER, pca_apply, pca_fit, recall_at_k, sim_matrix,
)
from src.npzdata import load_accumulate, load_countmask

# ---------------------------------------------------------------------------
# Tunables that main.py does not expose
# ---------------------------------------------------------------------------
MAX_N = 20                      # longest N plotted on the recall curve
# pca_fit's randomized SVD needs several GB of workspace on top of the bank itself, and a
# full Tokyo database is 75984 x 8448 fp32 = 2.6 GB — enough to OOM an 8 GB card. A seeded
# 20k-row subsample is far more than a 2048-d whitening basis needs (the estimator is over
# 9x oversampled even at the largest dim) and makes the fit fit in ~0.7 GB. A brisbane
# traverse is smaller than this, so it takes the exact-fit branch — which is what keeps the
# traverse numbers comparable to src.inference's own PCA path.
PCA_FIT_SAMPLES = 20_000
PCA_FIT_SEED = 0
MONTAGE_QUERIES, MONTAGE_TOPK = 6, 5


# ---------------------------------------------------------------------------
# 1. Cached per-split arrays
# ---------------------------------------------------------------------------
def cache_paths(args, method, split, kind=""):
    """``(array.npy, manifest.txt)`` for one split of one method.

    Mirrors :func:`src.inference._cache_path` but drops its ``dt``/``hp``/``ba`` tags —
    there is no slicing window and no stream filtering when each file is already one
    image — and carries the method tag instead, so all the methods coexist in one feature
    directory.

    ``split`` is the cache key's only handle on *which data this is*, so a caller scoring
    the same traverse from two different event sources must fold that into ``split``
    (:mod:`src.traversevpr` passes ``<seq>_<source>``). :func:`check_manifest` compares
    basenames, and the two arms of the ablation use identical basenames by construction, so
    it cannot catch that confusion on its own.
    """
    stem = f"{method.tag}_{split}"
    if kind:
        stem += f"_{kind}"
    if getattr(args, "limit", None):
        stem += f"_limit{args.limit}"
    out = os.path.join(args.feature_dir, args.dataset)
    return os.path.join(out, f"{stem}.npy"), os.path.join(out, f"{stem}_paths.txt")


def check_manifest(manifest_path, paths):
    """Guard a descriptor cache against the file list having changed underneath it.

    Row *i* of a cached bank means nothing without the ordering it was built from — a file
    added, removed or renamed would shift every descriptor away from its ground-truth
    coordinate and quietly produce a plausible-looking but wrong recall. Cheaper to
    re-extract than to publish that.
    """
    if not os.path.exists(manifest_path):
        return False
    with open(manifest_path) as f:
        cached = f.read().split("\n")
    cached = [line for line in cached if line]
    current = [os.path.basename(p) for p in paths]
    if cached == current:
        return True
    logger.warning(f"{manifest_path}: file list changed ({len(cached)} cached vs "
                   f"{len(current)} on disk) — ignoring the stale descriptor cache")
    return False


def cached_array(args, method, split, paths, kind, build, label, mmap=False):
    """Build (or reload) one per-split array, keyed on the file list it was built from.

    ``kind=""`` holds descriptors; ``kind="frames"`` holds sparse_event's intermediate
    count frames, which are cached because its pixel selection is stochastic and has to
    be repeated over several seeds — re-rendering 76k frames per seed would dominate the
    run, whereas re-reading them costs seconds.
    """
    arr_path, manifest_path = cache_paths(args, method, split, kind)
    if os.path.exists(arr_path) and check_manifest(manifest_path, paths):
        arr = np.load(arr_path, mmap_mode="r" if mmap else None)
        if arr.shape[0] == len(paths):
            logger.info(f"{split}: loaded {tuple(arr.shape)} {label} from {arr_path}")
            return arr
        logger.warning(f"{arr_path}: holds {arr.shape[0]} rows for {len(paths)} files "
                       f"— rebuilding")

    logger.info(f"{split}: {len(paths)} images, building {label}...")
    t0 = time.time()
    arr = build(paths, split)
    if torch.is_tensor(arr):
        arr = arr.numpy()
    logger.info(f"{split}: {tuple(arr.shape)} {label} in {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(arr_path), exist_ok=True)
    np.save(arr_path, arr)
    with open(manifest_path, "w") as f:
        f.write("\n".join(os.path.basename(p) for p in paths))
    logger.info(f"{split}: cached -> {arr_path} "
                f"({os.path.getsize(arr_path) / 1e6:.0f} MB)")
    return np.load(arr_path, mmap_mode="r") if mmap else arr


def pca_fit_subsampled(db_desc, device, n=PCA_FIT_SAMPLES, seed=PCA_FIT_SEED,
                       power=PCA_POWER):
    """:func:`src.inference.pca_fit` on a seeded subsample of the database bank.

    Whitening only needs the reference set's second-order statistics, and 20k rows
    estimate those as well as 76k for a 2048-d basis — while keeping the randomized SVD's
    working set small enough to stay on an 8 GB GPU.
    """
    if db_desc.size(0) <= n:
        return pca_fit(db_desc, device, power=power)
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(db_desc.size(0), generator=generator)[:n]
    logger.info(f"PCA: fitting on {n} of {db_desc.size(0)} database descriptors "
                f"(seed {seed}, power {power:g})")
    return pca_fit(db_desc[idx], device, power=power)


# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------
def recall_curve(sim, gt, max_n=MAX_N):
    """``{N: recall}`` for N = 1..``max_n``, from a single top-``max_n`` ranking.

    :func:`src.inference.recall_at_k` stays the authoritative number at the reported
    cutoffs; this exists because ``recallAtK`` argsorts the whole matrix per call, and a
    20-point curve would repeat that 20 times over 75984 x 315. Partition once, sort the
    survivors, and read every N off the cumulative hit mask.

    Queries with no positive at all are dropped first, matching ``recallAtK``'s
    denominator (``metrics.py:166``) so the two are directly comparable.
    """
    scorable = gt.sum(0) > 0
    s, g = sim[:, scorable], gt[:, scorable]
    n = int(min(max_n, s.shape[0]))
    top = np.argpartition(-s, n - 1, axis=0)[:n]                    # [n, nq], unordered
    order = np.argsort(-np.take_along_axis(s, top, axis=0), axis=0)
    ranked = np.take_along_axis(top, order, axis=0)                 # [n, nq], best first
    hit = np.take_along_axis(g, ranked, axis=0)
    found = np.cumsum(hit, axis=0) > 0
    return {k: float(found[k - 1].mean()) for k in range(1, n + 1)}, ranked, scorable


def map_at_k(ranked, gt, scorable, ks=KS):
    """MSLS mean average precision at each cutoff, matching its official evaluator."""
    query_indexes = np.flatnonzero(scorable)
    out = {}
    for k in ks:
        cutoff = min(k, ranked.shape[0])
        scores = []
        for column, query in enumerate(query_indexes):
            positives = gt[:, query]
            hits = positives[ranked[:cutoff, column]].astype(np.float64)
            precision = np.cumsum(hits) / np.arange(1, cutoff + 1)
            denominator = min(int(positives.sum()), k)
            scores.append(float((precision * hits).sum() / denominator))
        out[k] = float(np.mean(scores))
    return out


def cross_check(curve, rec):
    """Warn if the fast curve and ``recallAtK`` disagree at a shared cutoff.

    They can only differ through tie-breaking between two exactly equal similarities,
    which does not happen with float32 cosines in practice — so a mismatch means a real
    bug in the ranking above. Logged rather than raised: the authoritative numbers come
    from ``recallAtK`` either way, and this is not worth discarding a completed run over.
    """
    for k in KS:
        if k in curve and not np.isclose(curve[k], rec[k], atol=1e-9):
            logger.warning(f"recall curve disagrees with recallAtK at K={k}: "
                           f"{curve[k]:.6f} vs {rec[k]:.6f}")


# ---------------------------------------------------------------------------
# 3. Figures that need no geography
# ---------------------------------------------------------------------------
def style(ax):
    ax.tick_params(colors=C_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#dcdbd6")


# Enough to draw every space a method can report: native/pca/pca1 and a +rerank twin each.
# A short zip here used to truncate the plot silently, which is exactly the kind of missing
# line nobody notices.
CURVE_COLOURS = (C_MUTED, C_TP, "#6b6a67", C_FP, "#8ba888", "#b08968")
CURVE_MARKERS = ("o", "s", "^", "D", "v", "P")


def figure_recall_curve(curves, subtitle, out_png):
    """R@N against N, one line per descriptor space."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    for i, (tag, curve) in enumerate(curves.items()):
        colour = CURVE_COLOURS[i % len(CURVE_COLOURS)]
        marker = CURVE_MARKERS[i % len(CURVE_MARKERS)]
        ns = sorted(curve)
        ax.plot(ns, [curve[n] for n in ns], color=colour, marker=marker, markersize=3.5,
                linewidth=1.4, label=f"{tag}  (R@1 = {curve[1]:.3f})")
    ax.set_xlabel("N (number of retrieved candidates)", color=C_MUTED, fontsize=9)
    ax.set_ylabel("Recall@N", color=C_MUTED, fontsize=9)
    ax.set_title("Recall@N", fontsize=11, color=C_TEXT, loc="left", pad=16)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color=C_MUTED,
            va="bottom")
    ax.set_xlim(1, max(max(c) for c in curves.values()))
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.5, color="#ececea")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    style(ax)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def display_frame(path, representation="countmask"):
    """A rendered frame as ``[H, W, 3]`` uint8, for imshow.

    ``representation`` must match what the descriptor bank being illustrated was
    extracted with — a figure drawn in the other representation would show a frame
    the model never saw.
    """
    renderers = {"countmask": load_countmask, "accumulate": load_accumulate}
    if representation not in renderers:
        raise ValueError(f"no display renderer for representation {representation!r}")
    return np.transpose(renderers[representation](path), (1, 2, 0))


def figure_retrievals(db_paths, q_paths, ranked, gt, scorable, out_png,
                      n_queries=MONTAGE_QUERIES, topk=MONTAGE_TOPK):
    """Sampled queries and their top-``topk`` retrievals, bordered by correctness.

    Deliberately samples both successes and failures — a montage of hits says nothing
    about why the misses miss. Frames are re-rendered here (a few dozen, ~50 ms each)
    rather than cached, because nothing else in the pipeline needs them.

    Geography-free, so a traverse can use it too: the labels read as frame indices rather
    than image ids, and the border still means "inside the ground truth".
    """
    # ranked's columns run over the *scorable* queries only, so they have to be mapped back
    # through q_idx before they can index gt's columns.
    q_idx = np.flatnonzero(scorable)
    correct = gt[ranked[0], q_idx]
    rng = np.random.default_rng(0)
    half = max(1, n_queries // 2)
    picks = []
    for want in (True, False):
        pool = np.flatnonzero(correct == want)
        picks.append(rng.choice(pool, size=min(half, len(pool)), replace=False))
    cols = np.concatenate(picks) if any(len(p) for p in picks) else np.arange(0)
    if cols.size == 0:
        logger.warning("no queries to plot in the retrieval montage")
        return
    cols = np.sort(cols)

    rows = len(cols)
    fig, axes = plt.subplots(rows, topk + 1, figsize=(2.0 * (topk + 1), 1.7 * rows),
                             constrained_layout=True, squeeze=False)
    for r, col in enumerate(cols):
        ax = axes[r][0]
        ax.imshow(display_frame(q_paths[q_idx[col]]))
        ax.set_ylabel(f"query {q_idx[col]}", fontsize=7.5, color=C_MUTED)
        if r == 0:
            ax.set_title("query", fontsize=8.5, color=C_TEXT)
        ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(C_MUTED)
        for c in range(topk):
            ax = axes[r][c + 1]
            db_i = int(ranked[c, col])
            ax.imshow(display_frame(db_paths[db_i]))
            ok = bool(gt[db_i, q_idx[col]])
            if r == 0:
                ax.set_title(f"top-{c + 1}", fontsize=8.5, color=C_TEXT)
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(C_TP if ok else C_FP)
                spine.set_linewidth(2.2)
    fig.suptitle(f"top-{topk} retrievals — blue border = within the ground-truth radius, "
                 f"orange = outside", fontsize=10, color=C_TEXT)
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Scoring
# ---------------------------------------------------------------------------
def evaluate(tag, sim, gt, device):
    """Score one ``[n_db, n_q]`` similarity matrix -> ``(recalls, curve, ranked, scorable, sim)``.

    Takes the matrix rather than the descriptors because each method brings its own
    metric — cosine for megaevent, negated L1 for sparse_event, a raw dot product for
    EventVLAD — and only the ranking downstream of it is shared.
    """
    t0 = time.time()
    rec = recall_at_k(sim, gt)
    curve, ranked, scorable = recall_curve(sim, gt)
    cross_check(curve, rec)
    logger.info(f"[{tag}] " + "  ".join(f"R@{k}={rec[k]:.3f}" for k in KS)
                + f"   ({sim.shape[0]}db x {sim.shape[1]}q, {time.time() - t0:.0f}s)")
    return rec, curve, ranked, scorable, sim


def score_both(method, db_desc, q_desc, gt, device, results, curves, label="", paths=None,
               map_ks=()):
    """Score one descriptor bank under the method's native metric *and* PCA whitening.

    Every method is reported both ways on purpose. Our own headline number uses
    whitening, so applying it only to us and not to the baselines would rig the
    comparison; conversely the native metric is what each method actually published, and
    for sparse_event (L1 on unnormalised counts) PCA+cosine is *not* its method at all.
    Reporting both leaves nothing to argue about in either direction.

    A method may add to that in two ways, both optional and both currently only used by
    eventgem. ``extra_pca_powers`` adds whitening exponents beyond the shared
    :data:`src.inference.PCA_POWER`, for a method whose published configuration whitens
    differently — again so the comparison neither favours nor penalises it. ``rerank`` adds a
    second scoring pass over each query's shortlist, evaluated as a ``+rerank`` twin of every
    space, so the cost and benefit of that pass can be read off directly against the same
    method without it. Reranking needs the file lists, which is what ``paths`` carries.
    """
    def score(tag, sim):
        rec, curve, ranked, mask, sim = evaluate(tag, sim, gt, device)
        results["recall"][tag] = {str(k): rec[k] for k in KS}
        results["recall_curve"][tag] = {str(n): v for n, v in curve.items()}
        if map_ks:
            ap = map_at_k(ranked, gt, mask, map_ks)
            results.setdefault("map", {})[tag] = {str(k): ap[k] for k in map_ks}
            logger.info(f"[{tag}] " + "  ".join(f"mAP@{k}={ap[k]:.3f}" for k in map_ks))
        curves[tag] = curve
        return rec, curve, ranked, mask, sim

    def score_space(name, sim):
        """One space, plus its re-ranked twin if the method has one -> ``(plain, best)``."""
        tag = f"{label}{name}" if label else name
        plain = score(tag, sim)
        if paths is None or not hasattr(method, "rerank"):
            return plain, plain
        return plain, score(f"{tag}+rerank", method.rerank(plain[4], *paths, label=tag))

    native, _ = score_space("native", method.similarity(db_desc, q_desc, device))

    best = native
    results["pca"] = {}
    for power in (PCA_POWER,) + tuple(getattr(method, "extra_pca_powers", ())):
        pca = pca_fit_subsampled(db_desc, device, power=power)
        db_white, q_white = pca_apply(db_desc, pca, device), pca_apply(q_desc, pca, device)
        # The realised width, not pca["dim"]: svd_lowrank cannot return more components than
        # it had rows or columns, so a narrow bank silently yields a narrower basis.
        name = "pca" if power == PCA_POWER else f"pca{power:g}"
        results["pca"][name] = {"dim": int(db_white.size(1)), "power": pca["power"],
                                "eps": pca["eps"]}
        _, best = score_space(name, sim_matrix(db_white, q_white, device))
    # The caller saves the first of these as the method's similarity matrix and describes the
    # second in its figures, so: the plain native metric, and the strongest space scored —
    # the last whitening, re-ranked for a method that re-ranks.
    return native, best


def seed_summary(per_seed):
    """``{tag: {K: (mean, std)}}`` across sparse_event's pixel-selection seeds."""
    out = {}
    for tag in per_seed[0]:
        out[tag] = {}
        for k in per_seed[0][tag]:
            vals = [s[tag][k] for s in per_seed]
            out[tag][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                           "n": len(vals)}
    return out
