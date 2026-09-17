"""Figure-ready top-k retrieval panels for every benchmark, including mined failure cases.

    pixi run python3 scripts/topk_figures.py -d all --topk 3
    pixi run python3 scripts/topk_figures.py -d brisbane_event --database night \
        --cases impossible hard --tag v5_night_only

``src/scoring.py:figure_retrievals`` already draws a retrieval montage, but only as a
by-product of a full evaluation run, at a fixed 6 queries x top-5, and with no way to get at
the individual frames. A paper figure needs the frames themselves, composable, with a record
of what each one is. So this writes one PNG per frame at its native resolution, a montage per
case, and a ``manifest.json`` naming every image it wrote — query index, database index,
cosine similarity, ground-truth verdict and the metric distance behind that verdict.

**Sampling is the point.** A uniform draw shows the average retrieval, which is not what a
figure argues with. Five pools are mined from one deep ranking and sampled from *uniformly*
with ``--seed`` — an example drawn from a qualifying pool, never an extreme picked by argmax:

``easy``        top-1 correct, the whole rendered top-k correct, high similarity.
``hard``        top-1 wrong but a positive recovered inside the rendered top-k — orange,
                then blue, in the panel itself.
``nearmiss``    top-1 wrong, yet only just outside the radius (25-50 m by default). The
                ground truth is what calls these wrong, not the retrieval.
``impossible``  nothing correct anywhere in the mined depth, the top-1 lands far away, and
                the model is not confident about it either. Brisbane ``sunset1 -> night`` is
                the case in point: alone, that database gives R@1 0.099 and a top-1 a median
                1114 m from the query, against 0.750 / 16.8 m for ``sunrise``. The two
                recordings are effectively different event streams of the same road.
``random``      uniform over scorable queries, for a representative panel.

Every panel also carries the query's geographically **nearest true positive** (``--gt-column``,
on by default). That column is what makes ``impossible`` legible: it shows the answer that was
available, so a reader can see there was nothing there to retrieve.

Nothing that decides a number is re-implemented. Image datasets come from
:func:`src.imagesets.load_image_set`; pooled traverses from ``brisbane_pooled`` /
``nsavp_pooled`` via :mod:`scripts.figure_frames`; ranking from
``brisbane_pooled.topk_ranked``. R@1..R@k over the whole query set is printed for every run,
because a stale or misaligned bank produces a perfectly plausible-looking figure and an
obviously wrong recall.

``springfield_event`` is the third kind, and it is not ranked here at all: each query's top-20
is read from the dump ``scripts/springfield_diag.py --stage dump`` wrote — the ranking behind
the published number, whose R@1 is checked against the results JSON before anything is drawn.
Re-ranking would mean the 132,569 x 8448 pooled bank and a 2 GB ground-truth matrix, to draw
fifteen queries. ``--sweeps day dawn`` restricts the mined pools (and their quantile cuts) to
the daylight conditions; night is an appearance failure for every model and is reported as
numbers, not panels.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from matplotlib import image as mpimg
from matplotlib import pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from src.inference import C_FP, C_MUTED, C_TEXT, C_TP  # noqa: E402
from src.methods import get_method  # noqa: E402
from src.scoring import cached_array  # noqa: E402
from brisbane_pooled import topk_ranked  # noqa: E402
from figure_frames import (DATA_ROOT, load_image_benchmark, load_pooled_benchmark,  # noqa: E402
                           load_springfield_benchmark)

DEFAULT_CKPT = os.path.join(REPO, "ckpts", "megaevent_vitb_mloc.pt")
CASES = ("random", "easy", "hard", "nearmiss", "impossible")

# The six rows of `tab:native`, and where the shipping v5 ViT-B's descriptors for each of them
# live. Mirrors `scripts/table_native.py:DATASETS`: one place that says what the benchmark is,
# so a figure and a table cannot quietly be measuring different things. An image dataset with
# no `db_bank` is extracted on first use into <feature-dir>/<dataset>/<tag>_<split>.npy, the
# manifest-checked name every other script in this repo reads.
BENCHMARKS = {
    "tokyo247": {"kind": "image", "ref": "database", "query": "queries",
                 "db_bank": f"{DATA_ROOT}/tokyo_v5/r322_database_step3500.npy",
                 "query_bank": f"{DATA_ROOT}/tokyo_v5/r322_queries_step3500.npy"},
    "msls": {"kind": "image", "ref": "database", "query": "query"},
    "pitts250k": {"kind": "image", "ref": "database", "query": "queries"},
    "pitts": {"kind": "image", "ref": "ref", "query": "query"},
    # The three real-event rows carry the v8-B accumulate model (the current primary,
    # memory v8-accumulate-verdict); their `representation` pins the FRAME RENDER to
    # what those banks were extracted with, resolved in run_dataset (CLI overrides).
    "nycevent": {"kind": "image", "ref": "database", "query": "queries",
                 "representation": "accumulate",
                 "db_bank": f"{DATA_ROOT}/evaluations/nycevent/nycevent/"
                            f"b_v8_accum_s750_s750_3f4eb54780_accumulate_r322_database.npy",
                 "query_bank": f"{DATA_ROOT}/evaluations/nycevent/nycevent/"
                               f"b_v8_accum_s750_s750_3f4eb54780_accumulate_r322_queries.npy"},
    "brisbane_event": {"kind": "pooled", "query": "sunset1",
                       "database": ("daytime", "morning", "night", "sunrise"),
                       "eventlab_dir": "/media/adam/vprdatasets/eventgem",
                       "npz_root": f"{DATA_ROOT}/brisbane_npz",
                       "representation": "accumulate",
                       "bank_dir": f"{DATA_ROOT}/v8_bench/brisbane",
                       "bank_tag": "r322ba50", "bank_label": "v8"},
    "nsavp": {"kind": "pooled", "query": "R0_FA0",
              "database": ("R0_FN0", "R0_FS0", "R0_RA0", "R0_RN0", "R0_RS0"),
              "eventlab_dir": "/media/adam/vprdatasets/eventlab",
              "representation": "accumulate",
              "bank_dir": f"{DATA_ROOT}/v8_bench/nsavp",
              "bank_tag": "r322ba50", "bank_label": "v8"},
    # The full Springfield capture (memory springfield-full-eval): the same v8-B accumulate
    # model, BA-off, ranked once by springfield_diag's dump stage. `results` names the run,
    # `dump` its ranking; both are overridable together (--results/--dump) for the BA-on arm
    # or the ViT-S dump.
    "springfield_event": {
        "kind": "springfield", "representation": "accumulate",
        "root": "/media/adam/vprdatasets/megaevent/springfield/sessions",
        "bank_dir": f"{DATA_ROOT}/evaluations/springfield",
        "results": os.path.join(REPO, "output", "springfield_full",
                                "results_b_v8_accum_s750_s750_3f4eb54780_accumulate_r322"
                                "_hp1_baoff.json"),
        "dump": os.path.join(REPO, "output", "springfield_diag",
                             "dump_b_v8_accum_s750_s750_3f4eb54780_accumulate_r322"
                             "_hp1_baoff.npz")},
}
# `pitts` is same-panorama retrieval rather than a tab:native row, and `springfield_event` is
# not in that table yet, so both are reachable by name but not swept by `-d all`.
ALL = ("tokyo247", "msls", "pitts250k", "nycevent", "brisbane_event", "nsavp")


# ---------------------------------------------------------------------------
# 1. Banks
# ---------------------------------------------------------------------------
def load_given_bank(path, n_rows, split, expect_dim=None, names=None):
    """A precomputed ``[N, D]`` bank, checked as far as it can be checked.

    Banks written by ``scripts/tokyo_trajectory.py`` carry no ``_paths.txt`` manifest, so row
    order is all that ties a descriptor to its image — and that order is
    ``src.npzdata.list_npz`` on the split directory, which is exactly what
    :func:`src.imagesets.load_image_set` lists. The row count is therefore the only available
    check, and it is not a strong one, so where a sibling manifest *does* exist it is compared
    name for name.
    """
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2 or array.shape[0] != n_rows:
        raise SystemExit(f"{path}: {array.shape} for {n_rows} {split} images — this bank was "
                         f"not built from this split")
    if expect_dim is not None and array.shape[1] != expect_dim:
        raise SystemExit(f"{path}: {array.shape[1]}-d against the other bank's {expect_dim}-d")
    manifest = f"{os.path.splitext(path)[0]}_paths.txt"
    if os.path.exists(manifest) and names is not None:
        with open(manifest) as handle:
            cached = [line for line in handle.read().split() if line]
        if cached != [os.path.basename(p) for p in names]:
            raise SystemExit(f"{manifest}: the bank was built from a different file list than "
                             f"the dataset adapter returns")
        checked = "manifest verified"
    else:
        checked = "no manifest — aligned by list_npz order alone"
    print(f"  {split}: {array.shape} from {path}  ({checked})")
    return array


def image_banks(cli, spec, device):
    """``(dataset, (db, queries), meta)`` for an image dataset — given, or extracted."""
    from src.imagesets import load_image_set

    dataset = load_image_set(cli)
    n_db, n_q = len(dataset.db_paths), len(dataset.q_paths)
    if cli.db_bank or spec.get("db_bank"):
        db_path = cli.db_bank or spec["db_bank"]
        q_path = cli.query_bank or spec["query_bank"]
        if cli.limit:
            raise SystemExit("--limit reshapes the gallery, so a precomputed full-gallery "
                             "bank would no longer line up with it")
        db = load_given_bank(db_path, n_db, cli.ref, names=dataset.db_paths)
        queries = load_given_bank(q_path, n_q, cli.query, db.shape[1], dataset.q_paths)
        meta = {"db_bank": os.path.abspath(db_path), "query_bank": os.path.abspath(q_path),
                "native_metric": "cosine"}
    else:
        method = get_method(cli.method, cli, device)
        db = cached_array(cli, method, cli.ref, dataset.db_paths, "",
                          lambda p, s: method.descriptors(p, s), "descriptors", mmap=True)
        queries = cached_array(cli, method, cli.query, dataset.q_paths, "",
                               lambda p, s: method.descriptors(p, s), "descriptors", mmap=True)
        meta = {"method": method.name, "native_metric": method.native_metric,
                "bank_tag": method.tag, **method.meta}
    banks = (torch.from_numpy(np.asarray(db)), torch.from_numpy(np.asarray(queries)))
    return dataset, banks, meta


# ---------------------------------------------------------------------------
# 2. Case mining
# ---------------------------------------------------------------------------
def top1_distances(bench, rows):
    """Metres from each query to the database row ``rows[q]``, ``nan`` where undefined.

    ``nan`` is not "far": it means the pair has no shared metric frame at all, which for MSLS
    is a retrieval in another city and another UTM zone. The callers below decide what that
    means for each case, rather than a subtraction of two incomparable eastings deciding it
    for them.
    """
    n_q = bench.shape[1]
    if bench.db_xy is None or bench.q_xy is None:
        return np.full(n_q, np.nan)
    d = np.linalg.norm(bench.db_xy[rows] - bench.q_xy, axis=1)
    groups = bench.meta.get("groups")
    if groups is not None:
        d = np.where(groups[0][rows] == groups[1], d, np.nan)
    return d


def recall_from_hit(hit, scorable, ks):
    """``({k: R@k}, {n: R@n})`` over the ``scorable`` query indices.

    ``brisbane_pooled.recall_from_ranked``'s arithmetic, on the hit matrix
    :meth:`Benchmark.hit` returns — so a benchmark with no dense ground truth scores by the
    same rule as one with.
    """
    found = np.cumsum(hit, axis=0) > 0
    curve = {n: float(found[n - 1, scorable].mean()) for n in range(1, hit.shape[0] + 1)}
    return {k: curve[k] for k in ks}, curve


def query_stats(bench, hit, ranked, scores, topk):
    """Per-query facts the case rules are written in terms of. ``hit`` is ``bench.hit(ranked)``."""
    n_q = hit.shape[1]
    any_hit = hit.any(axis=0)
    # -1 for "no positive anywhere in the mined depth", which is what `impossible` is about.
    first_correct = np.where(any_hit, hit.argmax(axis=0), -1)
    return {"first_correct": first_correct,
            "all_topk_correct": hit[:topk].all(axis=0),
            "s1": scores[0].astype(np.float64),
            "margin": (scores[0] - scores[1]).astype(np.float64) if scores.shape[0] > 1
                      else np.zeros(n_q),
            "d1": top1_distances(bench, ranked[0])}


def mine(bench, stats, scorable, cli):
    """``{case: pool}`` — the qualifying query indices for each case, before sampling.

    ``hard`` is bounded by the *rendered* top-k rather than the mined depth, and that bound is
    the whole rule. Tokyo queries carry 144 positives out of 75,984 database frames, so a
    positive turning up at rank 13 of 20 is chance, not near-success: the panel would show
    three orange borders and a caption claiming the model nearly had it. Requiring the recovery
    inside the top-k means the figure shows it — orange, then blue.
    """
    first, s1, d1 = stats["first_correct"], stats["s1"], stats["d1"]
    # A retrieval with no shared metric frame is as wrong as a retrieval can be, so it counts
    # as far away; it is excluded from `nearmiss`, which is defined by a small distance.
    far = np.where(np.isnan(d1), np.inf, d1)
    hits = scorable[first[scorable] == 0]
    easy_cut = np.quantile(s1[hits], cli.easy_quantile) if hits.size else np.inf
    cold_cut = np.quantile(s1[scorable], cli.impossible_quantile) if scorable.size else -np.inf
    threshold = bench.threshold_m
    rules = {
        "random": np.ones(len(first), dtype=bool),
        "easy": (first == 0) & stats["all_topk_correct"] & (s1 >= easy_cut),
        "hard": (first >= 1) & (first < cli.topk),
        "nearmiss": (first != 0) & (d1 > threshold) & (d1 <= cli.near_max * threshold),
        "impossible": (first == -1) & (far > cli.far_min * threshold) & (s1 <= cold_cut),
    }
    keep = np.zeros(len(first), dtype=bool)
    keep[scorable] = True
    return {name: np.flatnonzero(rule & keep) for name, rule in rules.items()}


def nearest_positive(bench, q):
    """``(database row, distance)`` of the query's closest true positive, or ``(None, None)``.

    The answer that *was* available. Without it a reader has to take "there was nothing to
    retrieve" on trust; with it they can look at the frame and see it.
    """
    positives = bench.positives(q)
    if not positives.size:
        return None, None
    if bench.db_xy is None or bench.q_xy is None:
        return int(positives[0]), None
    d = np.linalg.norm(bench.db_xy[positives] - bench.q_xy[q], axis=1)
    best = int(np.argmin(d))
    return int(positives[best]), float(d[best])


# ---------------------------------------------------------------------------
# 3. Figures
# ---------------------------------------------------------------------------
def panel(ax, image, title, colour, width=2.2):
    ax.imshow(image)
    if title:
        ax.set_title(title, fontsize=8.5, color=C_TEXT)
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(colour)
        spine.set_linewidth(width)


def figure_montage(bench, entries, out_png, topk, dpi, case):
    """One row per sampled query: the query, its top-``topk``, and the nearest true positive.

    Modelled on :func:`src.scoring.figure_retrievals`, but it is handed the queries instead of
    choosing them, and ``ranked``'s columns here run over *every* query rather than the
    scorable subset — so a query index is a query index throughout, with no remapping to get
    wrong.
    """
    columns = topk + 1 + int(any("nearest_positive" in e for e in entries))
    fig, axes = plt.subplots(len(entries), columns, squeeze=False,
                             figsize=(2.0 * columns, 1.85 * len(entries)),
                             constrained_layout=True)
    for r, entry in enumerate(entries):
        q = entry["query_index"]
        panel(axes[r][0], bench.q_frames.render(q), "query" if r == 0 else None, C_MUTED, 1.2)
        caption = getattr(bench.q_frames, "caption", None)
        axes[r][0].set_ylabel(caption(q) if caption else f"q{q}", fontsize=7.5, color=C_MUTED)
        for c, item in enumerate(entry["retrievals"]):
            colour = C_TP if item["correct"] else C_FP
            distance = "" if item["distance_m"] is None else f"\n{item['distance_m']:.0f} m"
            panel(axes[r][c + 1], bench.db_frames.render(item["database_index"]),
                  f"top-{c + 1}" if r == 0 else None, colour)
            axes[r][c + 1].set_xlabel(f"{item['similarity']:.3f}{distance}", fontsize=6.5,
                                      color=C_MUTED)
        near = entry.get("nearest_positive")
        if near is not None and columns > topk + 1:
            ax = axes[r][-1]
            panel(ax, bench.db_frames.render(near["database_index"]),
                  "nearest GT" if r == 0 else None, C_TP, 1.2)
            label = "" if near["distance_m"] is None else f"{near['distance_m']:.0f} m"
            ax.set_xlabel(f"{label}  sim {near['similarity']:.3f}", fontsize=6.5, color=C_MUTED)
    fig.suptitle(f"{bench.dataset} — {case}: top-{topk} retrievals, blue = within "
                 f"{bench.threshold_m:g} m of the query, orange = outside",
                 fontsize=10, color=C_TEXT)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. One dataset
# ---------------------------------------------------------------------------
def write_case(bench, case, picks, ranked, scores, stats, out_dir, cli):
    """Write every sampled query of one case, and return its manifest entries."""
    case_dir = os.path.join(out_dir, case)
    os.makedirs(case_dir, exist_ok=True)
    entries = []
    for q in picks:
        q = int(q)
        q_dir = os.path.join(case_dir, f"q{q:05d}")
        os.makedirs(q_dir, exist_ok=True)
        mpimg.imsave(os.path.join(q_dir, "query.png"), bench.q_frames.render(q))

        retrievals = []
        for rank in range(cli.topk):
            db_i = int(ranked[rank, q])
            ok = bench.correct(db_i, q)
            name = f"top{rank + 1}_{'correct' if ok else 'wrong'}_db{db_i:06d}.png"
            mpimg.imsave(os.path.join(q_dir, name), bench.db_frames.render(db_i))
            retrievals.append({"rank": rank + 1, "database_index": db_i,
                               "database_key": bench.db_frames.key(db_i),
                               "similarity": float(scores[rank, q]), "correct": ok,
                               "distance_m": bench.distance(db_i, q), "image": name,
                               **bench.db_frames.provenance(db_i)})

        entry = {"case": case, "query_index": q, "query_key": bench.q_frames.key(q),
                 "n_positives": int(len(bench.positives(q))),
                 "first_correct_rank": (None if stats["first_correct"][q] < 0
                                        else int(stats["first_correct"][q]) + 1),
                 "top1_distance_m": (None if np.isnan(stats["d1"][q])
                                     else float(stats["d1"][q])),
                 "top1_similarity": float(stats["s1"][q]),
                 "margin_to_top2": float(stats["margin"][q]),
                 "directory": os.path.relpath(q_dir, cli.out_dir),
                 "retrievals": retrievals, **bench.q_frames.provenance(q)}
        # Springfield's dump knows the best-correct row's exact rank over the whole gallery —
        # the caption fact behind an `impossible` panel ("the correct frame sat at rank 718").
        for field, values in bench.extra.items():
            value = values[q]
            entry[field] = int(value) if np.issubdtype(values.dtype, np.integer) else float(value)

        if cli.gt_column:
            db_i, distance = nearest_positive(bench, q)
            if db_i is not None:
                name = f"gt_nearest_db{db_i:06d}.png"
                mpimg.imsave(os.path.join(q_dir, name), bench.db_frames.render(db_i))
                entry["nearest_positive"] = {
                    "database_index": db_i, "database_key": bench.db_frames.key(db_i),
                    "distance_m": distance, "similarity": bench.similarity(db_i, q),
                    "image": name,
                    **bench.db_frames.provenance(db_i)}
        entries.append(entry)

    if entries:
        figure_montage(bench, entries, os.path.join(case_dir, "montage.png"), cli.topk,
                       cli.dpi, case)
    return entries


def run_dataset(cli, name, device):
    """Mine, render and record one benchmark. -> the manifest it wrote."""
    spec = dict(BENCHMARKS[name])
    cli.dataset = name
    cli.ref, cli.query = spec.get("ref"), spec.get("query")
    # The frame render must match the bank's representation; the registry knows what
    # each dataset's bank was extracted with, and an explicit flag overrides it.
    cli.representation = cli.representation_override or spec.get("representation",
                                                                 "countmask")
    print(f"\n{'=' * 78}\n{name}  ({cli.representation})\n{'=' * 78}")

    if spec["kind"] == "image":
        dataset, banks, meta = image_banks(cli, spec, device)
        bench = load_image_benchmark(cli, dataset, banks)
    elif spec["kind"] == "springfield":
        bench = load_springfield_benchmark(cli, spec)
        meta = {"native_metric": "cosine"}
    else:
        cli.query = cli.query_traverse or spec["query"]
        cli.database = cli.database_traverses or list(spec["database"])
        cli.bank_dir = cli.bank_dir_override or spec["bank_dir"]
        cli.bank_label = cli.bank_label_override or spec["bank_label"]
        bench = load_pooled_benchmark(cli, spec)
        meta = {"native_metric": "cosine", "bank_dir": cli.bank_dir}
    meta = {**meta, **bench.meta}
    # `groups` is a pair of string arrays used only by Benchmark.distance; it is not JSON.
    meta.pop("groups", None)

    n_db, n_q = bench.shape
    scorable = np.flatnonzero(bench.n_positives_all() > 0)
    print(f"  {n_db} database x {n_q} queries, {len(scorable)} scorable "
          f"@ {bench.threshold_m:g} m")
    if cli.topk > n_db:
        raise SystemExit(f"--topk {cli.topk} exceeds the {n_db}-image database")

    depth = max(cli.mine_depth, cli.topk + 1)
    if bench.ranking is not None:
        ranked, scores = bench.ranking
        if depth > ranked.shape[0]:
            raise SystemExit(f"--mine-depth {cli.mine_depth}: the precomputed ranking holds "
                             f"{ranked.shape[0]} rows per query — re-run "
                             f"scripts/springfield_diag.py --stage dump with a larger K")
        ranked, scores = ranked[:depth], scores[:depth]
    else:
        ranked, scores = topk_ranked(bench.db, bench.queries, device, k=min(depth, n_db),
                                     chunk=cli.score_chunk, db_chunk=cli.db_chunk,
                                     return_scores=True)
    hit = bench.hit(ranked)
    ks = tuple(range(1, cli.topk + 1))
    recall, curve = recall_from_hit(hit, scorable, ks)
    print("  over all scorable queries: " + "  ".join(f"R@{k}={recall[k]:.4f}" for k in ks)
          + f"   R@{len(curve)}={curve[len(curve)]:.4f}")
    # A precomputed ranking is only as good as its provenance: the dump must reproduce the
    # number the results JSON published, or it is some other run's ranking.
    published = bench.meta.get("published_r1")
    if published is not None:
        if abs(recall[1] - published) > 1e-4:
            raise SystemExit(f"R@1 {recall[1]:.6f} from the ranking dump against "
                             f"{published:.6f} in the results JSON — not the same run")
        print(f"  matches the results JSON's R@1 {published:.4f}")

    # An optional query selection (Springfield's day/dawn/night). The pools, and the quantile
    # cuts inside them, are mined over the selection; the recall above is still the whole set.
    selection, selection_meta = scorable, None
    if cli.sweeps:
        if bench.q_labels is None:
            raise SystemExit(f"--sweeps: {name} carries no per-query condition labels")
        known = sorted(set(map(str, np.unique(bench.q_labels))))
        unknown = [s for s in cli.sweeps if s not in known]
        if unknown:
            raise SystemExit(f"--sweeps {unknown[0]}: {name}'s conditions are {known}")
        chosen = np.isin(bench.q_labels, cli.sweeps)
        selection = scorable[chosen[scorable]]
        sel_recall, _ = recall_from_hit(hit, selection, ks)
        print(f"  --sweeps {' '.join(cli.sweeps)}: {len(selection)} queries, "
              + "  ".join(f"R@{k}={sel_recall[k]:.4f}" for k in ks))
        selection_meta = {"sweeps": list(cli.sweeps), "n_queries": int(len(selection)),
                          "recall": {str(k): sel_recall[k] for k in ks}}

    stats = query_stats(bench, hit, ranked, scores, cli.topk)
    pools = mine(bench, stats, selection, cli)
    rng = np.random.default_rng(cli.seed)
    out_dir = os.path.join(cli.out_dir, name, cli.tag)
    os.makedirs(out_dir, exist_ok=True)

    sampled, pool_sizes = {}, {}
    for case in cli.cases:
        pool = pools[case]
        pool_sizes[case] = int(pool.size)
        if not pool.size:
            print(f"  {case:11s} pool empty — nothing qualifies")
            continue
        take = min(cli.per_case, pool.size)
        picks = np.sort(rng.choice(pool, size=take, replace=False))
        sampled[case] = write_case(bench, case, picks, ranked, scores, stats, out_dir, cli)
        hits = sum(1 for e in sampled[case] if e["retrievals"][0]["correct"])
        print(f"  {case:11s} pool {pool.size:6d}  drew {[int(p) for p in picks]}  "
              f"{hits}/{take} correct at top-1")

    manifest = {"dataset": name, "tag": cli.tag, "meta": meta,
                "n_database": int(n_db), "n_queries": int(n_q),
                "scorable_queries": int(len(scorable)),
                "threshold_m": bench.threshold_m, "seed": cli.seed, "topk": cli.topk,
                "mine_depth": int(ranked.shape[0]), "per_case": cli.per_case,
                "case_rules": {"easy_quantile": cli.easy_quantile,
                               "nearmiss_max_m": cli.near_max * bench.threshold_m,
                               "impossible_min_m": cli.far_min * bench.threshold_m,
                               "impossible_quantile": cli.impossible_quantile},
                "recall_all_queries": {str(k): recall[k] for k in ks},
                "query_selection": selection_meta,
                "case_pool_sizes": pool_sizes, "sampled": sampled}
    with open(os.path.join(out_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"  -> {out_dir}")
    return manifest


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", "-d", nargs="+", required=True,
                    help=f"one or more of {', '.join(sorted(BENCHMARKS))}, or 'all' for the "
                         f"six tab:native rows")
    ap.add_argument("--tag", default="v5_s3500",
                    help="names the output folder under output/<dataset>/")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--per-case", type=int, default=3,
                    help="queries sampled from each case's pool")
    ap.add_argument("--mine-depth", type=int, default=20,
                    help="how deep to rank when deciding whether a query was recoverable. "
                         "Only the rendered top-k is drawn, so depth is nearly free.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gt-column", dest="gt_column", action="store_false",
                    help="omit the nearest-true-positive panel")
    # Case thresholds, in multiples of the ground-truth radius so they travel across datasets.
    ap.add_argument("--near-max", type=float, default=2.0,
                    help="nearmiss upper bound, in multiples of the radius (default 2 = 50 m)")
    ap.add_argument("--far-min", type=float, default=4.0,
                    help="impossible lower bound, in multiples of the radius (default 4 = 100 m)")
    ap.add_argument("--easy-quantile", type=float, default=0.75)
    ap.add_argument("--impossible-quantile", type=float, default=0.25)

    ap.add_argument("--method", default="megaevent")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--eval-resolution", type=int, default=322,
                    help="what every reported megaevent number is measured at")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "output"))
    # main.py's ./data and ./features defaults are not where anything lives on this box.
    ap.add_argument("--data-dir", default=DATA_ROOT)
    ap.add_argument("--feature-dir", default=f"{DATA_ROOT}/evaluations")
    ap.add_argument("--positive-dist-threshold", type=float, default=25.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="shrink an image gallery to a deterministic subset — a smoke test, "
                         "not a measurement")
    ap.add_argument("--db-bank", default=None,
                    help="a precomputed [n_db, D] .npy to rank with instead of extracting, "
                         "overriding the registry. Single-dataset runs only.")
    ap.add_argument("--query-bank", default=None, help="the query half of --db-bank")
    ap.add_argument("--query-traverse", default=None,
                    help="pooled datasets: the query traverse, overriding the registry")
    ap.add_argument("--database", dest="database_traverses", nargs="+", default=None,
                    help="pooled datasets: the traverses pooled into the gallery. "
                         "`--database night` is the sunset -> night showcase.")
    ap.add_argument("--bank-dir", dest="bank_dir_override", default=None)
    ap.add_argument("--bank-label", dest="bank_label_override", default=None,
                    help="pooled datasets: the checkpoint label in <tag>_<seq>_<label>.npy")
    ap.add_argument("--representation", dest="representation_override", default=None,
                    help="override the registry's per-dataset representation (which "
                         "matches what each bank was extracted with) — figures render "
                         "frames in this representation")
    ap.add_argument("--sweeps", nargs="+", default=None,
                    help="springfield_event: draw every case from these query conditions only "
                         "(day dawn night). The whole-set recall is still printed and checked.")
    ap.add_argument("--results", default=None,
                    help="springfield_event: a springfield_full results JSON to draw instead of "
                         "the registry's (e.g. the _ba50 arm); goes with --dump")
    ap.add_argument("--dump", default=None,
                    help="springfield_event: the matching springfield_diag dump_<tag>.npz")
    ap.add_argument("--dt-ms", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="frames per forward pass during extraction (default 64). Memory "
                         "only — the bank is identical either way. Lower it when another job "
                         "is sharing the GPU: 64 at 322x322 needs an idle 8 GB card.")
    ap.add_argument("--score-chunk", type=int, default=256)
    ap.add_argument("--db-chunk", type=int, default=40000,
                    help="database rows resident on the GPU at once")
    ap.add_argument("--dpi", type=int, default=130,
                    help="montage only; the per-frame PNGs are written at native resolution")
    cli = ap.parse_args()

    names = list(ALL) if "all" in cli.dataset else list(dict.fromkeys(cli.dataset))
    unknown = [n for n in names if n not in BENCHMARKS]
    if unknown:
        raise SystemExit(f"unknown dataset {unknown[0]} — choose from "
                         f"{', '.join(sorted(BENCHMARKS))} or 'all'")
    if cli.per_case < 1 or cli.topk < 1:
        raise SystemExit("--per-case and --topk must both be at least 1")
    if bool(cli.db_bank) != bool(cli.query_bank):
        raise SystemExit("--db-bank and --query-bank go together")
    if bool(cli.dump) != bool(cli.results):
        raise SystemExit("--dump and --results go together")
    overrides = (cli.db_bank, cli.query_traverse, cli.database_traverses, cli.bank_dir_override,
                 cli.sweeps, cli.dump)
    if len(names) > 1 and any(overrides):
        raise SystemExit("per-dataset overrides (--db-bank, --query-traverse, --database, "
                         "--bank-dir, --sweeps, --dump/--results) apply to one dataset — name "
                         "it explicitly")
    # Ordered as the user asked for them, so a montage's rows read in a fixed order.
    cli.cases = [c for c in CASES if c in cli.cases]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = {}
    for name in names:
        summary[name] = run_dataset(cli, name, device)

    if len(names) > 1:
        path = os.path.join(cli.out_dir, f"summary_{cli.tag}.json")
        with open(path, "w") as handle:
            json.dump({n: {k: v for k, v in m.items() if k != "sampled"}
                       for n, m in summary.items()}, handle, indent=2)
        print(f"\n{'dataset':<16s}{'db x queries':>20s}  " +
              "  ".join(f"R@{k}" for k in range(1, cli.topk + 1)))
        for n, m in summary.items():
            size = f"{m['n_database']:,} x {m['scorable_queries']:,}"
            cells = "  ".join(f"{m['recall_all_queries'][str(k)]:.3f}"
                              for k in range(1, cli.topk + 1))
            print(f"{n:<16s}{size:>20s}  {cells}")
        print(f"\n-> {path}")


if __name__ == "__main__":
    main()
