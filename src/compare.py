"""Collect every method's ``results_<method>.json`` into one table and one figure.

Run after the per-method evaluations::

    pixi run python3 -m src.compare --feature-dir ./features --dataset tokyo247

Emits ``comparison.md`` and ``comparison.png`` next to the results it read. Methods
whose recalls were averaged over seeds (sparse_event) carry their spread through to both
outputs; the others report a single number.
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

from src.inference import C_FP, C_MUTED, C_TEXT, C_TP, KS

# Ordered so the table and the legend read the same way every time, whichever methods
# happen to be present.
METHOD_ORDER = ["megaevent", "sparse_event", "eventvlad", "eventgem"]
# megaevent is the blue the rest of the repo uses for "ours"; the baselines take the
# orange and neutral greens/greys, all distinguishable in greyscale and to a colourblind
# reader thanks to the marker shapes.
STYLE = {
    "megaevent": (C_TP, "o", "-"),
    "sparse_event": (C_FP, "s", "--"),
    "eventvlad": ("#6b6a67", "^", ":"),
    "eventgem": ("#8ba888", "D", "-"),
}
# The two spaces every method reports, and so the only two the figure has panels for. A
# method may report more (eventgem adds a full-whitening space and a re-ranked twin of each);
# those reach the table, and the re-ranked twins are drawn into the panel of the space they
# re-rank so the cost and benefit of that second pass reads off one axis.
SPACES = ("native", "pca")
SPACE_LABELS = {
    "native": "each method's own metric",
    "pca": "PCA-whitened, cosine",
    "pca1": "fully whitened, cosine",
}
RERANK = "+rerank"
# The event sources the real-vs-I2E ablation compares, real before synthetic so a delta
# reads as "what the synthetic conversion cost". `None` is an image set, which has no arm.
# The `_masked` pair repeats the comparison with the dead vignette removed from *both*
# sides, which separates the domain gap from I2E's noise amplification in that region.
SOURCE_ORDER = (None, "real", "i2e", "real_masked", "i2e_masked")
SOURCE_LABELS = {"real": "real events", "i2e": "I2E from DAVIS frames",
                 "real_masked": "real events, vignette masked",
                 "i2e_masked": "I2E from DAVIS frames, vignette masked"}
# The pairs a Δ row is drawn for: (baseline, variant).
SOURCE_PAIRS = (("real", "i2e"), ("real_masked", "i2e_masked"))
# Solid for a real arm, dashed for a synthetic one, in each method's own colour: two
# nearly-coincident lines per method means the synthetic domain costs that method nothing.
SOURCE_DASH = {None: "-", "real": "-", "i2e": (0, (4, 2)),
               "real_masked": (0, (1, 1)), "i2e_masked": (0, (5, 1, 1, 1))}


def space_label(space):
    base = space[:-len(RERANK)] if space.endswith(RERANK) else space
    label = SPACE_LABELS.get(base, base)
    return f"{label}, homography re-ranked" if space.endswith(RERANK) else label


def spaces_present(results):
    """Every descriptor space at least one method reported, in a stable order.

    Built from the results rather than hardcoded, so a method that reports more than
    ``native``/``pca`` shows up without the table having to know about it in advance.
    """
    seen = {s for res in results.values() for s in res.get("recall", {})}
    ordered = [s for base in SPACES + ("pca1",) for s in (base, base + RERANK)]
    return [s for s in ordered if s in seen] + sorted(seen - set(ordered))


def load_results(feature_dir, dataset):
    """``({(method, source): results}, [sources])`` for every ``results_*.json`` found.

    ``source`` is ``None`` for an image set and ``"real"``/``"i2e"`` for a traverse scored
    through :mod:`src.traversevpr`. Keying on the pair rather than the method alone is what
    keeps the two arms of the real-vs-I2E ablation from collapsing onto each other — they
    are the same method on the same traverse, and only the event source differs.
    """
    out_dir = os.path.join(feature_dir, dataset)
    found = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "results_*.json"))):
        if "_limit" in os.path.basename(path):      # smoke runs are not comparable
            continue
        with open(path) as f:
            res = json.load(f)
        found[(res.get("method", os.path.basename(path)[8:-5]), res.get("source"))] = res
    if not found:
        raise FileNotFoundError(
            f"no results_*.json in {out_dir} — run main.py for at least one --method first")

    sources = [s for s in SOURCE_ORDER if any(k[1] == s for k in found)]
    sources += sorted({k[1] for k in found} - set(sources), key=lambda s: (s is None, s))
    order = [(m, s) for s in sources for m in METHOD_ORDER if (m, s) in found]
    ordered = {k: found[k] for k in order} | {k: v for k, v in found.items() if k not in order}
    return out_dir, ordered, sources


def recall_entry(res, space, k):
    """``(mean, std_or_None)`` for one method / descriptor space / cutoff.

    Multi-seed methods store ``{"mean":…, "std":…, "n":…}``; single-run methods store a
    bare float. Normalising here keeps every consumer below free of the distinction.
    """
    val = res["recall"].get(space, {}).get(str(k))
    if val is None:
        return None, None
    if isinstance(val, dict):
        return float(val["mean"]), float(val["std"])
    return float(val), None


def curve_for(res, space):
    """``{N: (mean, std_or_None)}`` — averaged across seeds where there are seeds."""
    curves = res.get("recall_curve", {})
    if space in curves:
        return {int(n): (float(v), None) for n, v in curves[space].items()}
    per_seed = [c[space] for key, c in curves.items()
                if key.startswith("seed") and space in c]
    if not per_seed:
        return {}
    ns = sorted(int(n) for n in per_seed[0])
    return {n: (float(np.mean([c[str(n)] for c in per_seed])),
                float(np.std([c[str(n)] for c in per_seed]))) for n in ns}


def _fmt(mean, std):
    if mean is None:
        return "—"
    return f"{mean:.3f}" if std is None else f"{mean:.3f} ± {std:.3f}"


def _methods_in(results):
    """Method names present, in :data:`METHOD_ORDER` first, then whatever else turned up."""
    seen = list(dict.fromkeys(m for m, _ in results))
    return [m for m in METHOD_ORDER if m in seen] + [m for m in seen if m not in METHOD_ORDER]


def _scale_line(first):
    """One sentence describing the benchmark, whichever evaluation produced it."""
    n_db, n_q = first["n_database"], first["scorable_queries"]
    if "threshold_m" in first:                  # image set: geographic radius
        chance = first["positives_per_query_mean"] / n_db
        return (f"{n_db} database images, {n_q} scorable queries, "
                f"{first['threshold_m']:g} m ground-truth radius. "
                f"Chance R@1 for a random ranking is {chance:.4f}.")
    # traverse: an Event-LAB band, whose density *is* the chance rate — mean positives per
    # query is density x n_database, so dividing by n_database gives the density back.
    return (f"{n_db} reference frames, {n_q} scorable queries at {first.get('dt_ms', '?')} ms, "
            f"scored against the Event-LAB ground-truth band "
            f"(density {first.get('gt_density', float('nan')):.4f}, which is also chance R@1). "
            f"{first.get('dropped_database', 0)} reference and "
            f"{first.get('dropped_queries', 0)} query frames are excluded from both arms "
            f"because no DAVIS intensity frame fell within half a slice of them.")


def write_table(results, out_md, dataset, sources):
    first = next(iter(results.values()))
    paired = [s for s in sources if s is not None]

    lines = [f"# {dataset} — method comparison", "", _scale_line(first), ""]
    if len(paired) > 1:
        lines += ["Each method is scored twice on the same traverse pair, from the same "
                  "frame indices and against the same ground truth — only the event source "
                  "differs. **Δ is what converting the DAVIS intensity frames with I2E costs "
                  "that method**; a Δ near zero means the synthetic domain is not what holds "
                  "it back.", ""]

    for space in spaces_present(results):
        lines += [f"## {space} ({space_label(space)})", ""]
        if len(paired) > 1:
            lines += ["| method | metric | source | " + " | ".join(f"R@{k}" for k in KS) + " |",
                      "|---|---|---|" + "---|" * len(KS)]
            for name in _methods_in(results):
                rows = [(s, results[(name, s)]) for s in paired if (name, s) in results]
                rows = [(s, r) for s, r in rows if space in r.get("recall", {})]
                if not rows:
                    continue
                metric = (rows[0][1].get("native_metric", "?")
                          if space.startswith("native") else "cosine")
                have = dict(rows)
                for s, res in rows:
                    cells = [_fmt(*recall_entry(res, space, k)) for k in KS]
                    lines.append(f"| {name} | {metric} | {SOURCE_LABELS.get(s, s)} | "
                                 + " | ".join(cells) + " |")
                for base, variant in SOURCE_PAIRS:
                    if base not in have or variant not in have:
                        continue
                    deltas = []
                    for k in KS:
                        a, _ = recall_entry(have[base], space, k)
                        b, _ = recall_entry(have[variant], space, k)
                        deltas.append("—" if a is None or b is None else f"**{b - a:+.3f}**")
                    tail = " (masked)" if base.endswith("_masked") else ""
                    lines.append(f"| {name} | | **Δ i2e − real{tail}** | "
                                 + " | ".join(deltas) + " |")
            lines.append("")
            continue

        lines += ["| method | metric | " + " | ".join(f"R@{k}" for k in KS) + " |",
                  "|---|---|" + "---|" * len(KS)]
        for (name, _), res in results.items():
            if space not in res.get("recall", {}):
                continue                        # a space this method does not report
            cells = [_fmt(*recall_entry(res, space, k)) for k in KS]
            metric = res.get("native_metric", "?") if space.startswith("native") else "cosine"
            lines.append(f"| {name} | {metric} | " + " | ".join(cells) + " |")
        lines.append("")

    seeded = {m: r for (m, _), r in results.items() if "seeds" in r}
    if seeded:
        lines += ["> Values with ± are the mean and standard deviation over "
                  + ", ".join(f"{len(r['seeds'])} seeds ({m})" for m, r in seeded.items())
                  + ". That method selects its readout pixels at random, so a single "
                    "run is not a measurement.", ""]
    lines += ["> `pca` is reported for every method so the whitening that helps one is "
              "not withheld from the others. It is not the published metric for "
              "sparse_event (L1 on unnormalised counts) or EventVLAD.", ""]
    if any(s.endswith(RERANK) or s == "pca1" for s in spaces_present(results)):
        lines += ["> `pca1` whitens fully (power 1.0) rather than at the shared power 0.5, "
                  "which is what Event-GeM publishes; it is reported alongside `pca` so that "
                  "method is neither understated by the shared setting nor given a whitening "
                  "the others do not get. `+rerank` is a second pass that geometrically "
                  "verifies each query's top-K shortlist — only Event-GeM has one, and its "
                  "rows without it are the like-for-like global-descriptor comparison.", ""]

    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)


def _plot_curve(ax, curve, colour, marker, linestyle, width, label):
    ns = sorted(curve)
    mean = np.array([curve[n][0] for n in ns])
    std = np.array([curve[n][1] or 0.0 for n in ns])
    ax.plot(ns, mean, color=colour, marker=marker, markersize=3.5, linewidth=width,
            linestyle=linestyle, label=f"{label}  (R@1 = {mean[0]:.3f})")
    if std.any():
        ax.fill_between(ns, mean - std, mean + std, color=colour, alpha=0.18, linewidth=0)


def figure_arms(results, out_png, dataset, sources, k=1):
    """R@``k`` against the event source, one line per method — the ablation in one panel.

    A slope chart, because the question is *change between conditions per method*, not
    magnitude and not a distribution: a flat line means that method does not care whether
    its events were recorded or synthesised, and that is the whole finding. Stacking four
    methods x four sources onto the recall-curve panels instead would put sixteen lines in
    one axes and answer nothing.

    Colour identifies the method and comes from :data:`STYLE`, unchanged from every other
    figure in the repo so a reader carries one mapping across all of them. Every series is
    also direct-labelled: with four series that is within the "label them all" budget, it
    removes the colour lookup entirely (two of the four are deliberately low-chroma), and
    it is the visible-label relief the lightest of them needs against a white surface.
    """
    present = [s for s in sources if s is not None]
    methods = _methods_in(results)
    fig, axes = plt.subplots(1, len(SPACES), figsize=(5.6 * len(SPACES), 4.8),
                             constrained_layout=True, sharey=True)
    x = np.arange(len(present))
    for panel, (ax, space) in enumerate(zip(np.atleast_1d(axes), SPACES)):
        for name in methods:
            colour, marker, _ = STYLE.get(name, (C_MUTED, "d", "-."))
            ys, es, xs = [], [], []
            for i, source in enumerate(present):
                res = results.get((name, source))
                if res is None:
                    continue
                mean, std = recall_entry(res, space, k)
                if mean is None:
                    continue
                xs.append(i), ys.append(mean), es.append(std or 0.0)
            if not ys:
                continue
            ax.errorbar(xs, ys, yerr=es if any(es) else None, color=colour, marker=marker,
                        markersize=8, linewidth=2.0, capsize=3, elinewidth=1.2, zorder=3)
            ax.annotate(f" {name}", (xs[-1], ys[-1]), color=C_TEXT, fontsize=8.5,
                        va="center", ha="left", annotation_clip=False)
        ax.set_xticks(x)
        ax.set_xticklabels([SOURCE_LABELS.get(s, s).replace(", ", ",\n") for s in present],
                           fontsize=8)
        ax.set_xlim(-0.35, len(present) - 0.35)
        ax.set_ylim(0, 1)
        ax.set_title(f"({'ab'[panel]}) {space_label(space)}", fontsize=10, color=C_TEXT,
                     loc="left")
        ax.grid(True, axis="y", linewidth=0.5, color="#ececea")
        ax.set_axisbelow(True)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#dcdbd6")
    np.atleast_1d(axes)[0].set_ylabel(f"Recall@{k}", color=C_MUTED, fontsize=9)
    fig.suptitle(f"{dataset}: what the event source costs each method  —  "
                 f"a flat line means the source does not matter to it",
                 fontsize=11, color=C_TEXT)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure(results, out_png, dataset, sources):
    first = next(iter(results.values()))
    paired = [s for s in sources if s is not None]
    by_arm = len(paired) > 1
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True,
                             sharey=True)
    for panel, (ax, space) in enumerate(zip(axes, SPACES)):
        for (name, source), res in results.items():
            colour, marker, dash = STYLE.get(name, (C_MUTED, "d", "-."))
            if by_arm:
                # One colour per method, linestyle for the arm. Two nearly-coincident lines
                # of the same colour is the result the ablation is looking for; the
                # +rerank twins are left to the table so the panel stays readable.
                curve = curve_for(res, space)
                if curve:
                    _plot_curve(ax, curve, colour, marker, SOURCE_DASH.get(source, dash),
                                1.5, f"{name} — {SOURCE_LABELS.get(source, source)}")
                continue
            # The plain space, then its re-ranked twin on the same colour: same method, one
            # extra pass, so the gap between the two lines is what that pass bought.
            for variant, width in ((space, 1.5), (space + RERANK, 2.2)):
                curve = curve_for(res, variant)
                if not curve:
                    continue
                _plot_curve(ax, curve, colour, marker,
                            "-." if variant.endswith(RERANK) else dash, width,
                            f"{name}{RERANK if variant.endswith(RERANK) else ''}")
        ax.set_xlabel("N (number of retrieved candidates)", color=C_MUTED, fontsize=9)
        ax.set_title(f"({'ab'[panel]}) {space_label(space)}",
                     fontsize=10, color=C_TEXT, loc="left")
        ax.set_xlim(1, max(max(curve_for(r, space) or {1: 0}) for r in results.values()))
        ax.set_ylim(0, 1)
        ax.grid(True, linewidth=0.5, color="#ececea")
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", fontsize=7.5 if by_arm else 8.5, framealpha=0.9)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#dcdbd6")
    axes[0].set_ylabel("Recall@N", color=C_MUTED, fontsize=9)
    scale = (f"{first['n_database']} database x {first['scorable_queries']} scorable queries, "
             + (f"{first['threshold_m']:g} m" if "threshold_m" in first
                else f"{first.get('dt_ms', '?')} ms slices"))
    fig.suptitle(f"{dataset}: {scale}"
                 + ("  —  solid = real events, dashed = I2E from DAVIS frames"
                    if by_arm else ""),
                 fontsize=11, color=C_TEXT)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="./features")
    parser.add_argument("--dataset", "-d", default="tokyo247")
    args = parser.parse_args()

    out_dir, results, sources = load_results(args.feature_dir, args.dataset)
    print("runs: " + ", ".join(f"{m}" + (f"/{s}" if s else "") for m, s in results) + "\n")
    table = write_table(results, os.path.join(out_dir, "comparison.md"), args.dataset, sources)
    figure(results, os.path.join(out_dir, "comparison.png"), args.dataset, sources)
    made = ["comparison.md", "comparison.png"]
    if len([s for s in sources if s is not None]) > 1:
        figure_arms(results, os.path.join(out_dir, "comparison_arms.png"), args.dataset,
                    sources)
        made.append("comparison_arms.png")
    print(table)
    print(f"-> {out_dir}/" + ", ".join(made))


if __name__ == "__main__":
    main()
