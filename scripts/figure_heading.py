"""Viewpoint/heading selectivity: the failure mode our model and the baselines all share.

    pixi run python3 scripts/nsavp_provenance.py
    pixi run python3 scripts/figure_heading.py

Two provenance-clean panels:

(a) Springfield, our model (v8-B): recall broken out by the query camera's direction
    relative to the mapped route — with-route, cross, against — within each sweep. By day
    recall falls monotonically as the heading turns away from the mapped one (against-route
    roughly halves it); by night every direction floors together, because darkness is a
    separate, illumination-limited failure mode. This is our residual error, characterised.

(b) NSAVP, every method: each method's published same-heading cell against the same query
    scored on the *opposite*-heading reference (identical route, identical season gap, only
    the heading flips). The flip is catastrophic for all — the strong RGB controls and the
    event-trained model alike drop to ~0.1 — so viewpoint change is a regime no current
    method has addressed, not a place our model is uniquely strong. The annotation is each
    method's revealed heading preference from a balanced two-heading gallery.

Springfield numbers come from springfield_diag's v8-B analysis JSON (its banks are the full
grid, so the breakdown is exact). The multi-method Springfield offset curve is deliberately
NOT drawn: the baseline query banks on disk are a later reversal-excluded subset, so they
cannot reproduce those methods' published recall — the cross-method comparison lives on
NSAVP, where every bank is gated against the ledger.
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DIAG = os.path.join("output", "springfield_diag")
V8_ANALYSIS = ("analysis_b_v8_accum_s750_s750_3f4eb54780_accumulate_r322_hp1"
               "_baoff.json")
PROVENANCE = os.path.join("output", "heading", "nsavp_provenance.json")
OURS = "megaevent v8-B"
SWEEPS = ["day", "dawn", "night"]
# with-route vs against-route: the well-sampled viewpoint contrast (~12 passes each). The
# "cross" cells are dropped from the figure — they hold 3-7 passes and only add noise; the
# ordinal with->against pair carries the heading story cleanly. A sequential (not
# categorical) pair, because against is "more heading difference", not a separate identity.
DIRS = [("with", "#9ecae1", "with route"),
        ("against", "#08519c", "against route")]


def panel_springfield(ax):
    with open(os.path.join(DIAG, V8_ANALYSIS)) as handle:
        grid = json.load(handle)["grid"]
    x = np.arange(len(SWEEPS))
    w = 0.34
    for k, (d, color, label) in enumerate(DIRS):
        vals, ns = [], []
        for sw in SWEEPS:
            cell = grid.get(f"{sw}/{d}")
            vals.append(cell["mean_r1"] if cell else np.nan)
            ns.append(cell["n"] if cell else 0)
        xs = x + (k - 0.5) * w
        ax.bar(xs, vals, w, color=color, edgecolor="white", linewidth=0.8, label=label)
        for xi, v, n in zip(xs, vals, ns):
            if not np.isnan(v):
                ax.text(xi, v + 0.008, f"n={n}", ha="center", va="bottom", fontsize=6.5,
                        color="#888888")
    ax.set_xticks(x)
    ax.set_xticklabels(SWEEPS, fontsize=9)
    ax.set_ylabel("R@1", fontsize=9)
    ax.set_ylim(0, 0.58)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(loc="upper right", fontsize=7.5, frameon=False, title="query heading",
              title_fontsize=7.5)
    ax.set_title("(a) Springfield (v8-B): recall by query heading", fontsize=9.5)


def panel_nsavp(ax, query):
    with open(PROVENANCE) as handle:
        doc = json.load(handle)
    rows = []
    for name, entry in doc["results"].items():
        cell = entry[query]
        rows.append((name, float(cell["same"]["recall"]["1"]),
                     float(cell["cross"]["recall"]["1"]),
                     cell["balanced"]["top1_from_same_heading"]))
    rows.sort(key=lambda r: r[1])
    ys = np.arange(len(rows))
    for y, (name, same, cross, pref) in zip(ys, rows):
        ours = name == OURS
        color = "#08519c" if ours else "#9e9e9e"
        ax.plot([cross, same], [y, y], color=color,
                linewidth=2.2 if ours else 1.4, zorder=2 if ours else 1)
        ax.plot(same, y, "o", color=color, markersize=6.5 if ours else 5, zorder=3)
        ax.plot(cross, y, "o", markerfacecolor="white", markeredgecolor=color,
                markersize=6.5 if ours else 5, zorder=3)
        ax.annotate(f"{pref:.0%}", (1.01, y), xycoords=("axes fraction", "data"),
                    fontsize=7, va="center", color="#666666")
    ax.set_yticks(ys)
    labels = ["ours (v8-B)" if n == OURS else
              "ours (v8-S)" if n == "megaevent v8-S" else n for n, *_ in rows]
    ax.set_yticklabels(labels, fontsize=8)
    for tick, (name, *_) in zip(ax.get_yticklabels(), rows):
        if name == OURS:
            tick.set_fontweight("bold")
    ax.set_xlabel("R@1", fontsize=9)
    ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.annotate("top-1 from\nsame heading", (1.005, 1.015),
                xycoords="axes fraction", fontsize=7, color="#666666", va="bottom")
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color="#444444", markersize=6,
               label="same-heading map"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor="#444444", markersize=6, label="opposite-heading map")],
        loc="lower right", fontsize=7.5, frameon=False)
    ax.set_title("(b) NSAVP: recall collapses when the map is the other way",
                 fontsize=9.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--query", default="R0_FA0", choices=["R0_FA0", "R0_RA0"])
    ap.add_argument("--out", default=os.path.join("output", "heading", "figure_heading"))
    cli = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6),
                             gridspec_kw={"width_ratios": [0.85, 1.2]})
    panel_springfield(axes[0])
    panel_nsavp(axes[1], cli.query)
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.16, top=0.9, wspace=0.5)

    os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{cli.out}.{ext}", dpi=200)
        print(f"-> {cli.out}.{ext}")


if __name__ == "__main__":
    main()
