"""Brisbane condition-confusion heatmaps: which illumination crossings break which model.

    pixi run python3 scripts/brisbane_confusion.py --roster v8accum \
        --out-json output/confusion/brisbane_v8accum.json
    pixi run python3 scripts/figure_confusion.py

One panel per method: the full query-traverse x database-traverse R@1 grid from the
confusion JSON, single-traverse cells only (the POOLED column is the retired protocol and
is dropped). Rows and columns are ordered by illumination — daytime, morning, sunrise,
sunset, night — so distance from the diagonal reads as the size of the illumination gap.
The three published pairwise cells (reference sunset1, queries daytime/morning/sunrise)
are outlined. One hue, shared scale: R@1 is a magnitude, so the map is sequential.
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# Illumination order, bright to dark: the diagonal band structure is the point.
ORDER = ["daytime", "morning", "sunrise", "sunset1", "night"]
SHORT = {"daytime": "day", "morning": "morn", "sunrise": "sunrise",
         "sunset1": "sunset", "night": "night"}
PUBLISHED = [("daytime", "sunset1"), ("morning", "sunset1"), ("sunrise", "sunset1")]
DEFAULT_METHODS = ["ours v8-B", "MegaLoc", "QAA"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in-json", default="output/confusion/brisbane_v8accum.json")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--out", default="output/confusion/figure_confusion")
    cli = ap.parse_args()

    with open(cli.in_json) as handle:
        doc = json.load(handle)

    n = len(cli.methods)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n + 0.9, 3.3))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, cli.methods):
        grid = np.full((len(ORDER), len(ORDER)), np.nan)
        for i, q in enumerate(ORDER):
            for j, db in enumerate(ORDER):
                if q != db:
                    grid[i, j] = doc["results"][q][db][name]
        im = ax.imshow(grid, cmap="Blues", vmin=0.0, vmax=1.0)
        for i in range(len(ORDER)):
            for j in range(len(ORDER)):
                if i == j:
                    continue
                v = grid[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.6 else "#1a1a1a")
        for q, db in PUBLISHED:
            ax.add_patch(Rectangle((ORDER.index(db) - 0.5, ORDER.index(q) - 0.5), 1, 1,
                                   fill=False, edgecolor="#c1272d", linewidth=1.6))
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([SHORT[t] for t in ORDER], fontsize=8, rotation=45,
                           ha="right")
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([SHORT[t] for t in ORDER] if ax is axes[0] else [],
                           fontsize=8)
        ax.set_title("ours (v8-B)" if name == "ours v8-B" else name, fontsize=10)
        ax.set_xlabel("reference traverse", fontsize=8.5)
        if ax is axes[0]:
            ax.set_ylabel("query traverse", fontsize=8.5)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.02)
    cbar.set_label("R@1", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    for ext in ("pdf", "png"):
        fig.savefig(f"{cli.out}.{ext}", dpi=200, bbox_inches="tight")
        print(f"-> {cli.out}.{ext}")


if __name__ == "__main__":
    main()
