"""Failure-taxonomy figure: how each method's queries divide across the pairwise cells.

    pixi run python3 scripts/failure_taxonomy.py --dataset brisbane_event
    pixi run python3 scripts/failure_taxonomy.py --dataset nsavp
    pixi run python3 scripts/figure_failure_taxonomy.py

One stacked bar per (cell, method): correct at the base in a recessive gray, then the
failure classes in a single deepening hue — outranked@5 (recoverable at R@5),
outranked@100 (recoverable by any shortlist stage), and "nowhere" (the place is not
represented; nothing downstream can recover it). Severity is a magnitude, so it gets a
sequential ramp, not per-class hues; the methods are named under every bar, so no
categorical palette is needed at all.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

IN_DIR = os.path.join("output", "failure_taxonomy")
DATASETS = ("brisbane_event", "nsavp")
CELL_LABELS = {
    "sunset1->daytime": "sunset$\\rightarrow$day",
    "sunset1->morning": "sunset$\\rightarrow$morning",
    "sunset1->sunrise": "sunset$\\rightarrow$sunrise",
    "R0_FS0->R0_FA0": "snow$\\rightarrow$aft. (fwd)",
    "R0_RS0->R0_RA0": "snow$\\rightarrow$aft. (rev)",
}
METHOD_LABELS = {"megaevent v8-B": "ours", "MegaLoc": "MegaLoc",
                 "MixVPR": "MixVPR", "QAA": "QAA"}
# correct is recessive; the three failure classes deepen in one hue (sequential = severity).
CLASSES = [("correct", "#e7e7e7", "correct"),
           ("outranked@5", "#fdd0a2", "outranked (rank $\\leq$5)"),
           ("outranked@100", "#f58231", "outranked (rank 6–100)"),
           ("nowhere", "#8c2d04", "not represented ($>$100)")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(IN_DIR, "failure_taxonomy"))
    cli = ap.parse_args()

    docs = {}
    for ds in DATASETS:
        with open(os.path.join(IN_DIR, f"{ds}.json")) as handle:
            docs[ds] = json.load(handle)

    cells = [(ds, cell) for ds in DATASETS for cell in docs[ds]["results"]]
    methods = docs[DATASETS[0]]["methods"]

    group_w, bar_w = len(methods) + 1.2, 0.92
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    xticks, xlabels = [], []
    for g, (ds, cell) in enumerate(cells):
        stats = docs[ds]["results"][cell]
        for j, m in enumerate(methods):
            x = g * group_w + j
            base = 0.0
            for key, color, _ in CLASSES:
                v = stats[m]["classes"][key]
                ax.bar(x, v, bar_w, bottom=base, color=color,
                       edgecolor="white", linewidth=0.8)
                base += v
            xticks.append(x)
            xlabels.append(METHOD_LABELS.get(m, m))
        ax.text(g * group_w + (len(methods) - 1) / 2, -0.30,
                CELL_LABELS.get(cell, cell), ha="center", va="top", fontsize=9)

    for g in range(1, len(cells)):
        ax.axvline(g * group_w - 1.1, color="#cccccc", linewidth=0.7)
    # Brisbane | NSAVP divider a touch heavier
    n_bris = sum(1 for ds, _ in cells if ds == "brisbane_event")
    ax.axvline(n_bris * group_w - 1.1, color="#888888", linewidth=1.1)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("fraction of queries", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.8, (len(cells) - 1) * group_w + len(methods) - 0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(handles=[Patch(facecolor=c, label=lab) for _, c, lab in CLASSES],
              loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, fontsize=8,
              frameon=False)
    fig.text(0.34, 0.015, "Brisbane-Event (illumination)", ha="center", fontsize=9,
             style="italic")
    fig.text(0.81, 0.015, "NSAVP (season, per heading)", ha="center", fontsize=9,
             style="italic")
    fig.subplots_adjust(bottom=0.32, top=0.88, left=0.07, right=0.99)

    for ext in ("pdf", "png"):
        fig.savefig(f"{cli.out}.{ext}", dpi=200)
        print(f"-> {cli.out}.{ext}")


if __name__ == "__main__":
    main()
