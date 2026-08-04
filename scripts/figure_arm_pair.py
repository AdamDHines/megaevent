"""Side-by-side countmask frames from both arms of the real-vs-I2E ablation.

Run from the repo root::

    pixi run python3 scripts/figure_arm_pair.py --seq sunset1

The recall table says *how much* the synthetic conversion costs each method. This says
*what the methods are actually looking at* — the same instant on the same route, rendered
from the DAVIS's real event stream on one row and from an I2E micro-saccade over the
DAVIS's intensity frame on the other. It is the fastest way to see why the deltas come out
the way they do, and the one panel that makes the setup legible to someone who has not read
the pipeline.

The index alignment is what makes the pairing meaningful: column *i* is the same moment in
both rows, to within half a slice, because ``scripts/extract_aps.py`` chose each intensity
frame by nearest slice centre.

Frames are rendered through :func:`src.npzdata.load_countmask`, the same representation
megaevent consumes, so the panel shows the model's actual input rather than a prettier
stand-in.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import C_MUTED, C_TEXT                                   # noqa: E402
from src.npzdata import read_events                                         # noqa: E402
from src.scoring import display_frame                                       # noqa: E402

ROWS = (("real", "real events (DAVIS event stream, 50 ms)"),
        ("i2e", "I2E events (micro-saccade over the DAVIS intensity frame)"))


def pick(npz_root, dataset, seq, n, skip_edges=0.05):
    """``n`` evenly spaced frame indices that both arms have and the alignment kept.

    Edges are trimmed because the excluded indices cluster there, and a montage of the two
    frames the alignment could not pair is not what anyone wants to look at.
    """
    real = os.path.join(npz_root, dataset, "real", seq)
    total = len([f for f in os.listdir(real) if f.startswith("frame_")])
    with open(os.path.join(npz_root, dataset, "aps", seq, "select.json")) as f:
        bad = set(json.load(f)["out_of_tolerance"])
    lo, hi = int(skip_edges * total), int((1 - skip_edges) * total)
    want = np.linspace(lo, hi, n).astype(int)
    return [int(i) for i in want if i not in bad][:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", default="sunset1")
    ap.add_argument("--dataset", default="brisbane_event")
    ap.add_argument("--npz-root", default="/media/adam/vprdatasets/megaevent/brisbane_npz")
    ap.add_argument("--out", default=None,
                    help="default: <feature-dir>/<dataset>/arm_pair_<seq>.png")
    ap.add_argument("--feature-dir",
                    default="/media/adam/vprdatasets/megaevent/brisbane_ablation/features")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    idx = pick(args.npz_root, args.dataset, args.seq, args.n)
    out = args.out or os.path.join(args.feature_dir, args.dataset,
                                   f"arm_pair_{args.seq}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    fig, axes = plt.subplots(len(ROWS), len(idx), squeeze=False,
                             figsize=(2.1 * len(idx), 1.85 * len(ROWS) + 0.5),
                             constrained_layout=True)
    for r, (source, label) in enumerate(ROWS):
        base = os.path.join(args.npz_root, args.dataset, source, args.seq)
        for c, i in enumerate(idx):
            path = os.path.join(base, f"frame_{i:06d}.npz")
            ax = axes[r][c]
            ax.imshow(display_frame(path))
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#dcdbd6")
            n_ev = read_events(path)[0].size
            ax.set_xlabel(f"{n_ev / 1e3:.0f}k events", fontsize=7, color=C_MUTED, labelpad=2)
            if c == 0:
                ax.set_ylabel(label, fontsize=7.5, color=C_TEXT, wrap=True)
            if r == 0:
                ax.set_title(f"frame {i}", fontsize=8, color=C_MUTED)

    fig.suptitle(f"{args.dataset} / {args.seq}: the same instants, from both event sources"
                 f"  —  countmask, the representation the model consumes",
                 fontsize=10.5, color=C_TEXT)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"-> {out}  (frames {idx})")


if __name__ == "__main__":
    sys.exit(main())
